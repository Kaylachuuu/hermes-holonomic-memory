import time
import numpy as np, pytest
from conftest import TopicEmbedder, track
from holonomic import HolonomicMemory, HashEmbedder
from holonomic import vsa
from holonomic.engine import split_text


def mem(tmp_path, **kw):
    return track(HolonomicMemory(tmp_path / "m", kw.pop("embedder", HashEmbedder()), **kw))


def test_algebra_roundtrip():
    P = vsa.make_projection(64, 2048)
    proj = vsa.Projector(P, np.zeros(64, dtype=np.float32))
    rng = np.random.default_rng(0)
    x = proj.prep(rng.standard_normal((3, 64)))
    p = proj.phasor(x)
    assert np.allclose(np.abs(p), 1.0, atol=1e-5)
    back = proj.back(p)
    cos = np.sum(back * x, axis=1) / np.linalg.norm(back, axis=1)
    assert (cos > 0.95).all()
    key = vsa.atom("k", 2048)
    plate = key * p[0] + vsa.atom("other", 2048) * p[1]
    est = proj.back((plate * np.conj(key))[None, :])[0]
    assert est @ x[0] > 0.8 and abs(est @ x[2]) < 0.2
    assert np.array_equal(vsa.atom("k", 2048), vsa.atom("k", 2048))
    assert sorted(vsa.permutation("cue", 2048)) == list(range(2048))


def test_direct_recall(tmp_path):
    m = mem(tmp_path)
    m.remember("Kayla prefers dark roast coffee in the morning", session="a")
    m.remember("The V100 graphics card runs the Gemma model", session="b")
    hits = m.recall("what coffee does Kayla prefer")
    assert hits and "coffee" in hits[0].text and hits[0].direct > 0.3


def test_associative_recall_finds_dissimilar_neighbour(tmp_path):
    m = mem(tmp_path)
    m.remember("Mara moved to Lisbon last spring", session="s1")
    target = m.remember("Shellfish gives her a dangerous allergic reaction", session="s1")[0]
    for i in range(30):
        m.remember(f"unrelated filler note number{i} about gardening topic{i}", session=f"f{i}")
    hits = {h.id: h for h in m.recall("Lisbon Mara moved", k=5)}
    assert target in hits, "the neighbouring memory should surface through the plate"
    assert hits[target].direct < 0.25 and hits[target].assoc > 0.3


def test_key_probe(tmp_path):
    m = mem(tmp_path)
    a = m.remember("She plays the cello", keys=["Mara"], session="x")[0]
    m.remember("The server needs more RAM", keys=["anubis"], session="y")
    hits = m.recall(keys=["  mara "], k=3)
    assert [h.id for h in hits] == [a]


def test_links_connect_reflection_to_sources(tmp_path):
    m = mem(tmp_path)
    src = m.remember("Kayla asked many careful questions before deciding", session="s")[0]
    for i in range(10):
        m.remember(f"noise entry number{i} zebra{i}", session=f"n{i}")
    ref = m.remember("Thoroughness matters to her; rushing would erode confidence", kind="reflection",
                     session="reflect", links=[src])[0]
    hits = {h.id: h for h in m.recall("careful questions before deciding", k=4)}
    assert ref in hits and hits[ref].assoc > 0.3
    assert src in {h.id for h in m.associates(ref, k=4)}


def test_realms_are_isolated(tmp_path):
    m = mem(tmp_path)
    m.remember("We walked along the river yesterday", session="s")
    d = m.remember("The river turned into glass and we walked on it", realm="dream", kind="dream", session="d")[0]
    assert d not in {h.id for h in m.recall("walked river", k=10)}
    assert d in {h.id for h in m.recall("walked river", k=10, realms=("dream",))}
    assert {h.realm for h in m.recall("walked river", k=10, realms=("waking", "dream"))} == {"waking", "dream"}


def test_persistence(tmp_path):
    m = mem(tmp_path)
    m.remember("Mara moved to Lisbon last spring", session="s1")
    m.remember("Shellfish gives her a dangerous allergic reaction", session="s1", keys=["mara"])
    before = [(h.id, round(h.score, 4), round(h.assoc, 4)) for h in m.recall("Lisbon Mara moved", k=5)]
    stats = m.stats()
    m.close()
    m2 = mem(tmp_path)
    assert m2.stats() == stats
    assert [(h.id, round(h.score, 4), round(h.assoc, 4)) for h in m2.recall("Lisbon Mara moved", k=5)] == before
    # the chain continues across restarts
    nxt = m2.remember("Her doctor gave her an epinephrine pen", session="s1")[0]
    assert nxt in {h.id for h in m2.recall("dangerous allergic reaction shellfish", k=5)}
    m2.close()
    with pytest.raises(ValueError):
        HolonomicMemory(tmp_path / "m", HashEmbedder(dim=128))


def test_fragment_of_plate_still_recalls(tmp_path):
    m = mem(tmp_path)
    m.remember("Mara moved to Lisbon last spring", session="s1")
    t = m.remember("Shellfish gives her a dangerous allergic reaction", session="s1")[0]
    for frac in (1.0, 0.5, 0.25):
        hits = {h.id: h for h in m.recall("Lisbon Mara moved", k=5, aperture=frac)}
        assert t in hits and hits[t].assoc > 0.25, frac


def test_forget_reinforce_decay(tmp_path):
    m = mem(tmp_path)
    a = m.remember("Kayla prefers dark roast coffee", session="a")[0]
    b = m.remember("Kayla prefers dark roast coffee beans from Ethiopia", session="b")[0]
    m.reinforce([b], 1.0)
    assert m.recall("dark roast coffee Kayla prefers beans Ethiopia", k=2)[0].id == b
    m.decay(0.5)
    assert m.get(a)["strength"] == pytest.approx(0.5)
    assert m.forget(a) and not m.forget(a)
    assert a not in {h.id for h in m.recall("dark roast coffee", k=5)}
    assert m.get(a) is None and m.stats()["memories"] == 1


def test_long_text_is_chunked_and_chained(tmp_path):
    m = mem(tmp_path, max_chars=120)
    text = "Alpha section talks about holograms and lasers. " * 3 + "\n\n" + "Beta section talks about violins and orchestras. " * 3
    ids = m.remember(text, session="doc")
    assert len(ids) >= 2 and all(len(c) <= 120 for c in split_text(text, 120))
    hits = {h.id: h for h in m.recall("holograms lasers", k=5)}
    assert set(ids) <= set(hits)          # later chunks arrive by association


def test_plates_seal_and_stay_usable(tmp_path):
    m = mem(tmp_path, embedder=TopicEmbedder(), plate_capacity=32)
    pairs = []
    for i in range(120):
        a = m.remember(f"topic{i} first half", session=f"s{i}")[0]
        b = m.remember(f"other{i} second half", session=f"s{i}")[0]
        pairs.append((f"topic{i} first half", b))
    assert m.stats()["plates"] > 5
    found = sum(b in {h.id for h in m.recall(q, k=4)} for q, b in pairs)
    assert found >= 0.95 * len(pairs)


def test_ollama_embedder_against_stub_server():
    import json, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from holonomic import OllamaEmbedder, EmbeddingError
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            out = json.dumps({"embeddings": [[float(len(t)), 1.0, 0.0] for t in body["input"]]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a): pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        e = OllamaEmbedder("nomic-embed-text:latest", host=f"http://127.0.0.1:{srv.server_port}")
        v = e.embed(["hello", "hi"], "query")
        assert v.shape == (2, 3) and e.dim == 3 and e.signature == "ollama:nomic-embed-text"
        assert seen[0][0] == "/api/embed" and seen[0][1]["input"] == ["search_query: hello", "search_query: hi"]
        assert e.embed(["doc"], "document")[0, 0] == len("search_document: doc")
    finally:
        srv.shutdown()
    with pytest.raises(EmbeddingError):
        OllamaEmbedder(host="http://127.0.0.1:9", timeout=1).embed(["x"])


def test_echoes_look_at_age_before_taking_the_best_few(tmp_path):
    """The likeliest candidates were taken first and their age checked afterwards.  With many recent memories
    like the seed, they filled every place, and nothing older came back though something older fitted."""
    import time
    from conftest import track
    from holonomic import HolonomicMemory, HashEmbedder
    m = track(HolonomicMemory(tmp_path / "m", HashEmbedder()))
    now = time.time()
    old = m.remember("the lake at dusk with a heron standing in the shallows", kind="said_user", session="old", created_at=now - 30 * 86400)[0]
    older_question = m.remember("the lake at dusk with a heron and a boat", kind="asked_user", session="old", created_at=now - 30 * 86400)[0]
    recent = [m.remember(f"the lake at dusk with a heron standing near reed bed number {i}", kind="said_user", session=f"r{i}", created_at=now - 3600)[0]
              for i in range(30)]
    seed = m.remember("the lake at dusk with a heron standing quite still", kind="said_user", session="now", created_at=now)[0]
    got = m.echoes(seed, older_than=now - 86400, k=2, low=0.05, high=0.999)
    assert [e["id"] for e in got][:1] in ([old], [older_question]) and {e["id"] for e in got} == {old, older_question}
    assert not set(recent) & {e["id"] for e in got}
    # a kind the caller will not use is left out before the best are taken, not after
    assert [e["id"] for e in m.echoes(seed, older_than=now - 86400, k=1, low=0.05, high=0.999, skip_kinds=("asked_user",))] == [old]
    assert m.echoes(seed, older_than=now - 365 * 86400, k=2, low=0.05, high=0.999) == []      # nothing that old


def test_a_plate_is_not_written_past_its_capacity(tmp_path):
    """Capacity was a soft line: the room a write needed was estimated as though its terms were unrelated, and
    the real energy measured afterwards.  Terms that agree add up to more, and a plate could end above it."""
    from conftest import track
    from holonomic import HolonomicMemory, HashEmbedder
    m = track(HolonomicMemory(tmp_path / "m", HashEmbedder(), plate_capacity=24.0))
    hub = m.remember("Kayla has a cat named Sushi", kind="said_user", session="s")[0]
    for i in range(60):            # many near-identical conclusions bound to one source: the terms agree
        m.remember(f"Kayla has a cat named Sushi, noted again {i % 3}", kind="fact", session="reflection", chain=False, links=[hub], salience=2.0)
    loads = [float(r["load"]) for r in m._db.execute("SELECT load FROM plates ORDER BY id")]
    assert len(loads) > 3 and max(loads) <= 24.0 + 1e-3
    measured = [float(np.mean(np.abs(np.frombuffer(r["trace"], dtype=np.complex64)) ** 2)) for r in m._db.execute("SELECT trace FROM plates ORDER BY id")]
    assert max(measured) <= 24.0 + 1e-3 and all(abs(a - b) < 1e-2 for a, b in zip(loads, measured))
    # all but the last are closed, and the log still gives every plate back though one memory's terms may lie on two
    assert [r["sealed"] for r in m._db.execute("SELECT sealed FROM plates ORDER BY id")][:-1] == [1] * (len(loads) - 1)
    from holonomic import audit
    v = audit.verify(m)
    assert v["checked"] == v["agree"] == len(loads) and v["worst"] < 1e-4
    assert len({r["plate_id"] for r in m._db.execute("SELECT plate_id FROM bindings")}) == len(loads)
    # recall still works when a conversation runs across many small plates
    lines = [m.remember(f"line {i}: {w} is discussed at length here", kind="said_user", session="talk")[0]
             for i, w in enumerate(["tomatoes", "the lake", "assembly", "the boot sector", "sunsets", "a heron", "graphics cards", "glasses",
                                    "network cables", "the mountain", "fireworks", "a tabby cat", "pine trees", "the airport", "coffee", "the moon"] * 3)]
    found = sum(1 for x, y in zip(lines, lines[1:]) if x in {h.id for h in m.associates(y, k=10)})
    assert found >= 0.9 * (len(lines) - 1) and m.stats()["plates"] > len(loads)
    assert max(float(r["load"]) for r in m._db.execute("SELECT load FROM plates")) <= 24.0 + 1e-3


def test_a_capacity_too_small_for_one_association_is_refused(tmp_path):
    """One association at weight w puts w squared on a plate.  The term-by-term rule lets any term onto an empty
    plate, so a capacity below the largest single term would have been broken by the first bright write."""
    from conftest import track
    from holonomic import HolonomicMemory, HashEmbedder
    from holonomic import engine as eng
    assert eng.MIN_PLATE_CAPACITY == eng.MAX_WEIGHT ** 2 == 4.0
    for bad in (3.9, 0.0, -1.0, float("nan")):
        try:
            HolonomicMemory(tmp_path / f"bad{bad}", HashEmbedder(), plate_capacity=bad)
            raise AssertionError(f"a capacity of {bad} was accepted")
        except ValueError as exc:
            assert "at least 4" in str(exc)
        assert not (tmp_path / f"bad{bad}" / "projection.npy").exists()          # refused before anything was set up
    # the least allowed holds exactly one association at full weight, and is never exceeded
    m = track(HolonomicMemory(tmp_path / "least", HashEmbedder(), plate_capacity=4.0))
    a = m.remember("the first thing said", kind="said_user", session="s", salience=5.0)[0]       # salience is held to 2
    m.remember("the second thing said", kind="said_user", session="s", salience=5.0)
    loads = [float(r["load"]) for r in m._db.execute("SELECT load FROM plates ORDER BY id")]
    assert len(loads) == 2 and max(loads) <= 4.0 + 1e-4
    # a term larger than a whole plate is a fault in the caller: refused, and nothing of it is kept
    row = m._row[a]
    p = m._phasor_of(row)
    before = (m._db.execute("SELECT COUNT(*) FROM bindings").fetchone()[0], [float(r["load"]) for r in m._db.execute("SELECT load FROM plates")])
    m._db.execute("BEGIN IMMEDIATE")
    try:
        m._write("waking", [(m._cue(p), p, row, 3.0, a, None)])
        raise AssertionError("a term of energy 9 went onto a plate that holds 4")
    except ValueError as exc:
        assert "more than a whole plate holds" in str(exc)
    finally:
        m._db.execute("ROLLBACK")
        m._load()
    assert before == (m._db.execute("SELECT COUNT(*) FROM bindings").fetchone()[0], [float(r["load"]) for r in m._db.execute("SELECT load FROM plates")])
    # a store that somehow holds a smaller value says so when it is opened
    m._db.execute("UPDATE meta SET value = '2.0' WHERE key = 'plate_capacity'")
    m.close()
    try:
        HolonomicMemory(tmp_path / "least", HashEmbedder())
        raise AssertionError("a store with capacity 2 was opened")
    except ValueError as exc:
        assert "below the least" in str(exc)


def test_what_is_turned_down_does_not_use_up_the_shortlist(tmp_path):
    """linked() and echoes() took their best k and the caller then turned some down (already used, too short),
    so a usable memory just below them was never seen."""
    m = mem(tmp_path)
    now = time.time()
    old = now - 40 * 86400
    hub = m.remember("We planted tomatoes along the south fence in spring", kind="said_user", session="a", created_at=old)[0]
    tied = [m.remember(t, kind="said_user", session=f"b{i}", chain=False, links=[hub], created_at=old + i)[0]
            for i, t in enumerate(("ok", "The neighbour's dog dug up half of them", "We put wire around the whole bed", "yes"))]
    order = [x["id"] for x in m.linked(hub, k=10)]
    assert set(order) == set(tied)
    first = order[0]
    assert [x["id"] for x in m.linked(hub, k=1)] == [first]
    assert [x["id"] for x in m.linked(hub, k=1, skip_ids={first})] == [order[1]]            # the next one down, not nothing
    long = [i for i in order if len(m.get(i)["text"]) >= 25]
    assert len(long) == 2 and [x["id"] for x in m.linked(hub, k=2, min_chars=25)] == long
    assert [x["id"] for x in m.linked(hub, k=1, min_chars=25, skip_ids={long[0]})] == [long[1]]
    assert m.linked(hub, k=3, skip_ids=set(tied)) == []
    seed = m.remember("The tomatoes along the fence are ripe", kind="said_user", session="now", created_at=now)[0]
    likes = [e["id"] for e in m.echoes(seed, older_than=now - 86400, k=10, low=0.01, high=0.999)]
    assert len(likes) >= 2
    assert [e["id"] for e in m.echoes(seed, older_than=now - 86400, k=1, low=0.01, high=0.999, skip_ids={likes[0]})] == [likes[1]]
