import numpy as np, pytest
from conftest import TopicEmbedder
from holonomic import HolonomicMemory, HashEmbedder
from holonomic import vsa
from holonomic.engine import split_text


def mem(tmp_path, **kw):
    return HolonomicMemory(tmp_path / "m", kw.pop("embedder", HashEmbedder()), **kw)


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
