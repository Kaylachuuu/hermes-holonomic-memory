import json, sys

import numpy as np
import pytest

from test_provider import HAVE_HERMES, make
from conftest import track
from holonomic import HolonomicMemory, HashEmbedder
from holonomic import audit, engine as eng


def store(tmp_path, name="m", **kw):
    return track(HolonomicMemory(tmp_path / name, HashEmbedder(), **kw))


def rows(m):
    return [dict(r) for r in m._db.execute("SELECT plate_id, cue_memory, cue_key, target, weight, encoding FROM bindings ORDER BY id")]


def test_every_term_written_to_a_plate_is_in_the_log(tmp_path):
    m = store(tmp_path)
    a = m.remember("My name is Kayla and I work in IT", kind="said_user", session="s1", keys=["Kayla"])[0]
    assert [(r["cue_memory"], r["cue_key"], r["target"]) for r in rows(m)] == [(None, "kayla", a)]
    b = m.remember("I have two cats, Sushi and Theo", kind="said_user", session="s1", salience=1.5)[0]
    c = m.remember("Kayla has two cats.", kind="fact", session="reflection", chain=False, links=[b, a], salience=1.1)[0]
    log = rows(m)
    assert [(r["cue_memory"], r["target"]) for r in log[1:3]] == [(a, b), (b, a)]            # the line before recalls this one, and this one it
    assert {(r["cue_memory"], r["target"]) for r in log[3:]} == {(b, c), (c, b), (a, c), (c, a)}
    assert log[1]["weight"] == 1.5 and log[3]["weight"] == pytest.approx(1.1) and {r["encoding"] for r in log} == {m._encoding}
    st = audit.log_status(m)
    assert st["bindings"] == 7 and st["plates"] == st["plates_logged"] == 1 and st["encodings"] == 1 and st["log_started"]
    # The log alone gives the plate back.
    v = audit.verify(m)
    assert v["checked"] == v["agree"] == 1 and v["worst"] < 1e-5 and not v["disagree"] and not v["cannot"] and v["unlogged"] == 0


def test_a_write_can_be_reproduced_from_what_the_store_keeps(tmp_path):
    """The phasor used to be made from the full-precision embedding; the store keeps half precision, so a write
    could not be made again exactly.  It is now made from the stored vector."""
    m = store(tmp_path)
    for i in range(30):
        m.remember(f"Line {i} of the conversation is about topic {i % 5} and thing {i}", kind="said_user", session=f"s{i % 3}", keys=[f"topic{i % 5}"])
    assert audit.verify(m)["worst"] < 1e-5
    m.close()
    again = store(tmp_path)                                    # from the database alone, in another process's shoes
    v = audit.verify(again)
    assert v["checked"] == v["agree"] >= 1 and v["worst"] < 1e-5 and audit.log_status(again)["encodings"] == 1
    row = again._row[1]
    kept = np.frombuffer(again._db.execute("SELECT vec FROM memories WHERE id = 1").fetchone()["vec"], dtype=np.float16).astype(np.float32)
    assert np.array_equal(again._X.a[row], kept)
    # Tampering with a plate, or with the log, is seen.
    pid = int(again._pid.a[0])
    trace = again._trace.a[0].copy(); trace[:40] += 1.0
    again._db.execute("UPDATE plates SET trace = ? WHERE id = ?", (trace.tobytes(), pid))
    bad = audit.verify(again)
    assert bad["agree"] == bad["checked"] - 1 and bad["disagree"][0]["plate"] == pid


def test_the_log_and_the_plate_are_written_together_or_not_at_all(tmp_path):
    m = store(tmp_path)
    m.remember("The first line goes in whole", kind="said_user", session="s1")
    m.remember("And a second one after it", kind="said_user", session="s1")
    before = (len(rows(m)), m.stats()["memories"], m._trace.a[0].copy())
    real, calls = m._key_atom, []

    def flaky(key):
        calls.append(key)
        if len(calls) > 1:
            raise RuntimeError("the power went out")
        return real(key)
    m._key_atom = flaky
    with pytest.raises(RuntimeError):
        m.remember("One sentence is stored. Then a second one fails.", kind="said_user", session="s1", keys=["power"], sentences=True)
    m._key_atom = real
    assert len(rows(m)) == before[0] and m.stats()["memories"] == before[1]                  # nothing of the failed write is in the log
    assert np.array_equal(m._trace.a[0], before[2]) and audit.verify(m)["agree"] == 1        # nor on the plate


def test_a_store_from_before_the_log_is_closed_off_cleanly(tmp_path):
    m = store(tmp_path)
    for i in range(6):
        m.remember(f"An old line number {i} about the lake", kind="said_user", session="old")
    # make it look as it did before this version
    m._db.execute("DELETE FROM bindings"); m._db.execute("DELETE FROM encodings")
    m._db.execute("UPDATE plates SET logged = 0, sealed = 0"); m._db.execute("DELETE FROM meta WHERE key = 'log_started'")
    m._db.execute("ALTER TABLE plates DROP COLUMN logged"); m._db.execute("ALTER TABLE plate_members DROP COLUMN gone")
    old_plates = m.stats()["plates"]
    m.close()
    again = store(tmp_path)
    assert [dict(r) for r in again._db.execute("SELECT sealed, logged FROM plates")] == [{"sealed": 1, "logged": 0}] * old_plates
    new = again.remember("A new line about the lake at dusk", kind="said_user", session="old")[0]
    assert again.stats()["plates"] == old_plates + 1                                           # no plate holds both kinds of write
    assert {r["plate_id"] for r in rows(again)} == {int(again._pid.a[old_plates])}
    v = audit.verify(again)
    assert v["unlogged"] == old_plates and v["checked"] == v["agree"] == 1
    # old associations still work, and the new line is tied to the old conversation
    assert new - 1 in {h.id for h in again.associates(new)} and 2 in {h.id for h in again.associates(1)}
    again.close()
    third = store(tmp_path)                                    # opening it again closes nothing more
    assert third.stats()["plates"] == old_plates + 1 and [r["sealed"] for r in third._db.execute("SELECT sealed FROM plates ORDER BY id")][-1] == 0


def test_forgetting_keeps_a_record_of_where_the_traces_are(tmp_path):
    m = store(tmp_path)
    a = m.remember("Kayla has a cat named Sushi", kind="said_user", session="s1")[0]
    b = m.remember("Sushi is a long-haired tuxedo cat", kind="said_user", session="s1")[0]
    assert m.forget(a)
    assert [dict(r) for r in m._db.execute("SELECT memory_id, gone FROM plate_members ORDER BY memory_id")] == [
        {"memory_id": a, "gone": 1}, {"memory_id": b, "gone": 0}]
    v = audit.verify(m)                                        # its vector is gone, so its terms cannot be made again
    assert v["checked"] == 0 and v["cannot"][0]["forgotten"] == 2
    m.close()
    again = store(tmp_path)
    assert b in again._row and a not in again._row and all(a not in {int(again._ids.a[r]) for r in mem} for mem in again._members)


def test_an_encoding_is_named_and_a_changed_one_is_another(tmp_path):
    m = store(tmp_path)
    m.remember("One line under the first encoding", kind="said_user", session="s1")
    m.remember("And another to bind it to", kind="said_user", session="s1")
    first, mark = m._encoding, m._fingerprint()
    m.close()
    assert store(tmp_path)._fingerprint() == mark
    eng_formula = eng.CUE_FORMULA
    try:
        eng.CUE_FORMULA = "something-else/2"
        other = store(tmp_path)
        assert other._encoding != first and audit.log_status(other)["encodings"] == 2
        assert audit.verify(other)["cannot"][0]["other_encoding"] == 2                         # not silently rebuilt the wrong way
    finally:
        eng.CUE_FORMULA = eng_formula


def test_what_the_plates_do_at_recall_is_measured_without_changing_anything(tmp_path):
    m = store(tmp_path)
    said = [m.remember(t, kind="said_user", session="s1")[0] for t in (
        "I have a cat named Sushi who sleeps on the couch", "My other cat Theo is a brown tabby", "I work as a network administrator",
        "The lake photo was taken at sunrise", "My operating system is written in assembly")]
    facts = [m.remember(t, kind="fact", session="reflection", chain=False, links=[s])[0] for t, s in (
        ("Kayla has a cat named Sushi.", said[0]), ("Kayla has a cat named Sushi who sleeps on the couch.", said[0]),
        ("Kayla has a cat named Theo.", said[1]), ("Kayla is a network administrator.", said[2]))]
    m._db.execute("UPDATE memories SET meta = ? WHERE id = ?", (json.dumps({"sources": [said[0]]}), facts[1]))
    m.supersede(facts[0], facts[1], reason="said the same")
    m.forget(said[3])
    before = (m.stats()["memories"], len(rows(m)), m._trace.a[0].copy())
    d = audit.diagnose(m, sample=50)
    assert (m.stats()["memories"], len(rows(m))) == before[:2] and np.array_equal(m._trace.a[0], before[2])       # read-only
    assert d["cues"] == d["live"] == 7 and d["out_of_recall"] == 1
    assert d["plates"]["members_out_of_recall"] == 1 and d["plates"]["with_out_of_recall"] == 1
    assert d["forgotten"] == {"total": 1, "on_record": 1, "plates_unknown": 0}
    assert d["resonance"]["limit"] == 24 and d["resonance"]["most"] >= 1 and d["resonance"]["cues_with_plates_turned_away"] == 0
    r = d["recovery"]
    assert r["as it is"]["expected"] == r["without them"]["expected"] == r["no limit"]["expected"] > 0
    assert 0 < r["as it is"]["found"] <= r["as it is"]["expected"] and r["no limit"]["found"] == r["as it is"]["found"]
    assert all(r[w]["unexplained"] <= r[w]["returned"] for w in r) and d["known_links"]["cues_with_any"] >= 6
    assert all(r[w]["found"] + r[w]["explained"] + r[w]["conversation"] + r[w]["related"] + r[w]["unexplained"] == r[w]["returned"]
               for w in r)                                                                              # each return is sorted once
    # by the kind of plate read from: every plate here has all its writes on record
    assert d["plates"]["logged"] == d["plates"]["count"] and d["returned_by_plate"]["legacy"] == 0
    assert d["unexplained_by_plate"]["legacy"] == 0 and d["unexplained_by_plate"]["logged"] == r["as it is"]["unexplained"]
    assert d["returned_by_plate"]["logged"] == r["as it is"]["returned"]
    # how strongly each kind comes back
    assert d["strength"]["known"]["count"] == r["as it is"]["found"] and d["strength"]["known"]["median"] > 0
    # and what she would actually be given
    g = d["given"]
    assert g["cues"] == 7 and g["given"] == g["by_likeness"] + g["by_association"] and g["given"] <= 7 * 6
    assert g["known"] + g["explained"] + g["conversation"] + g["related"] + g["unexplained"] == g["by_association"]
    assert g["unexplained_logged"] >= g["unexplained"] and g["unexplained_legacy"] == 0
    assert all(set(e) == {"cue", "given", "read_from"} for e in g["examples"])
    # the retired twin is never counted as something to find, nor returned
    retired_seen = [e for e in d["examples"] if facts[0] in e["gained"] + e["lost"]]
    assert not retired_seen and audit.diagnose(store(tmp_path, "empty"))["cues"] == 0


def test_cli_plates(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io, types
    p = make(tmp_path)
    p.sync_turn("My name is Kayla and I have a cat named Sushi", "Nice to meet you, Kayla. Sushi sounds lovely.", session_id="s1")
    p.sync_turn("He is a long-haired tuxedo cat", "A handsome one, then.", session_id="s1")
    p.shutdown()
    home = tmp_path / "home"
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    try:
        import holonomic.cli as cli, holonomic.embed as embed
        real = embed.OllamaEmbedder
        embed.OllamaEmbedder = lambda *a, **k: HashEmbedder()
        parser = argparse.ArgumentParser(); cli.register_cli(parser)

        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                args = parser.parse_args(list(argv)); args.func(args)
            return out.getvalue()
        text = run("plates")
        assert "have every write in the write log" in text and "1 of 1 plate(s) agree" in text and "hermes holonomic plates check" in text
        text = run("plates", "check", "-n", "20")
        assert "How many plates answer a cue" in text and "as recall does it now" in text and "with no limit on plates read" in text
        assert "More returned is not better in itself" in text and "What she would be given" in text
        assert "from plates with every write on record" in text and "That is not the same as wrong" in text
    finally:
        embed.OllamaEmbedder = real
        sys.modules.pop("hermes_constants", None)


def test_returns_are_told_apart_by_whether_their_plate_is_on_record(tmp_path):
    """An unexplained return from a plate written before the log may only be a missing record; from a plate with
    every write on record, nothing was written that would explain it."""
    m = store(tmp_path)
    old = [m.remember(f"Kayla has a cat and the cat sleeps on the couch, take {i}", kind="said_user", session=f"s{i}")[0] for i in range(8)]
    for i in range(0, 8, 2):
        m.remember(f"Kayla has a cat that sleeps on the couch, conclusion {i}", kind="fact", session="reflection", chain=False, links=[old[i]])
    # these plates are from before the log: no rows, not marked, closed
    m._db.execute("DELETE FROM bindings"); m._db.execute("UPDATE plates SET logged = 0, sealed = 1")
    m.close()
    m = store(tmp_path)
    new = [m.remember(f"Kayla has a cat and the cat sleeps on the couch, later take {i}", kind="said_user", session=f"n{i}")[0] for i in range(4)]
    m.remember("Kayla still has a cat that sleeps on the couch.", kind="fact", session="reflection", chain=False, links=[new[0]])
    d = audit.diagnose(m, sample=50)
    assert 0 < d["plates"]["logged"] < d["plates"]["count"]
    by, back = d["unexplained_by_plate"], d["returned_by_plate"]
    assert back["legacy"] > 0 and back["logged"] > 0 and by["legacy"] <= back["legacy"] and by["logged"] <= back["logged"]
    assert by["legacy"] + by["logged"] >= d["recovery"]["as it is"]["unexplained"]             # read from both, counted under each
    assert d["known_links"]["from_log"] == 2                                                   # the new conclusion and its source, both ways


def test_an_image_and_its_parts_explain_one_another(tmp_path):
    m = store(tmp_path)
    said = m.remember("Here is a photo of Sushi on the pink blanket", kind="said_user", session="s1")[0]
    whole = m.remember("A photo of a black and white cat on a pink blanket", kind="image", session="s1", chain=False, links=[said], meta={"image_id": 7})[0]
    part = m.remember("A close-up of pink woven fabric", kind="image_part", session="s1", chain=False, links=[whole], meta={"image_id": 7})[0]
    linked = audit._probably_linked(m)
    assert part in linked[whole] and whole in linked[part]
    d = audit.diagnose(m, sample=10)
    r = d["recovery"]["as it is"]
    assert r["expected"] > 0 and r["found"] + r["explained"] + r["conversation"] + r["related"] + r["unexplained"] == r["returned"]


def test_what_comes_back_is_sorted_by_the_best_account_of_why(tmp_path):
    """Looking at real pairs the report had called unexplained: two were the user's own consecutive statements,
    which are bound on purpose; three were what a memory like the cue had been bound to; one of those was another
    photograph altogether."""
    m = store(tmp_path)
    a = m.remember("Good morning, I slept well", kind="said_user", session="talk")[0]
    reply = m.remember("Good morning! I am glad you slept well", kind="said_assistant", session="talk")[0]
    b = m.remember("That is incredible news about the lake", kind="said_user", session="talk", links=[a])[0]
    later = m.remember("We should visit the lake in the spring", kind="said_assistant", session="talk")[0]
    other = m.remember("Tomatoes need watering early in the day", kind="said_user", session="garden")[0]
    linked = audit._probably_linked(m)
    assert b in linked[a] and a in linked[b]                           # what she said before, across the reply between
    assert reply in linked[a] and other not in linked.get(a, set())
    photo1 = m.remember("A sunset over a field with orange clouds", kind="image", session="talk", chain=False, meta={"image_id": 1})[0]
    part1 = m.remember("Orange clouds above a dark ridge at sunset", kind="image_part", session="talk", chain=False, links=[photo1], meta={"image_id": 1})[0]
    photo2 = m.remember("A sunset over water with orange clouds and pine trees", kind="image", session="pics", chain=False, meta={"image_id": 2})[0]
    d = audit.diagnose(m, sample=50)
    r, g = d["recovery"]["as it is"], d["given"]
    for t in (r,):
        assert t["found"] + t["explained"] + t["conversation"] + t["related"] + t["unexplained"] == t["returned"]
    assert g["known"] + g["explained"] + g["conversation"] + g["related"] + g["unexplained"] == g["by_association"]
    assert set(d["strength"]) == set(audit.SORTS) and r["other_image"] >= 0 and g["other_image"] >= 0
    assert all(set(e) == {"cue", "given"} for e in g["other_image_examples"])
    # the sorting itself, on returns chosen by hand: cue 1, in conversation "talk", part of image 7
    alike = {30: 0.9, 40: 0.1, 50: 0.5, 60: 0.8}
    got = audit.sort_returns({10, 11, 20, 30, 40, 50, 60, 70}, [1],
                             known={1: {10, 99}}, probable={1: {10, 11, 12}}, live={10, 11, 20, 30, 40, 50, 60, 70},
                             session_of={1: "talk", 20: "talk", 30: "garden", 40: "garden", 50: "reflection", 60: "pics", 70: "reflection"},
                             image_of={1: 7, 11: 7, 60: 8, 20: 7}, likeness=lambda i, cues: alike.get(i, 0.0))
    assert got["expected"] == {10, 99} and got["known"] == {10} and got["explained"] == {11}       # known comes before explained
    assert got["conversation"] == {20}                                # said in the same conversation, bound to nothing
    assert got["related"] == {30, 50, 60}                             # alike enough to have been a cue itself
    assert got["unexplained"] == {40, 70}                             # none of these
    assert got["other_image"] == {60}                                 # alike, and from another picture: 11 and 20 are the same picture
    # sessions that are not conversations do not make two memories conversation-mates
    got = audit.sort_returns({70}, [50], known={}, probable={}, live={50, 70}, session_of={50: "reflection", 70: "reflection"},
                             image_of={}, likeness=lambda i, cues: 0.0)
    assert got["conversation"] == set() and got["unexplained"] == {70} and got["other_image"] == set()
    assert audit.ALIKE == 0.3


def test_cli_plates_names_the_new_sorts(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io, types
    p = make(tmp_path)
    p.sync_turn("My name is Kayla and I have a cat named Sushi", "Nice to meet you, Kayla. Sushi sounds lovely.", session_id="s1")
    p.sync_turn("He is a long-haired tuxedo cat", "A handsome one, then.", session_id="s1")
    p.shutdown()
    home = tmp_path / "home"
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    try:
        import holonomic.cli as cli, holonomic.embed as embed
        real = embed.OllamaEmbedder
        embed.OllamaEmbedder = lambda *a, **k: HashEmbedder()
        parser = argparse.ArgumentParser(); cli.register_cli(parser)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            args = parser.parse_args(["plates", "check", "-n", "20"]); args.func(args)
        text = out.getvalue()
        assert "same conversation" in text and "related" in text and "the thing\n              you said before" in text
        assert "from a different image than the cue's" in text and "from a different image than the message's" in text
    finally:
        embed.OllamaEmbedder = real
        sys.modules.pop("hermes_constants", None)
