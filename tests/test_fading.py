"""What becomes of her memories is hers to decide (2026-10-10): faded memories put back, fading only with her
agreement when a persona service takes part, and a dream strengthening only the old memories she chooses."""
import json, random

import pytest

from test_persona import service
from test_sleep import DAY, NOW, OS, GARDEN, fake, store, talk

FADING_SERVICE = "thymos/0.11.0 accounts=1 idle=1 slept=1 fading=1 dream_choice=1"


def test_unfade_puts_faded_memories_back_and_never_lowers(tmp_path):
    from holonomic.sleep import unfade
    m = store(tmp_path)
    old = talk(m, "s-old", NOW - 10 * DAY, OS)                 # stored at 1.0 here; a reply of hers is stored at 0.8
    live = talk(m, "s-live", NOW - 60, GARDEN)
    m.kv_set("episode:s-old", str(old[-1]))                   # she has an account of s-old
    m.fade(old, 0.5)
    m.reinforce([old[0]], 0.9)                                # recalled since: 0.5 + 0.9, above where it started
    m.fade(live, 0.5)                                         # not summarised, so never the fade step's doing
    dry = unfade(m, {}, apply=False, now=NOW)
    assert dry["said"] == 5 and dry["below"] == 4 and dry["raised"] == 0
    assert dry["lowest"]["ratio"] == pytest.approx(0.5, abs=0.01) and dry["under_threshold"] == 0
    assert m.get(old[1])["strength"] == pytest.approx(0.5, abs=0.01)                      # a dry run changes nothing
    done = unfade(m, {}, apply=True, now=NOW)
    assert done["raised"] == 4
    assert m.get(old[0])["strength"] == pytest.approx(1.4, abs=0.01)                      # kept what it had
    assert m.get(old[1])["strength"] == pytest.approx(0.8, abs=0.01)                      # her side, as stored
    assert m.get(old[2])["strength"] == pytest.approx(1.0, abs=0.01)
    assert m.get(live[0])["strength"] == pytest.approx(0.5, abs=0.01)                     # outside what fading touches
    assert json.loads(m.kv_get("fade:restored"))["raised"] == 4
    assert unfade(m, {}, apply=True, now=NOW)["below"] == 0


def test_with_a_service_that_takes_part_nothing_fades_until_she_agrees(tmp_path):
    from holonomic import persona
    from holonomic.sleep import sleep_once
    m = store(tmp_path)
    home = tmp_path / "home"
    old = talk(m, "s-old", NOW - 10 * DAY, OS)
    m.kv_set("episode:s-old", str(old[-1]))
    m.kv_set("sleep:last_fade", str(NOW - 5 * DAY))
    assert persona.her_fading(home, {}) is None                                           # no service: the setting decides
    with service(FADING_SERVICE):
        assert persona.her_fading(home, {}) is False                                      # no decision is not agreement
        r = sleep_once(m, {}, steps=["fade"], now=NOW, her_fading=persona.her_fading(home, {}))
        assert r["fade"]["skipped"] == "she has not agreed to fading" and m.get(old[0])["strength"] == pytest.approx(1.0)
        (persona.folder(home)).mkdir(parents=True, exist_ok=True)
        (persona.folder(home) / persona.FADING).write_text(json.dumps({"fading": True}))
        assert persona.her_fading(home, {}) is True
        r = sleep_once(m, {}, steps=["fade"], now=NOW, her_fading=persona.her_fading(home, {}))
        assert r["fade"]["changed"] == 5
        # Her agreement does not override the setting: off is off.
        assert sleep_once(m, {"fade_enabled": False}, steps=["fade"], now=NOW + DAY, her_fading=True)["fade"] is None


def test_the_settings_she_is_told_of_are_written_when_they_change(tmp_path):
    from holonomic import persona
    home = tmp_path / "home"
    assert persona.tell_settings(home, {}) is None                                        # no service, nothing written
    with service(FADING_SERVICE):
        path = persona.tell_settings(home, {"fade_enabled": False}, now=NOW)
        first = json.loads(path.read_text())
        assert first["fade_enabled"] is False and first["dream_reinforce"] == "off" and first["at"] == NOW
        persona.tell_settings(home, {"fade_enabled": False}, now=NOW + 60)
        assert json.loads(path.read_text())["at"] == NOW                                  # unchanged, not rewritten
        persona.tell_settings(home, {"fade_enabled": False}, restored={"at": NOW, "raised": 4}, now=NOW + 90)
        persona.tell_settings(home, {"fade_enabled": False, "dream_reinforce": "chosen"}, now=NOW + 120)
        now = json.loads(path.read_text())
        assert now["dream_reinforce"] == "chosen" and now["restored"]["raised"] == 4      # the restore stays told


def test_dream_reinforce_modes(tmp_path):
    from holonomic.sleep import reinforce_mode, sleep_config
    assert [reinforce_mode(v) for v in (False, True, "all", "chosen", "off", "bogus")] == ["off", "all", "all", "chosen", "off", "off"]
    assert sleep_config({})["dream_reinforce"] == "off"


def _dream_store(tmp_path):
    m = store(tmp_path)
    old = talk(m, "s-old", NOW - 20 * DAY, OS)
    new = talk(m, "s-new", NOW - DAY, [("said_user", "I planted an assembly of herbs in the garden twenty steps from the door"),
                                       ("said_user", "The garden project started again after years of waiting")])
    dream_text = ("I am kneeling in the garden and the herbs are growing in rows of assembly, each leaf a line of code. " * 2).strip()
    llm = fake({"dream": json.dumps({"dream": dream_text}), "wake": json.dumps({"thoughts": "", "connections": []})})
    return m, old, new, llm


def test_chosen_a_dream_strengthens_nothing_and_says_what_it_reached(tmp_path):
    from holonomic import persona
    from holonomic.sleep import sleep_once
    m, old, new, llm = _dream_store(tmp_path)
    before = {i: m.get(i)["strength"] for i in old}
    report = sleep_once(m, {"dream_reinforce": "chosen"}, llm=llm, steps=["dream"], rng=random.Random(3), now=NOW, persona=True)
    d = report["dream"]
    assert {i: m.get(i)["strength"] for i in old} == before
    assert d["reached"] and all(r["id"] in old for r in d["reached"]) and all(r["text"] for r in d["reached"])
    assert m.get(d["id"])["meta"]["older"] == sorted(r["id"] for r in d["reached"])
    home = tmp_path / "home"
    with service(FADING_SERVICE):
        item = json.loads(persona.tell_slept(home, {"dream_reinforce": "chosen"}, report, now=NOW).read_text())
    assert item["dreams"][0]["reached"][0]["id"] == d["reached"][0]["id"] and item["keep_closer_amount"] == 0.1
    with service("thymos/0.4.0 accounts=1 idle=1 slept=1"):                              # a service that does not choose
        item = json.loads(persona.tell_slept(home, {"dream_reinforce": "chosen"}, report, now=NOW + 1).read_text())
    assert "reached" not in item["dreams"][0]


def test_she_keeps_closer_only_what_she_chose_and_only_what_the_dream_reached(tmp_path):
    from holonomic import persona
    from holonomic.sleep import sleep_once
    m, old, new, llm = _dream_store(tmp_path)
    cfg = {"dream_reinforce": "chosen"}
    d = sleep_once(m, cfg, llm=llm, steps=["dream"], rng=random.Random(3), now=NOW, persona=True)["dream"]
    pick = d["reached"][0]["id"]
    before = {i: m.get(i)["strength"] for i in old + new}
    home = tmp_path / "home"
    folder = persona.folder(home) / persona.DREAM_THOUGHTS
    folder.mkdir(parents=True)
    (folder / "1-a.json").write_text(json.dumps({"dream_id": d["id"], "thoughts": "", "keep_closer": [pick, new[0]],
                                                 "choice_entry_hash": "c1", "written_at": NOW}))
    out = persona.take_dream_thoughts(m, home, cfg=cfg)
    assert out == [{"dream": d["id"], "thoughts": "", "kept_closer": [pick]}]             # new[0] was not reached
    assert m.get(pick)["strength"] == pytest.approx(before[pick] + 0.1, abs=1e-4)
    assert m.get(new[0])["strength"] == pytest.approx(before[new[0]])
    meta = m.get(d["id"])["meta"]
    assert meta["kept_closer"] == [pick] and meta["kept_closer_entry"] == "c1" and "her_thoughts" not in meta
    # With the setting no longer 'chosen', her choice is kept with the dream and strengthens nothing.
    (folder / "2-b.json").write_text(json.dumps({"dream_id": d["id"], "keep_closer": [pick]}))
    persona.take_dream_thoughts(m, home, cfg={})
    assert m.get(pick)["strength"] == pytest.approx(before[pick] + 0.1, abs=1e-4)
    meta = m.get(d["id"])["meta"]
    assert meta["chose_closer"] == [pick] and meta["kept_closer"] == []


def test_cli_unfade_reports_then_raises(tmp_path):
    import contextlib, io, sys, types
    from holonomic import cli
    m = store(tmp_path)
    old = talk(m, "s-old", NOW - 10 * DAY, OS)
    m.kv_set("episode:s-old", str(old[-1]))
    m.fade(old, 0.6)
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: tmp_path / "home")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli._unfade(m, {}, types.SimpleNamespace(apply=False, no_images=False))
    text = out.getvalue()
    assert "5 memories could have faded" in text and "The lowest is at 0.60" in text and "Nothing was changed" in text
    assert "Fading is still on" in text
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli._unfade(m, {"fade_enabled": False}, types.SimpleNamespace(apply=True, no_images=False))
    assert "Raised 5" in out.getvalue() and "Fading is still on" not in out.getvalue()
