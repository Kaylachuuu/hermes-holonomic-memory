import json, sys, time, types

import pytest

from test_provider import HAVE_HERMES, make, tool
from conftest import track
from holonomic import HolonomicMemory, HashEmbedder


def seeded(tmp_path):
    m = track(HolonomicMemory(tmp_path / "m", HashEmbedder()))
    ids = {}
    ids["name"] = m.remember("Hello! My name is Kayla and I work in IT", kind="said_user", session="s1")[0]
    ids["hi"] = m.remember("It is a pleasure to meet you, Kayla", kind="said_assistant", session="s1")[0]
    ids["os"] = m.remember("The project I am most excited about is an operating system in x86 assembly", kind="said_user", session="s1")[0]
    ids["q"] = m.remember("What should we build first?", kind="asked_user", session="s1")[0]
    return m, ids


def answer(ids, **over):
    data = {"user_facts": [{"text": "Kayla works in IT.", "sources": [ids["name"]]},
                           {"text": "Kayla is writing an operating system in x86 assembly.", "sources": [ids["os"], 999]},
                           {"text": "Kayla owns a yacht.", "sources": [999]},            # cites nothing real
                           {"text": "short", "sources": [ids["name"]]}],
            "self_notes": [{"text": "I greeted Kayla warmly when we first met.", "sources": [ids["hi"]]}],
            "insights": [],
            "user_profile": "Kayla works in IT and is writing an operating system in x86 assembly.",
            "self_profile": "I am Kayla's assistant and we have just started working together."}
    data.update(over)
    return json.dumps(data)


def test_reflection_stores_linked_facts_and_profiles(tmp_path):
    from holonomic.reflect import reflect_once, pending, FACT, SELF_NOTE
    m, ids = seeded(tmp_path)
    seen = {}
    def llm(system, user):
        seen["prompt"] = user
        return answer(ids)
    assert pending(m) == 4
    dry = reflect_once(m, {}, llm=llm, dry_run=True)
    assert dry["read"] == 4 and m.stats()["memories"] == 4 and pending(m) == 4 and m.profile("user") == ""
    assert f"[{ids['name']}] USER: Hello! My name is Kayla" in seen["prompt"] and "ASSISTANT: It is a pleasure" in seen["prompt"]
    report = reflect_once(m, {}, llm=llm)
    assert len(report["stored"]) == 3 and report["profiles_updated"] == ["user", "self"] and pending(m) == 0
    kinds = {r["text"]: r["kind"] for r in m.recent(10)}
    assert kinds["Kayla works in IT."] == FACT and kinds["I greeted Kayla warmly when we first met."] == SELF_NOTE
    assert "Kayla owns a yacht." not in kinds and "short" not in kinds          # unsupported and trivial items dropped
    assert m.profile("user").startswith("Kayla works in IT")
    # the fact is linked through the plates to the memory it came from, in both directions
    fact = next(r["id"] for r in m.recent(10) if r["text"].startswith("Kayla is writing"))
    assert ids["os"] in {h.id for h in m.associates(fact)} and fact in {h.id for h in m.associates(ids["os"])}
    # nothing new: nothing read, the model is not called
    assert reflect_once(m, {}, llm=lambda s, u: (_ for _ in ()).throw(AssertionError("called")))["read"] == 0


def test_repeated_conclusions_reinforce_instead_of_duplicating(tmp_path):
    from holonomic.reflect import reflect_once
    m, ids = seeded(tmp_path)
    reflect_once(m, {}, llm=lambda s, u: answer(ids))
    before = m.stats()["memories"]
    new = m.remember("I have worked in IT for twenty years now", kind="said_user", session="s2")[0]
    again = json.dumps({"user_facts": [{"text": "Kayla works in IT.", "sources": [new]}], "self_notes": [], "insights": [],
                        "user_profile": "Kayla works in IT and is writing an operating system in x86 assembly.",
                        "self_profile": "I am Kayla's assistant and we have just started working together."})
    report = reflect_once(m, {}, llm=lambda s, u: again)
    assert report["stored"] == [] and len(report["reinforced"]) == 1 and report["profiles_updated"] == []
    assert m.stats()["memories"] == before + 1 and len(m.profile_history("user")) == 1


def test_bad_model_output_changes_nothing(tmp_path):
    from holonomic.reflect import reflect_once, ReflectionError, pending
    m, ids = seeded(tmp_path)
    with pytest.raises(ReflectionError):
        reflect_once(m, {}, llm=lambda s, u: "I think Kayla is nice.")
    with pytest.raises(ReflectionError):
        reflect_once(m, {})                                                    # no model configured
    assert pending(m) == 4 and m.stats()["memories"] == 4
    wrapped = "Here you go:\n" + answer(ids) + "\nHope that helps."
    assert len(reflect_once(m, {}, llm=lambda s, u: wrapped)["stored"]) == 3


def test_idle_reflector_respects_toggle_idleness_and_backlog(tmp_path, monkeypatch=None):
    from holonomic import reflect
    m, ids = seeded(tmp_path)
    cfg = {"reflect_enabled": False, "reflect_model": "gemma", "reflect_min_new": 4, "reflect_idle_seconds": 60}
    r = reflect.IdleReflector(m, lambda: cfg, poll_seconds=3600)
    try:
        r.last_activity = time.time() - 120
        assert r.due(cfg) is False                                             # switched off
        cfg["reflect_enabled"] = True
        assert r.due(cfg) is True
        r.touch()
        assert r.due(cfg) is False                                             # the user is active
        r.last_activity = time.time() - 120
        cfg["reflect_min_new"] = 5
        assert r.due(cfg) is False                                             # not enough new memories
        cfg["reflect_min_new"] = 4
        real = reflect.ollama_chat
        reflect.ollama_chat = lambda host, model, system, user, **kw: answer(ids)
        try:
            assert len(r.run_if_due()["stored"]) == 3 and r.run_if_due() is None
        finally:
            reflect.ollama_chat = real
    finally:
        r.stop()


def test_profiles_reach_the_system_prompt_and_reflections_are_recalled(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.reflect import reflect_once
    p = make(tmp_path)
    p.sync_turn("Hello! My name is Kayla and I work in IT.", "It is a pleasure to meet you, Kayla, and to start working together.", session_id="s1")
    assert "What you know about the user" not in p.system_prompt_block()
    rows = {r["text"]: r["id"] for r in p._engine.recent(10)}
    ids = {"name": rows["Hello! My name is Kayla and I work in IT."], "hi": max(rows.values()), "os": 1}
    reflect_once(p._engine, p._cfg, llm=lambda s, u: answer(ids))
    block = p.system_prompt_block()
    assert "What you know about the user" in block and "Kayla works in IT" in block and "How you understand yourself" in block
    recalled = p.prefetch("where does Kayla work, is it in IT?", session_id="s2")
    assert "learned about the user) Kayla works in IT." in recalled
    assert p._cfg["reflect_enabled"] is False
    p.shutdown()


def test_cli_toggle_status_and_profile(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io
    from holonomic.provider import load_config
    p = make(tmp_path)
    p.sync_turn("Hello! My name is Kayla and I work in IT.", "It is a pleasure to meet you, Kayla, and to start working together.", session_id="s1")
    p._engine.set_profile("user", "Kayla works in IT.")
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
        assert "needs a model" in run("reflect", "on") and load_config(home)["reflect_enabled"] is False
        assert "Reflection is ON (model gemma4-64k)" in run("reflect", "on", "--model", "gemma4-64k")
        cfg = load_config(home)
        assert cfg["reflect_enabled"] is True and cfg["reflect_model"] == "gemma4-64k" and cfg["embedder"] == "hash"
        status = run("reflect")
        assert "enabled: True" in status and "memories waiting: 2" in status and "last run: never" in status
        assert "Reflection is OFF" in run("reflect", "off") and load_config(home)["reflect_model"] == "gemma4-64k"
        assert "Kayla works in IT." in run("profile") and "About herself: (none yet)" in run("profile")
        assert "Reflection failed" in run("reflect", "now")                    # no Ollama server here
    finally:
        embed.OllamaEmbedder = real
        sys.modules.pop("hermes_constants", None)
