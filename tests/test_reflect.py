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
    assert "What you know about the user" in block and "Kayla works in IT" in block and "Who you have become" in block
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
        assert "Foundation (SOUL.md" in run("profile") and "not found" in run("profile")
        (home / "SOUL.md").write_text("You are Athena. Be curious, direct and kind.", encoding="utf-8")
        assert "44 characters" in run("profile")
        assert "Profile 'us' set" in run("profile", "--set", "us", "We are just getting started on an operating system together.")
        assert "Usage:" in run("profile", "--set", "them", "nope")
        shown = run("profile", "--history")
        assert "About the two of you:" in shown and "getting started on an operating system" in shown
        assert "Reflection failed" in run("reflect", "now")                    # no Ollama server here
    finally:
        embed.OllamaEmbedder = real
        sys.modules.pop("hermes_constants", None)


def test_foundation_and_relationship(tmp_path):
    from holonomic.reflect import reflect_once, BOND_NOTE
    m, ids = seeded(tmp_path)
    seen = {}
    def llm(system, user):
        seen["prompt"] = user
        return answer(ids, relationship_notes=[{"text": "We have agreed to build an operating system together.", "sources": [ids["os"]]}],
                      relationship_profile="We have just met and plan to build an operating system together.")
    report = reflect_once(m, {}, llm=llm, foundation="You are Athena. Be curious, direct and kind.")
    assert "FOUNDATION:\nYou are Athena. Be curious, direct and kind." in seen["prompt"]
    assert "CURRENT RELATIONSHIP PROFILE:\n(empty)" in seen["prompt"]
    assert report["profiles_updated"] == ["user", "self", "us"] and m.profile("us").startswith("We have just met")
    assert {r["text"]: r["kind"] for r in m.recent(10)}["We have agreed to build an operating system together."] == BOND_NOTE
    # without a SOUL.md the prompt says so instead of leaving a gap
    m.remember("One more thing worth noting about tomorrow", kind="said_user", session="s2")
    reflect_once(m, {}, llm=llm)
    assert "FOUNDATION:\n(none)" in seen["prompt"] and "CURRENT RELATIONSHIP PROFILE:\nWe have just met" in seen["prompt"]


def test_subject_recall_and_relationship_in_system_prompt(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.reflect import reflect_once
    p = make(tmp_path)
    p.sync_turn("Hello! My name is Kayla and I work in IT.", "It is a pleasure to meet you, Kayla, and to start working together.", session_id="s1")
    rows = {r["text"]: r["id"] for r in p._engine.recent(10)}
    ids = {"name": rows["Hello! My name is Kayla and I work in IT."], "hi": max(rows.values()), "os": 1}
    (tmp_path / "home" / "SOUL.md").write_text("You are Athena.", encoding="utf-8")
    reflect_once(p._engine, p._cfg, llm=lambda s, u: answer(
        ids, relationship_notes=[{"text": "We greeted each other warmly on the first day.", "sources": [ids["hi"]]}],
        relationship_profile="We have just met and are getting to know each other."))
    block = p.system_prompt_block()
    assert "Your relationship with the user" in block and "We have just met" in block and "Who you have become" in block
    assert "You are Athena" not in block                                  # the foundation is Hermes' job, not ours
    about_user = tool(p, action="recall", query="what does she do for work", subject="user")
    assert about_user["profile"].startswith("Kayla works in IT") and {h["kind"] for h in about_user["results"]} == {"fact"}
    about_us = tool(p, action="recall", query="how did we start out", subject="us")
    assert [h["kind"] for h in about_us["results"]] == ["bond_note"] and about_us["profile"].startswith("We have just met")
    assert "error" in tool(p, action="recall", query="x", subject="them")
    p.shutdown()


def test_a_fact_is_retired_only_when_the_user_contradicts_it(tmp_path):
    from holonomic.reflect import reflect_once
    m, ids = seeded(tmp_path)
    reflect_once(m, {}, llm=lambda s, u: answer(ids))
    old = next(r["id"] for r in m.recent(20) if r["text"] == "Kayla works in IT.")
    other = next(r["id"] for r in m.recent(20) if r["text"].startswith("Kayla is writing"))
    said = m.remember("Kayla works as a sailing instructor now, I left the IT job last month", kind="said_user", session="s3")[0]
    reply = m.remember("That is a big change from working in IT, congratulations", kind="said_assistant", session="s3")[0]
    seen = {}
    def llm(system, user):
        seen["prompt"] = user
        return json.dumps({"user_facts": [], "self_notes": [], "relationship_notes": [], "insights": [],
            "superseded": [{"fact": old, "replacement": "Kayla works as a sailing instructor.", "sources": [said]},
                           {"fact": other, "replacement": "Kayla has given up on the operating system.", "sources": [reply]},  # only the assistant said so
                           {"fact": 4242, "replacement": "Kayla lives on the moon these days.", "sources": [said]}],            # never shown
            "user_profile": "Kayla works as a sailing instructor and is writing an operating system in x86 assembly.",
            "self_profile": "", "relationship_profile": ""})
    report = reflect_once(m, {}, llm=llm)
    assert f"[{old}] Kayla works in IT." in seen["prompt"].split("EXISTING FACTS ABOUT THE USER")[1].split("MEMORIES:")[0]
    assert [s["fact"] for s in report["superseded"]] == [old]
    new = next(r["id"] for r in m.recent(20) if r["text"] == "Kayla works as a sailing instructor.")
    assert m.get(old)["trust"] == 0.0 and m.get(old)["meta"]["superseded_by"] == new and m.get(other)["trust"] == 0.6
    # the old fact leaves ordinary recall but is still on record, linked to what replaced it
    assert old not in {h.id for h in m.recall("where does Kayla work, in IT?", k=10, min_score=0.0, min_trust=0.15)}
    assert old in {h.id for h in m.recall("where does Kayla work, in IT?", k=10, min_score=0.0)}
    assert old in {h.id for h in m.associates(new)}
    # and it is no longer offered to the model as an existing fact
    m.remember("Sailing lessons start again in the spring", kind="said_user", session="s4")
    reflect_once(m, {}, llm=llm)
    assert f"[{old}]" not in seen["prompt"].split("EXISTING FACTS ABOUT THE USER")[1].split("MEMORIES:")[0]


def test_facts_about_the_user_are_exempt_from_fading(tmp_path):
    m, ids = seeded(tmp_path)
    fact = m.remember("Computers have been a lifelong passion of Kayla's", kind="fact", session="reflection", chain=False)[0]
    for _ in range(50):
        m.decay(0.9, exempt_kinds=("fact",))
    assert m.get(fact)["strength"] == pytest.approx(1.0) and m.get(ids["os"])["strength"] == pytest.approx(0.1)


def test_ollama_chat_caps_output_and_reports_runaway():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from holonomic import reflect
    seen, mode = [], {"v": "ok"}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if mode["v"] == "nothink" and "think" in body:
                out, code = json.dumps({"error": "this model does not support think"}).encode(), 400
            elif mode["v"] == "runaway":
                out, code = json.dumps({"message": {"content": "{ \n \n \n"}, "done_reason": "length", "eval_count": 2000}).encode(), 200
            else:
                out, code = json.dumps({"message": {"content": "{}"}, "done_reason": "stop", "eval_count": 2, "prompt_eval_count": 9}).encode(), 200
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
        def log_message(self, *a): pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{srv.server_port}"
    try:
        assert reflect.ollama_chat(host, "m", "s", "u", timeout=5, temperature=0.3, max_tokens=123) == "{}"
        assert seen[-1]["options"]["num_predict"] == 123 and seen[-1]["think"] is False and seen[-1]["stream"] is False
        assert reflect.LAST_CALL["reply_tokens"] == 2 and reflect.LAST_CALL["done_reason"] == "stop"
        mode["v"] = "nothink"
        assert reflect.ollama_chat(host, "m", "s", "u", timeout=5, temperature=0.3) == "{}" and "think" not in seen[-1]
        mode["v"] = "runaway"
        with pytest.raises(reflect.ReflectionError):
            reflect.ollama_chat(host, "m", "s", "u", timeout=5, temperature=0.3)
    finally:
        srv.shutdown()
