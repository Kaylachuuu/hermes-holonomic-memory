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
    def llm(system, user, step):
        seen[step] = user
        return answer(ids)
    assert pending(m) == 4
    dry = reflect_once(m, {}, llm=llm, dry_run=True)
    assert dry["read"] == 4 and m.stats()["memories"] == 4 and pending(m) == 4 and m.profile("user") == ""
    assert f"[{ids['name']}] USER: Hello! My name is Kayla" in seen["propose"] and "ASSISTANT: It is a pleasure" in seen["propose"]
    assert list(seen) == ["propose", "check", "profiles"] and dry["depth"] == 3
    # the checker sees each statement with only the lines it cites; the profile writer sees no conversation at all
    assert "STATEMENT 1 (fact about the user): Kayla works in IT." in seen["check"] and "What should we build first?" not in seen["check"]
    assert "(fact about the user) Kayla works in IT." in seen["profiles"] and "USER:" not in seen["profiles"]
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
        out = run("reflect", "now", "--again")
        assert "Going over the last 3 day(s) again, from memory #" in out and "reinforced, not stored twice" in out and "Reflection failed" in out
        assert "Going over the last 0.5 day(s) again" in run("reflect", "now", "--again", "0.5", "--dry-run")
    finally:
        embed.OllamaEmbedder = real
        sys.modules.pop("hermes_constants", None)


def test_foundation_and_relationship(tmp_path):
    from holonomic.reflect import reflect_once, BOND_NOTE
    m, ids = seeded(tmp_path)
    seen = {}
    def llm(system, user, step):
        seen[step] = user
        return answer(ids, relationship_notes=[{"text": "We have agreed to build an operating system together.", "sources": [ids["os"]]}],
                      relationship_profile="We have just met and plan to build an operating system together.")
    report = reflect_once(m, {}, llm=llm, foundation="You are Athena. Be curious, direct and kind.")
    assert "FOUNDATION:\nYou are Athena. Be curious, direct and kind." in seen["propose"]
    assert "FOUNDATION:\nYou are Athena. Be curious, direct and kind." in seen["profiles"] and "FOUNDATION" not in seen["check"]
    assert "CURRENT RELATIONSHIP PROFILE:\n(empty)" in seen["profiles"]
    assert report["profiles_updated"] == ["user", "self", "us"] and m.profile("us").startswith("We have just met")
    assert {r["text"]: r["kind"] for r in m.recent(10)}["We have agreed to build an operating system together."] == BOND_NOTE
    # without a SOUL.md the prompt says so instead of leaving a gap
    m.remember("One more thing worth noting about tomorrow", kind="said_user", session="s2")
    reflect_once(m, {}, llm=llm)
    assert "FOUNDATION:\n(none)" in seen["propose"]
    from holonomic.reflect import build_profile_prompt
    prompt = build_profile_prompt([], [], {"us": m.profile("us")}, "")
    assert "CURRENT RELATIONSHIP PROFILE:\nWe have just met" in prompt and "FOUNDATION:\n(none)" in prompt


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
    def llm(system, user, step):
        seen[step] = user
        return json.dumps({"user_facts": [], "self_notes": [], "relationship_notes": [], "insights": [],
            "superseded": [{"fact": old, "replacement": "Kayla works as a sailing instructor.", "sources": [said]},
                           {"fact": other, "replacement": "Kayla has given up on the operating system.", "sources": [reply]},  # only the assistant said so
                           {"fact": 4242, "replacement": "Kayla lives on the moon these days.", "sources": [said]}],            # never shown
            "user_profile": "Kayla works as a sailing instructor and is writing an operating system in x86 assembly.",
            "self_profile": "", "relationship_profile": ""})
    report = reflect_once(m, {}, llm=llm)
    assert f"[{old}] Kayla works in IT." in seen["propose"].split("EXISTING FACTS ABOUT THE USER")[1].split("MEMORIES:")[0]
    assert [s["fact"] for s in report["superseded"]] == [old]
    assert "(no longer true) Kayla works in IT.  Now: Kayla works as a sailing instructor." in seen["profiles"]
    new = next(r["id"] for r in m.recent(20) if r["text"] == "Kayla works as a sailing instructor.")
    assert m.get(old)["trust"] == 0.0 and m.get(old)["meta"]["superseded_by"] == new and m.get(other)["trust"] == 0.6
    # the old fact leaves ordinary recall but is still on record, linked to what replaced it
    assert old not in {h.id for h in m.recall("where does Kayla work, in IT?", k=10, min_score=0.0, min_trust=0.15)}
    assert old in {h.id for h in m.recall("where does Kayla work, in IT?", k=10, min_score=0.0)}
    assert old in {h.id for h in m.associates(new)}
    # and it is no longer offered to the model as an existing fact
    m.remember("Sailing lessons start again in the spring", kind="said_user", session="s4")
    reflect_once(m, {}, llm=llm)
    assert f"[{old}]" not in seen["propose"].split("EXISTING FACTS ABOUT THE USER")[1].split("MEMORIES:")[0]


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
        assert reflect.reflect_config({})["reflect_think"] is False
        assert reflect.LAST_CALL["reply_tokens"] == 2 and reflect.LAST_CALL["done_reason"] == "stop"
        mode["v"] = "nothink"
        assert reflect.ollama_chat(host, "m", "s", "u", timeout=5, temperature=0.3) == "{}" and "think" not in seen[-1]
        mode["v"] = "runaway"
        with pytest.raises(reflect.ReflectionError):
            reflect.ollama_chat(host, "m", "s", "u", timeout=5, temperature=0.3)
    finally:
        srv.shutdown()


def test_a_fact_resting_only_on_the_assistants_words_is_dropped(tmp_path):
    from holonomic.reflect import reflect_once
    m, ids = seeded(tmp_path)
    out = answer(ids, user_facts=[{"text": "Kayla values meticulous minimalism above all.", "sources": [ids["hi"]]},      # assistant line only
                                  {"text": "Kayla is writing an operating system in x86 assembly.", "sources": [ids["hi"], ids["os"]]}])
    report = reflect_once(m, {}, llm=lambda s, u: out, dry_run=True)
    assert report["dropped_assistant_only"] == ["Kayla values meticulous minimalism above all."]
    assert [f["text"] for f in report["proposed"]["fact"]] == ["Kayla is writing an operating system in x86 assembly."]


def test_the_check_step_drops_and_rewrites(tmp_path):
    from holonomic.reflect import reflect_once
    m, ids = seeded(tmp_path)
    proposal = json.dumps({"user_facts": [{"text": "Kayla works in IT.", "sources": [ids["name"]]},
                                          {"text": "Kayla is a highly skilled developer writing an operating system.", "sources": [ids["os"]]},
                                          {"text": "Kayla values meticulous minimalism in code.", "sources": [ids["name"], ids["hi"]]}],
                           "self_notes": [{"text": "I should remember that Kayla is impressive.", "sources": [ids["hi"]]}],
                           "relationship_notes": [], "insights": [], "superseded": []})
    verdicts = json.dumps({"verdicts": [{"item": 2, "verdict": "rewrite", "text": "Kayla is writing an operating system in x86 assembly."},
                                        {"item": 3, "verdict": "drop", "text": ""},
                                        {"item": 4, "verdict": "drop", "text": ""},
                                        {"item": 99, "verdict": "drop", "text": ""}]})      # no such item; item 1 gets no verdict
    profiles = json.dumps({"user_profile": "Kayla works in IT and is writing an operating system in x86 assembly.",
                           "self_profile": "", "relationship_profile": ""})
    seen = {}
    def llm(system, user, step):
        seen[step] = user
        return {"propose": proposal, "check": verdicts, "profiles": profiles}[step]
    report = reflect_once(m, {}, llm=llm)
    stored = {r["text"]: r["kind"] for r in m.recent(20) if r["kind"] in ("fact", "self_note")}
    assert stored == {"Kayla works in IT.": "fact", "Kayla is writing an operating system in x86 assembly.": "fact"}
    assert [(c["verdict"], c["kind"]) for c in report["checked"]] == [("rewrite", "fact"), ("drop", "fact"), ("drop", "self_note")]
    accepted = seen["profiles"].split("NEWLY ACCEPTED STATEMENTS:")[1]
    assert "meticulous minimalism" not in accepted and "highly skilled" not in accepted and "x86 assembly" in accepted
    assert report["profiles_updated"] == ["user"] and m.profile("self") == ""


def test_depth_levels(tmp_path):
    from holonomic.reflect import reflect_once, reflect_config
    assert reflect_config({})["reflect_depth"] == 3 and reflect_config({"reflect_depth": 9})["reflect_depth"] == 3
    m, ids = seeded(tmp_path)
    steps = []
    def llm(system, user, step):
        steps.append((step, user))
        return answer(ids)
    one = reflect_once(m, {"reflect_depth": 1}, llm=llm, dry_run=True)
    assert [s for s, _ in steps] == ["propose", "profiles"] and "FOUNDATION" not in steps[0][1] and "self_notes" not in steps[0][1]
    assert one["proposed"]["self_note"] == [] and len(one["proposed"]["fact"]) == 2
    assert one["profiles"]["user"].startswith("Kayla works in IT") and one["profiles"]["self"] == ""
    steps.clear()
    two = reflect_once(m, {"reflect_depth": 2}, llm=llm, dry_run=True)
    assert [s for s, _ in steps] == ["propose", "profiles"] and len(two["proposed"]["self_note"]) == 1 and two["profiles"]["self"]
    steps.clear()
    reflect_once(m, {"reflect_depth": 3}, llm=llm, dry_run=True)
    assert [s for s, _ in steps] == ["propose", "check", "profiles"]


def test_runaway_thinking_is_repeated_without_thinking(tmp_path):
    from holonomic import reflect
    m, ids = seeded(tmp_path)
    calls = []
    def fake(host, model, system, user, *, timeout, temperature, max_tokens, think, schema):
        calls.append((think, max_tokens, sorted(schema["properties"])))
        reflect.LAST_CALL.clear(); reflect.LAST_CALL.update({"seconds": 1.0, "prompt_tokens": 10, "reply_tokens": max_tokens if think else 50,
                                                             "done_reason": "length" if think else "stop", "think": think})
        if think and len(calls) == 1:
            raise reflect.ReflectionError(f"The model hit the {max_tokens}-token reply limit without finishing. Its reply began: ''")
        return answer(ids)
    real = reflect.ollama_chat
    reflect.ollama_chat = fake
    try:
        report = reflect.reflect_once(m, {"reflect_model": "gemma", "reflect_max_tokens": 1000, "reflect_think": True}, dry_run=True)
    finally:
        reflect.ollama_chat = real
    assert calls[0][:2] == (True, 4000) and calls[1][:2] == (False, 1000)          # same step, repeated plainly
    assert [c["step"] for c in report["calls"]] == ["propose", "propose (repeated without thinking)", "check", "profiles"]
    assert "failed" in report["calls"][0] and calls[2][2] == ["verdicts"] and "user_profile" in calls[3][2]
    assert len(report["proposed"]["fact"]) == 2


def test_quotes_are_kept_only_around_the_users_own_words(tmp_path):
    from holonomic.reflect import reflect_once, unquote_unsaid
    said = "I feel like software today is so bloated. I wanted it to host apps from other systems."
    assert unquote_unsaid("Kayla wants a 'no-bloat' approach.", said) == "Kayla wants a no-bloat approach."
    assert unquote_unsaid('Kayla thinks software is "so bloated" today.', said) == 'Kayla thinks software is "so bloated" today.'
    assert unquote_unsaid("Kayla's vision is a \u2018meta-platform\u2019 for apps.", said) == "Kayla's vision is a meta-platform for apps."
    assert unquote_unsaid("Kayla's daughter's school doesn't open early.", said) == "Kayla's daughter's school doesn't open early."
    m, ids = seeded(tmp_path)
    out = answer(ids, user_facts=[{"text": "Kayla's goal is a 'meta-platform' built as an operating system in x86 assembly.",
                                   "sources": [ids["os"], ids["hi"]]}],
                 self_notes=[{"text": "I called her plan a 'meta-platform' and she seemed to like it.", "sources": [ids["hi"]]}])
    report = reflect_once(m, {"reflect_depth": 2}, llm=lambda s, u: out, dry_run=True)
    assert report["proposed"]["fact"][0]["text"] == "Kayla's goal is a meta-platform built as an operating system in x86 assembly."
    assert "'meta-platform'" in report["proposed"]["self_note"][0]["text"]            # her own words stay quoted in her own note
    assert [c["verdict"] for c in report["checked"]] == ["unquote"]


def test_facts_are_checked_against_the_users_lines_only(tmp_path):
    from holonomic.reflect import reflect_once
    m, ids = seeded(tmp_path)
    out = answer(ids, user_facts=[{"text": "Kayla is writing an operating system, expanding its functionality.", "sources": [ids["os"], ids["hi"]]}],
                 self_notes=[{"text": "I greeted Kayla warmly when we first met.", "sources": [ids["hi"]]}])
    seen = {}
    def llm(system, user, step):
        seen[step] = user
        return out
    report = reflect_once(m, {}, llm=llm, dry_run=True)
    fact_block = seen["check"].split("STATEMENT 1")[1].split("STATEMENT 2")[0]
    assert "USER: The project I am most excited about" in fact_block and "ASSISTANT:" not in fact_block
    assert "ASSISTANT: It is a pleasure" in seen["check"].split("STATEMENT 2")[1]        # her own note keeps her own line
    assert report["proposed"]["fact"][0]["sources"] == [ids["os"]]


def test_checker_is_told_which_names_the_cited_lines_never_mention(tmp_path):
    from holonomic.reflect import unsupported_names, build_check_prompt, FACT, SELF_NOTE
    line = ("If there's a way to use C/C++ that doesn't include a bunch of abstraction or unnecessary extra bloat, that could be "
            "an option, or if there's some other language that might be a better fit I am open to suggestions.")
    stmt = "Kayla is open to using C, C++, or other languages like Zig or Rust for her next project."
    assert unsupported_names(stmt, line, known_text="Hello! My name is Kayla.") == ["Zig", "Rust"]
    assert unsupported_names(stmt, line) == ["Zig", "Rust"]                      # "Kayla" opens the sentence
    assert unsupported_names("She reduced CHOICE.COM from 5KB to 52 bytes.", "my CHOICE.COM was 52 bytes, the original 5kb") == []
    assert unsupported_names("It shrank to 48 bytes.", "my version was 52 bytes") == ["48"]
    by_id = {40: {"id": 40, "kind": "said_user", "text": line}, 41: {"id": 41, "kind": "said_assistant", "text": "Zig or Rust would suit you."}}
    prompt = build_check_prompt([{"kind": FACT, "text": stmt, "sources": [40]},
                                 {"kind": SELF_NOTE, "text": "I suggested Zig and Rust to her.", "sources": [41]}], by_id, "Kayla")
    first, second = prompt.split("STATEMENT 2")
    assert "appear nowhere in its lines: Zig, Rust" in first and "note:" not in second


def test_reflection_can_go_over_old_ground_without_losing_its_place(tmp_path):
    """A pass can miss something.  Reading from an earlier point goes over conversation it has read before, stores
    nothing twice, and never moves the mark for what is new backwards."""
    from holonomic.reflect import reflect_once, pending, WATERMARK
    m, ids = seeded(tmp_path)
    llm = lambda system, user, step: answer(ids)
    first = reflect_once(m, {}, llm=llm)
    assert len(first["stored"]) == 3 and pending(m) == 0 and first["last_id"] == max(ids.values())
    mark, count = int(m.kv_get(WATERMARK)), m.stats()["memories"]
    seen = {}

    def again(system, user, step):
        seen[step] = user
        return answer(ids)
    report = reflect_once(m, {}, llm=again, start_after=min(ids.values()) - 1)
    assert report["read"] >= 4 and f"[{ids['name']}] USER: Hello! My name is Kayla" in seen["propose"]     # the old conversation is read again
    assert report["stored"] == [] and len(report["reinforced"]) == 3 and m.stats()["memories"] == count    # and nothing is stored twice
    assert int(m.kv_get(WATERMARK)) >= mark and pending(m) == 0
    # starting from the middle leaves the mark where it was
    m.remember("I also have a cat called Theo.", kind="said_user", session="s2")
    assert pending(m) == 1
    reflect_once(m, {}, llm=llm, start_after=min(ids.values()), dry_run=True)
    assert pending(m) == 1
    assert m.first_id_since(0) == min(r["id"] for r in m.recent(50)) and m.first_id_since(time.time() + 60) is None


def test_a_name_one_letter_off_is_put_right(tmp_path):
    """A reflection wrote 'Kaylar' through a whole run and the checker let it stand."""
    from holonomic.reflect import reflect_once, respell_names
    known = "My name is Kayla. Kayla works in IT. Kayla has a cat. Theo is there, and Thea too. Thea is a friend. Thea sings."
    assert respell_names("Kaylar's cat is called Theo.", known) == "Kayla's cat is called Theo."
    assert respell_names("Kayl works in IT.", known) == "Kayla works in IT."
    assert respell_names("Kayla has a cat.", known) == "Kayla has a cat."
    assert respell_names("Theo sleeps on the bed.", known) == "Theo sleeps on the bed."        # in what is known: a name of its own
    assert respell_names("Thep sleeps on the bed.", known, own="my friend Thep") == "Thep sleeps on the bed."   # her own word
    assert respell_names("Kaylee came to visit.", known) == "Kaylee came to visit."            # two letters off: someone else
    assert respell_names("Where does Kayla work?", "There it is. There you go. There now. Kayla") == "Where does Kayla work?"
    # a real run: her name only ever opened a sentence, and 'They' is one letter from 'Theo'
    facts = "Kayla works in IT. Kayla has a cat named Theo. Kayla owns a cat named Theo. Theo is a tabby. Kayla is a mother."
    assert respell_names("Kaylar's memory system keeps everything.", facts) == "Kayla's memory system keeps everything."
    assert respell_names("They are moving on. Then Theo slept. Them too.", facts) == "They are moving on. Then Theo slept. Them too."
    assert respell_names("Thea came by.", facts, seen="my sister thea") == "Thea came by."
    assert respell_names("Kaylar is here.", facts + " c:/users/kayla/documents") == "Kayla is here."      # a path does not unmake a name
    # a misspelling already among the facts does not protect itself, while the right name is far more common
    assert respell_names("Kaylar likes tea.", known + " Kayla" * 6 + " Kaylar") == "Kayla likes tea."
    # a misspelling already among the facts is copied by the next reflection: being common there protects nothing,
    # and what settles it is whether she ever wrote the word herself
    stored = facts + " Kaylar's system fades. Kaylar's system reflects. Kaylar's system has a tool."
    assert respell_names("Kaylar's system fades.", stored) == "Kaylar's system fades."
    assert respell_names("Kaylar's system fades.", stored, wrote=lambda w: False) == "Kayla's system fades."
    assert respell_names("Kaylar's system fades.", stored, wrote=lambda w: w == "Kaylar") == "Kaylar's system fades."
    m, ids = seeded(tmp_path)
    assert m.user_wrote("Kayla") and m.user_wrote("kayla") and not m.user_wrote("Kaylar") and not m.user_wrote("Kay")
    assert not m.user_wrote("pleasure")                    # the assistant said that
    m.remember("Kayla here again. Kayla is my name, and I have a cat called Theo.", kind="said_user", session="s1")
    last = m.recent(1)[0]["id"]
    seen = {}

    def llm(system, user, step):
        seen[step] = user
        return answer(ids, user_facts=[{"text": "Kaylar has a cat called Theo.", "sources": [last]}], self_notes=[],
                      user_profile="Kaylar works in IT and has a cat called Theo.")
    report = reflect_once(m, {}, llm=llm)
    assert [c for c in report["checked"] if c["verdict"] == "respell"][0]["now"] == "Kayla has a cat called Theo."
    assert "Kayla has a cat called Theo." in {r["text"] for r in m.recent(10)}
    assert "(fact about the user) Kayla has a cat called Theo." in seen["profiles"]
    assert m.profile("user") == "Kayla works in IT and has a cat called Theo."


def test_the_profile_is_written_from_everything_a_run_took_in(tmp_path):
    """Going over old ground reads in passes.  The profile used to be written from the last pass alone, and one cut
    for length said nothing about it."""
    from holonomic.reflect import reflect_once, _PROFILES
    m, ids = seeded(tmp_path)
    seen = {}

    def llm(system, user, step):
        seen[step] = user
        return answer(ids, user_profile="Kayla works in IT. " + "She likes long walks by the lake very much. " * 40)
    earlier = [{"kind": "fact", "text": "Kayla has a cat named Theo."}, {"kind": "fact", "text": "Kayla works in IT."}]
    report = reflect_once(m, {}, llm=llm, earlier=earlier)
    assert "(fact about the user) Kayla has a cat named Theo." in seen["profiles"]
    assert seen["profiles"].count("Kayla works in IT.") == 1 or seen["profiles"].count("(fact about the user) Kayla works in IT.") == 1
    assert {"kind": "fact", "text": "Kayla works in IT."} in report["accepted"]
    assert "shorten" not in seen and m.profile("user") == "Kayla works in IT. She likes long walks by the lake very much."   # a repeat is said once
    assert "profile_cut" not in report

    def long(system, user, step):
        if step == "shorten":
            assert "It may have at most 160" in user and " words. It may" in user and "walk number 59" in user
            return json.dumps({"text": "Kayla works in IT and has two cats, Sushi and Theo."})
        return answer(ids, user_profile="Kayla works in IT. " + " ".join(f"She likes walk number {i} by the lake." for i in range(60)))
    report = reflect_once(m, {}, llm=long, start_after=0)
    assert report["profile_shortened"]["user"] > 0 and m.profile("user") == "Kayla works in IT and has two cats, Sushi and Theo."

    def stubborn(system, user, step):
        return answer(ids, text="", user_profile="Kayla works in IT. " + " ".join(f"She likes walk number {i} by the lake." for i in range(60)))
    report = reflect_once(m, {}, llm=stubborn, start_after=0)
    assert report["profile_cut"]["user"] > 0 and len(m.profile("user")) <= 1200 and m.profile("user").endswith(".")
    assert "Never leave out a person or an animal that has a name" in _PROFILES and "cut off and lost" in _PROFILES


def test_a_profile_with_nothing_new_is_left_as_it_is(tmp_path):
    """Every rewrite drifts a little: with no note about the two of them accepted, 'networking' in the profile of
    the pair came back as 'inquiry'."""
    from holonomic.reflect import reflect_once
    m, ids = seeded(tmp_path)
    m.set_profile("us", "We talk about her work in networking.")
    m.set_profile("self", "I like a tidy answer.")
    drift = dict(user_profile="Kayla works in IT and writes assembly.", self_profile="I like an untidy answer.",
                 relationship_profile="We talk about her work in inquiry.")
    report = reflect_once(m, {}, llm=lambda system, user, step: answer(ids, self_notes=[], **drift))
    assert report["profiles_updated"] == ["user"]
    assert m.profile("us") == "We talk about her work in networking." and m.profile("self") == "I like a tidy answer."
    report = reflect_once(m, {}, llm=lambda system, user, step: answer(ids, **drift), start_after=0)      # a self note this time
    assert m.profile("self") == "I like an untidy answer." and m.profile("us") == "We talk about her work in networking."
