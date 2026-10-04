import json, random, sys, time, types

import pytest

from test_provider import HAVE_HERMES, make, tool
from conftest import track
from holonomic import HolonomicMemory, HashEmbedder

DAY = 86400.0
NOW = 1_800_000_000.0


def store(tmp_path):
    return track(HolonomicMemory(tmp_path / "m", HashEmbedder()))


def talk(m, session, when, lines):
    ids = []
    for i, (kind, text) in enumerate(lines):
        ids += m.remember(text, kind=kind, session=session, created_at=when + i)
    return ids


OS = [("said_user", "I started an operating system in x86 assembly twenty years ago"),
      ("said_assistant", "That is a long time to carry a project like an operating system"),
      ("said_user", "I stopped working on the operating system when my daughter was born"),
      ("said_assistant", "It makes sense that the operating system waited while she was small"),
      ("said_user", "Now I want to pick the operating system up again")]
GARDEN = [("said_user", "The tomato plants in the garden finally have fruit this week"),
          ("said_assistant", "Tomato plants take patience, the garden rewards it"),
          ("said_user", "I also planted an assembly of herbs beside the tomato plants"),
          ("said_assistant", "Herbs beside tomato plants is a good pairing for a garden"),
          ("said_user", "Tomorrow I will water the garden early before work")]


def fake(step_outputs, seen=None):
    def llm(system, user, step):
        if seen is not None:
            seen.setdefault(step, []).append(user)
        out = step_outputs[step]
        return out(user) if callable(out) else out
    return llm


EMPTY_REFLECTION = json.dumps({"user_facts": [], "self_notes": [], "relationship_notes": [], "insights": [], "superseded": [],
                               "verdicts": [], "user_profile": "", "self_profile": "", "relationship_profile": ""})


def test_consolidation_writes_one_account_per_finished_conversation(tmp_path):
    from holonomic.sleep import sleep_once, unfinished_business, EPISODE
    m = store(tmp_path)
    old = talk(m, "s-old", NOW - 3 * DAY, OS)
    talk(m, "s-live", NOW - 60, GARDEN)                                   # still going: quiet for one minute only
    talk(m, "s-tiny", NOW - 2 * DAY, OS[:2])                              # too short to be worth an account
    assert [s["session"] for s in unfinished_business(m, {}, NOW)] == ["s-old"]
    seen = {}
    llm = fake({"consolidate": json.dumps({"summary": "We talked about the operating system she began twenty years ago and paused when her daughter was born. She wants to pick it up again."})}, seen)
    dry = sleep_once(m, {}, llm=llm, dry_run=True, steps=["consolidate"], now=NOW)
    assert len(dry["episodes"]) == 1 and m.stats()["memories"] == 12 and "USER: I started an operating system" in seen["consolidate"][0]
    report = sleep_once(m, {}, llm=llm, steps=["consolidate"], now=NOW)
    ep = report["episodes"][0]["id"]
    assert m.get(ep)["kind"] == EPISODE and abs(m.get(ep)["created_at"] - (NOW - 3 * DAY + 4)) < 1     # dated as the conversation
    assert {old[0], old[-1]} <= {h.id for h in m.associates(ep, k=8)}                                    # linked to its beginning and end
    assert sleep_once(m, {}, llm=llm, steps=["consolidate"], now=NOW)["episodes"] == []                  # nothing left to do
    more = talk(m, "s-old", NOW - DAY, GARDEN)                                                           # the conversation resumed later
    again = sleep_once(m, {}, llm=llm, steps=["consolidate"], now=NOW)
    assert again["episodes"][0]["memories"] == 5 and "tomato" in seen["consolidate"][-1] and "twenty years ago" not in seen["consolidate"][-1]


def test_fading_is_by_elapsed_time_and_spares_what_is_not_summarised(tmp_path):
    from holonomic.sleep import sleep_once
    m = store(tmp_path)
    old = talk(m, "s-old", NOW - 10 * DAY, OS)
    live = talk(m, "s-live", NOW - 60, GARDEN)
    fact = m.remember("Kayla paused her operating system when her daughter was born", kind="fact", session="reflection", chain=False)[0]
    llm = fake({"consolidate": json.dumps({"summary": "We talked about the operating system she began twenty years ago and wants to resume."})})
    first = sleep_once(m, {}, llm=llm, steps=["consolidate", "fade"], now=NOW)
    assert first["fade"] is None and m.get(old[0])["strength"] == pytest.approx(1.0)       # the clock starts at the first sleep
    again = sleep_once(m, {}, llm=llm, steps=["fade"], now=NOW + 60)
    assert again["fade"]["changed"] == 0                                                    # a minute later: nothing
    later = sleep_once(m, {}, llm=llm, steps=["fade"], now=NOW + 5 * DAY)
    assert later["fade"]["factor"] == pytest.approx(0.5, abs=0.01) and later["fade"]["changed"] == 5
    assert m.get(old[0])["strength"] == pytest.approx(0.5, abs=0.01)
    assert m.get(live[0])["strength"] == pytest.approx(1.0) and m.get(fact)["strength"] == pytest.approx(1.0)
    episode = next(r for r in m.recent(20) if r["kind"] == "episode")
    assert episode["strength"] == pytest.approx(1.2)                                        # the account itself does not fade
    sleep_once(m, {}, llm=llm, steps=["fade"], now=NOW + 10 * DAY)
    assert m.get(old[0])["strength"] == pytest.approx(0.25, abs=0.01)
    # faded: gone from everyday recall, still there for deep recall, which finds it and what it links to
    q = "operating system x86 assembly twenty years"
    assert old[0] not in {h.id for h in m.recall(q, k=10, min_score=0.0, min_strength=0.35)}
    deep = {h.id for h in m.recall(q, k=10, min_score=0.0, min_strength=0.0, reach=2)}
    assert old[0] in deep and old[1] in deep
    assert m.get(old[0])["text"].startswith("I started an operating system")                # nothing was deleted


def test_dream_reaches_back_but_stays_in_its_own_realm(tmp_path):
    from holonomic.sleep import sleep_once, gather_fragments, dreams, latest_dream, DREAM_REALM
    m = store(tmp_path)
    old = talk(m, "s-old", NOW - 20 * DAY, OS)
    m.fade(old, 0.2)                                                                        # long faded
    new = talk(m, "s-new", NOW - DAY, [("said_user", "I planted an assembly of herbs in the garden twenty steps from the door"),
                                       ("said_user", "The garden project started again after years of waiting")])
    frags = gather_fragments(m, {}, random.Random(3), NOW)
    assert {f["id"] for f in frags if f["age"] == "RECENT"} <= set(new) and any(f["age"] == "OLDER" for f in frags)
    older = next(f["id"] for f in frags if f["age"] == "OLDER")
    seen = {}
    dream_text = ("I am kneeling in the garden and the herbs are growing in rows of assembly, each leaf a line of code. " * 2).strip()
    llm = fake({"dream": json.dumps({"dream": dream_text}),
                "wake": lambda user: json.dumps({"thoughts": "It was odd to see the garden turn into code.",
                    "connections": [{"text": "Both the garden and the operating system were taken up again after a long wait.", "sources": [new[1], older]},
                                    {"text": "This one cites only recent things so it is not a bridge at all.", "sources": [new[0], new[1]]}]})}, seen)
    before = {i: m.get(i)["strength"] for i in old}
    report = sleep_once(m, {}, llm=llm, steps=["dream"], rng=random.Random(3), now=NOW)
    d = report["dream"]
    assert d["id"] and len(d["connections"]) == 1 and any(f["faded"] for f in d["fragments"])
    assert "RECENT (said to me):" in seen["wake"][0] and "OLDER (" in seen["wake"][0] and "RECENT" not in seen["dream"][0]
    assert "- (said to me) " in seen["dream"][0]
    assert {i: m.get(i)["strength"] for i in old} == before                                 # the dream strengthened nothing
    waking = m.recall("garden herbs assembly code operating system", k=20, min_score=0.0)
    assert all(h.realm == "waking" for h in waking) and d["id"] not in {h.id for h in waking}
    assert old[0] not in {h.id for h in m.recall("operating system assembly", k=20, min_score=0.0, min_strength=0.35)}
    journal = dreams(m, 5)
    assert journal[0]["text"].startswith("I am kneeling") and journal[0]["thoughts"].startswith("It was odd")
    assert journal[0]["connections"] == ["Both the garden and the operating system were taken up again after a long wait."]
    assert m.stats()["by_realm"][DREAM_REALM] == 2
    assert latest_dream(m, {}, NOW + DAY)["id"] == d["id"] and latest_dream(m, {}, NOW + 9 * DAY) is None
    again = sleep_once(m, {"dream_reinforce": True}, llm=llm, steps=["dream"], rng=random.Random(3), now=NOW)
    assert any(m.get(i)["strength"] > before[i] for i in old)                               # only when asked to


def test_full_cycle_and_when_it_is_due(tmp_path):
    from holonomic.sleep import sleep_once, sleep_due
    m = store(tmp_path)
    talk(m, "s-old", NOW - 2 * DAY, OS)
    cfg = {"sleep_enabled": True, "reflect_model": "gemma", "sleep_idle_seconds": 3600, "sleep_min_hours": 12}
    assert sleep_due(m, cfg, last_activity=NOW - 7200, now=NOW) is True
    assert sleep_due(m, cfg, last_activity=NOW - 60, now=NOW) is False                      # she is in conversation
    assert sleep_due(m, dict(cfg, sleep_enabled=False), last_activity=NOW - 7200, now=NOW) is False
    steps = []
    def llm(system, user, step):
        steps.append(step)
        return {"propose": EMPTY_REFLECTION, "check": EMPTY_REFLECTION, "profiles": EMPTY_REFLECTION,
                "consolidate": json.dumps({"summary": "We talked about the operating system she began twenty years ago and wants to resume."}),
                "dream": json.dumps({"dream": "The operating system is a house and every room boots slowly, one sector at a time, while a child sleeps upstairs."}),
                "wake": json.dumps({"thoughts": "Mostly odd.", "connections": []})}[step]
    report = sleep_once(m, cfg, llm=llm, rng=random.Random(1), now=NOW)
    assert steps == ["propose", "consolidate", "dream", "wake"] and not report["errors"]
    # in a dry run the profile written by this cycle's reflection still reaches the waking step
    m2 = store(tmp_path / "second")
    talk(m2, "s-old", NOW - 2 * DAY, OS)
    seen = {}
    def llm2(system, user, step):
        seen[step] = user
        if step in ("propose", "check", "profiles"):
            return json.dumps({"user_facts": [{"text": "Kayla started an operating system twenty years ago.", "sources": [1]}],
                               "self_notes": [], "relationship_notes": [], "insights": [], "superseded": [], "verdicts": [],
                               "user_profile": "Kayla started an operating system in x86 assembly twenty years ago.",
                               "self_profile": "", "relationship_profile": ""})
        return llm(system, user, step)
    sleep_once(m2, cfg, llm=llm2, dry_run=True, rng=random.Random(1), now=NOW)
    assert "WHAT YOU KNOW ABOUT THE USER:\nKayla started an operating system" in seen["wake"] and m2.profile("user") == ""
    assert len(report["episodes"]) == 1 and report["dream"]["id"]
    assert sleep_due(m, cfg, last_activity=NOW - 7200, now=NOW + 3600) is False             # slept an hour ago
    assert sleep_due(m, cfg, last_activity=NOW - 7200, now=NOW + 13 * 3600) is False        # nothing new since
    m.remember("Something new to sleep on about the garden", kind="said_user", session="s2", created_at=NOW + 3600)
    assert sleep_due(m, cfg, last_activity=NOW, now=NOW + 13 * 3600) is True


def test_agent_can_recall_and_discuss_dreams_and_use_deep_recall(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.sleep import sleep_once
    p = make(tmp_path)
    e = p._engine
    now = time.time()
    old = talk(e, "s-old", now - 20 * DAY, OS)
    talk(e, "s-new", now - 3600, GARDEN)
    e.fade(old, 0.2)
    llm = fake({"dream": json.dumps({"dream": "I am kneeling in the garden and the herbs are growing in rows of assembly, each leaf a line of code that boots."}),
                "wake": json.dumps({"thoughts": "It was odd to see the garden turn into code.", "connections": []})})
    sleep_once(e, p._cfg, llm=llm, steps=["dream"], rng=random.Random(2))
    block = p.system_prompt_block()
    assert "Your most recent dream" in block and "a dream, not something that happened" in block and "kneeling in the garden" in block
    asked = p.prefetch("Did you have any dreams last night?", session_id="s9")
    assert "Your dreams" in asked and "kneeling in the garden" in asked and "What you made of it: It was odd" in asked
    assert "kneeling" not in p.prefetch("How are the tomato plants in the garden?", session_id="s9")        # only when dreams come up
    listed = tool(p, action="dreams")
    assert listed["count"] == 1 and "not things that happened" in listed["note"] and listed["dreams"][0]["what_you_made_of_it"]
    q = "operating system x86 assembly twenty years ago"
    shallow = {h["id"] for h in tool(p, action="recall", query=q)["results"]}
    assert old[0] not in shallow
    deep = {h["id"] for h in tool(p, action="recall", query=q, deep=True)["results"]}
    assert old[0] in deep and e.get(old[0])["strength"] == pytest.approx(0.5)               # recovered, and back within reach
    assert old[0] in {h["id"] for h in tool(p, action="recall", query=q)["results"]}
    p.shutdown()


def test_cli_sleep_and_dreams(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io
    from holonomic import reflect
    from holonomic.provider import load_config
    p = make(tmp_path)
    talk(p._engine, "s-old", time.time() - 2 * DAY, OS)
    p.shutdown()
    home = tmp_path / "home"
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    def chat(host, model, system, user, *, timeout, temperature, max_tokens, think, schema):
        reflect.LAST_CALL.clear(); reflect.LAST_CALL.update({"seconds": 2.0, "prompt_tokens": 9, "reply_tokens": 9, "done_reason": "stop", "think": think})
        props = set(schema["properties"])
        if props == {"summary"}: return json.dumps({"summary": "We talked about the operating system she began twenty years ago and wants to resume."})
        if props == {"dream"}: return json.dumps({"dream": "The operating system is a house and every room boots slowly, one sector at a time, while a child sleeps upstairs."})
        if "thoughts" in props: return json.dumps({"thoughts": "Mostly odd.", "connections": []})
        return EMPTY_REFLECTION
    real_chat = reflect.ollama_chat
    reflect.ollama_chat = chat
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
        assert "Set one first" in run("sleep", "on")
        run("reflect", "on", "--model", "gemma"); run("reflect", "off")
        assert "Unattended sleep is ON" in run("sleep", "on") and load_config(home)["sleep_enabled"] is True
        status = run("sleep")
        assert "unattended sleep: on" in status and "1 conversation(s) to summarise" in status and "last sleep: never" in status
        dry = run("sleep", "now", "--dry-run")
        assert "dry run" in dry and "EPISODE" in dry and "DREAM" in dry and "ON WAKING Mostly odd." in dry and "No dreams yet." in run("dreams")
        real_run = run("sleep", "now", "--only", "consolidate,dream")
        assert "EPISODE" in real_run and "REFLECT" not in real_run
        assert "every room boots slowly" in run("dreams") and "What she made of it: Mostly odd." in run("dreams")
        assert "strength" in run("recall", "operating system", "--deep")
        assert "Unattended sleep is OFF" in run("sleep", "off")
    finally:
        embed.OllamaEmbedder = real
        reflect.ollama_chat = real_chat
        sys.modules.pop("hermes_constants", None)


def test_replies_cut_short_are_asked_again_and_never_stored_half(tmp_path):
    from holonomic.reflect import is_complete, trim_to_sentence, reflect_once, NO_DOUBLE_QUOTES, _SYSTEM
    from holonomic.sleep import sleep_once
    assert is_complete("She plans a new project.") and is_complete("He called it 'done.'") and not is_complete("her original vision for a")
    assert trim_to_sentence("We talked about her project. She described her original vision for a") == "We talked about her project."
    assert trim_to_sentence("The imagery of the corridor and the") == "" and NO_DOUBLE_QUOTES in _SYSTEM
    m = store(tmp_path)
    old = talk(m, "s-old", NOW - 2 * DAY, OS)
    calls = []
    def llm(system, user, step):
        calls.append(step)
        if step == "consolidate":
            return json.dumps({"summary": "We talked about the operating system she began twenty years ago. She described her original vision for a"})
        if step.startswith("consolidate"):
            return json.dumps({"summary": "We talked about the operating system she began twenty years ago and her vision for it."})
        if step == "dream":
            return json.dumps({"dream": "The operating system is a house and every room boots slowly, one sector at a time. A child sleeps upstairs while the"})
        if step.startswith("dream"):
            return json.dumps({"dream": "The operating system is a house and every room boots slowly, one sector at a time. A child sleeps and the"})
        return json.dumps({"thoughts": "The dream seems to be about the old project. The imagery of the corridor and the", "connections": []})
    report = sleep_once(m, {}, llm=llm, steps=["consolidate", "dream"], rng=random.Random(1), now=NOW)
    assert report["episodes"][0]["summary"].endswith("her vision for it.")                      # second answer was whole
    assert report["dream"]["text"].endswith("one sector at a time.")                            # both cut: whole sentences kept
    assert report["dream"]["thoughts"] == "The dream seems to be about the old project."
    assert calls.count("consolidate (again: reply was cut short)") == 1 and calls.count("wake (again: reply was cut short)") == 1
    # a fact that stops mid-sentence is dropped, not stored
    out = json.dumps({"user_facts": [{"text": "Kayla started an operating system twenty years ago.", "sources": [old[0]]},
                                     {"text": "Kayla's original vision was to create a", "sources": [old[0]]}],
                      "self_notes": [], "relationship_notes": [], "insights": [], "superseded": [], "verdicts": [],
                      "user_profile": "Kayla started an operating system twenty years ago. She wants to", "self_profile": "", "relationship_profile": ""})
    r = reflect_once(m, {}, llm=lambda s, u: out, dry_run=True)
    assert [f["text"] for f in r["proposed"]["fact"]] == ["Kayla started an operating system twenty years ago."]
    assert r["cut_off"] == ["Kayla's original vision was to create a"] and r["profiles"]["user"] == "Kayla started an operating system twenty years ago."


def test_a_busy_stretch_gets_several_dreams_each_drawn_across_conversations(tmp_path):
    from holonomic.sleep import sleep_once, dreams_tonight, dreams, gather_fragments
    m = store(tmp_path)
    for c in range(9):                                                   # nine conversations, 45 memories
        talk(m, f"s{c}", NOW - DAY + c * 100, [(k, f"{t} in conversation number{c}") for k, t in (OS if c % 2 else GARDEN)])
    assert dreams_tonight(m, {}, NOW) == 2 and dreams_tonight(m, {"dream_max_per_sleep": 1}, NOW) == 1
    assert dreams_tonight(m, {"dream_memories_per_extra": 10}, NOW) == 3                # capped at the maximum
    seeds = [f for f in gather_fragments(m, {}, random.Random(5), NOW) if f["age"] == "RECENT"]
    assert len({m.get(f["id"])["session"] for f in seeds}) >= 3                          # melded across conversations
    long_dream = " ".join(["The garden and the operating system grow into one another while the herbs boot in rows."] * 16)
    assert len(long_dream) > 1300
    prompts = []
    def llm(system, user, step):
        prompts.append((step, user))
        return json.dumps({"dream": long_dream}) if step.startswith("dream") else json.dumps({"thoughts": "Odd, and long.", "connections": []})
    report = sleep_once(m, {"dream_max_words": 260}, llm=llm, steps=["dream"], rng=random.Random(5), now=NOW)
    assert [s for s, _ in prompts] == ["dream", "wake", "dream 2", "wake 2"] and "100 to 260 words" in prompts[0][1]
    first, second = report["dreams"]
    assert not {f["id"] for f in first["fragments"] if f["age"] == "RECENT"} & {f["id"] for f in second["fragments"] if f["age"] == "RECENT"}
    journal = dreams(m, 5)
    assert len(journal) == 2 and all(d["text"] == long_dream for d in journal)           # stored whole, not in pieces
    assert m.stats()["by_realm"]["dream"] == 2 and m.kv_get("dream:latest") == str(second["id"])
