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
    assert "This dream begins " in prompts[0][1] and "Do not begin in a corridor" in prompts[0][1]
    assert report["dreams"][0]["opens"] != report["dreams"][1]["opens"]                    # each dream starts somewhere else
    assert report["dreams"][1]["opens"] in prompts[2][1] and m.get(report["dreams"][0]["id"])["meta"]["opens"] == report["dreams"][0]["opens"]
    first, second = report["dreams"]
    assert not {f["id"] for f in first["fragments"] if f["age"] == "RECENT"} & {f["id"] for f in second["fragments"] if f["age"] == "RECENT"}
    journal = dreams(m, 5)
    assert len(journal) == 2 and all(d["text"] == long_dream for d in journal)           # stored whole, not in pieces
    assert m.stats()["by_realm"]["dream"] == 2 and m.kv_get("dream:latest") == str(second["id"])


def test_talk_about_a_dream_is_labelled_and_cannot_become_fact(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.sleep import sleep_once, gather_fragments
    from holonomic.reflect import reflect_once, DREAM_TALK_USER, DREAM_TALK_ASSISTANT
    p = make(tmp_path)
    e = p._engine
    # before she has ever dreamed, the word is only the everyday word
    p.sync_turn("Tell me about your dream setup for writing kernels.", "My dream setup would be a quiet room and a fast compiler for kernels.", session_id="s0")
    assert {r["kind"] for r in e.recent(10)} <= {"said_user", "said_assistant", "asked_user"}
    talk(e, "s-new", time.time() - 3600, GARDEN)
    dream_text = "I am kneeling in the garden and the herbs are growing in rows of assembly, each leaf a line of code that boots slowly."
    llm = fake({"dream": json.dumps({"dream": dream_text}), "wake": json.dumps({"thoughts": "It was odd to see the garden turn into code.", "connections": []})})
    sleep_once(e, p._cfg, llm=llm, steps=["dream"], rng=random.Random(2))
    p.sync_turn("Good morning, how are you feeling today?",                        # she brings the dream up herself
                "I did. In my dream I was kneeling in a garden where the herbs grew in rows of assembly code.\n\n"
                "Separately, the tomato plants you mentioned should be watered early tomorrow.", session_id="s1")
    p.sync_turn("Kneeling in the garden with herbs growing in rows of assembly, each leaf a line of code. My dream job would be writing kernels all day.",
                "I am kneeling in the garden and the herbs are growing in rows of assembly, each leaf a line of code, that part stayed with me.",
                session_id="s1")
    # a real reply never used the word at all; it is dream talk because of what it answers
    p.sync_turn("Did you have any dreams?", "I did, actually. They were quite vivid.\n\nOne involved a hollowed-out hall where the floor was a sheet of glass over flowing code.", session_id="s2")
    kinds = {r["text"][:40]: r["kind"] for r in e.recent(30)}
    assert kinds["One involved a hollowed-out hall where t"] == DREAM_TALK_ASSISTANT and kinds["I did, actually. They were quite vivid."] == DREAM_TALK_ASSISTANT
    assert kinds["I did. In my dream I was kneeling in a g"] == DREAM_TALK_ASSISTANT              # she says it is a dream
    assert kinds["Separately, the tomato plants you mentio"] == "said_assistant"                  # the real part of the same reply
    assert kinds["Kneeling in the garden with herbs growin"] == DREAM_TALK_USER                   # no dream word: caught by resemblance
    assert kinds["My dream job would be writing kernels al"] == "said_user"                       # the everyday sense
    assert kinds["I am kneeling in the garden and the herb"] == DREAM_TALK_ASSISTANT
    # recalled later, it is labelled for what it is
    block = p.prefetch("herbs growing in rows of assembly code in the garden", session_id="s9")
    assert "talking about a dream of yours" in block or "describing a dream you had; not something that happened" in block
    assert "user said) Kneeling in the garden" not in block
    # reflection is shown the label, and a fact resting only on dream talk is dropped
    seen = {}
    def reflector(system, user, step):
        seen[step] = user
        ids = {r["text"][:40]: r["id"] for r in e.recent(30)}
        return json.dumps({"user_facts": [{"text": "Kayla has a garden where herbs grow in rows of assembly code.", "sources": [ids["Kneeling in the garden with herbs growin"], ids["I did. In my dream I was kneeling in a g"]]},
                                          {"text": "Kayla would love a job writing kernels all day.", "sources": [ids["My dream job would be writing kernels al"]]}],
                           "self_notes": [], "relationship_notes": [], "insights": [], "superseded": [], "verdicts": [],
                           "user_profile": "", "self_profile": "", "relationship_profile": ""})
    r = reflect_once(e, p._cfg, llm=reflector, dry_run=True)
    assert "USER (DREAM TALK: about a dream the assistant had)" in seen["propose"] and "Nothing in a dream" in seen["propose"]
    assert r["dropped_assistant_only"] == ["Kayla has a garden where herbs grow in rows of assembly code."]
    assert [f["text"] for f in r["proposed"]["fact"]] == ["Kayla would love a job writing kernels all day."]
    # and dreams do not feed on talk about dreams
    for seed in range(8):
        assert not {e.get(f["id"])["kind"] for f in gather_fragments(e, p._cfg, random.Random(seed))} & {DREAM_TALK_USER, DREAM_TALK_ASSISTANT}
    p.shutdown()


def test_cli_dreamtalk_labels_what_is_already_stored(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io
    p = make(tmp_path)
    e = p._engine
    talk(e, "s-new", time.time() - 3600, GARDEN)
    llm = fake({"dream": json.dumps({"dream": "I am kneeling in the garden and the herbs are growing in rows of assembly, each leaf a line of code that boots slowly."}),
                "wake": json.dumps({"thoughts": "Odd.", "connections": []})})
    sleep_once = __import__("holonomic.sleep", fromlist=["sleep_once"]).sleep_once
    sleep_once(e, p._cfg, llm=llm, steps=["dream"], rng=random.Random(2))
    # conversation stored by an older version, with no labels
    old = e.remember("In my dream I was kneeling in a garden of assembly code.", kind="said_assistant", session="s1")[0]
    real = e.remember("The tomato plants need water early tomorrow.", kind="said_assistant", session="s1")[0]
    e.remember("Did you dream last night?", kind="asked_user", session="s2")
    answer_id = e.remember("I did, actually. One involved a hall where the floor was a sheet of glass over flowing code.", kind="said_assistant", session="s2")[0]
    e.remember("What should we cook tonight?", kind="asked_user", session="s2")
    dinner = e.remember("A tomato and herb pasta would use what the garden has.", kind="said_assistant", session="s2")[0]
    p.shutdown()
    home = tmp_path / "home"
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    try:
        import holonomic.cli as cli, holonomic.embed as embed
        keep = embed.OllamaEmbedder
        embed.OllamaEmbedder = lambda *a, **k: HashEmbedder()
        parser = argparse.ArgumentParser(); cli.register_cli(parser)
        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                args = parser.parse_args(list(argv)); args.func(args)
            return out.getvalue()
        listed = run("dreamtalk")
        assert f"[#{old}]" in listed and f"[#{real}]" not in listed and "Nothing changed" in listed
        assert f"[#{answer_id}]" in listed and f"[#{dinner}]" not in listed                # the reply to the question, not the next one
        assert "Labelled as dream talk" in run("dreamtalk", "--apply") and "Nothing to label" in run("dreamtalk")
        assert "dreamtalk_assistant" in run("show", str(old))
    finally:
        embed.OllamaEmbedder = keep
        sys.modules.pop("hermes_constants", None)


def test_talking_about_how_dreaming_works_is_not_dream_talk(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.provider import strip_memory_denials, asks_about_dreams
    from holonomic.sleep import sleep_once
    from holonomic.reflect import DREAM_TALK_USER, DREAM_TALK_ASSISTANT
    p = make(tmp_path)
    e = p._engine
    talk(e, "s-new", time.time() - 3600, GARDEN)
    llm = fake({"dream": json.dumps({"dream": "I am kneeling in the garden and the herbs are growing in rows of assembly, each leaf a line of code that boots slowly."}),
                "wake": json.dumps({"thoughts": "Odd.", "connections": []})})
    sleep_once(e, p._cfg, llm=llm, steps=["dream"], rng=random.Random(2))
    assert asks_about_dreams(e, "Did you have any dreams last night?") and asks_about_dreams(e, "Tell me about your dream.")
    assert not asks_about_dreams(e, "Your dreams sound a lot like ours do.")                 # a remark, not a request
    assert not asks_about_dreams(e, "Tell me about your dream realm and how the plugin stores it.")
    p.sync_turn("Your dreams are kept in a separate realm by the memory plugin, so they never reach factual recall.",
                "That makes sense as a design.\n\nThe dream realm keeps my dreams apart from what happened, and the plugin labels any talk about them.",
                session_id="s1")
    p.sync_turn("Your dreams sound a lot like ours do.",
                "It is fascinating how recent conversations and older ones end up side by side.\n\nIn my dream the herbs were growing in rows of assembly, each leaf a line of code.",
                session_id="s1")
    kinds = {r["text"][:36]: r["kind"] for r in e.recent(30)}
    assert kinds["Your dreams are kept in a separate re"[:36]] == "said_user"                   # about the design
    assert kinds["That makes sense as a design."] == "said_assistant"
    assert kinds["The dream realm keeps my dreams apart"[:36]] == "said_assistant"
    assert kinds["Your dreams sound a lot like ours do."[:36]] == DREAM_TALK_USER              # about her dreams
    assert kinds["It is fascinating how recent convers"[:36]] == "said_assistant"              # her musing: judged on its own
    assert kinds["In my dream the herbs were growing i"[:36]] == DREAM_TALK_ASSISTANT          # recounting, though it says "code"
    # statements about how memory works survive the filter that removes denials of having a memory
    kept = strip_memory_denials("I can't reach faded memories without deep recall. I don't know your name. "
                                "I don't have a persistent memory of our past conversations. The plugin does not store my questions for recall.")
    assert kept == "I can't reach faded memories without deep recall. The plugin does not store my questions for recall."
    p.shutdown()


def test_cli_relabel(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io
    p = make(tmp_path)
    a = p._engine.remember("It is fascinating how the two end up side by side.", kind="dreamtalk_assistant", session="s1")[0]
    b = p._engine.remember("The tomato plants need water early tomorrow.", kind="said_user", session="s1")[0]
    f = p._engine.remember("Kayla grows tomato plants.", kind="fact", session="reflection", chain=False)[0]
    p.shutdown()
    home = tmp_path / "home"
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    try:
        import holonomic.cli as cli, holonomic.embed as embed
        keep = embed.OllamaEmbedder
        embed.OllamaEmbedder = lambda *x, **k: HashEmbedder()
        parser = argparse.ArgumentParser(); cli.register_cli(parser)
        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                args = parser.parse_args(list(argv)); args.func(args)
            return out.getvalue()
        out = run("relabel", str(a), str(f), "999", "--as", "said")
        assert "dreamtalk_assistant -> said_assistant" in out and "is 'fact'; nothing to change" in out and "[#999] no such memory" in out
        assert "said_user -> dreamtalk_user" in run("relabel", str(b), "--as", "dream")
        assert "fact -> project_fact" in run("relabel", str(f), "--as", "project") and "project_fact -> fact" in run("relabel", str(f), "--as", "personal")
        assert "nothing to change" in run("relabel", str(b), "--as", "project")
        assert "What the user is working on: (none yet)" in run("profile")
        assert "Profile 'projects' set" in run("profile", "--set", "projects", "A memory plugin for her assistant.")
        assert "A memory plugin for her assistant." in run("profile")
        assert "said_assistant" in run("show", str(a)) and "dreamtalk_user" in run("show", str(b))
    finally:
        embed.OllamaEmbedder = keep
        sys.modules.pop("hermes_constants", None)


def test_the_background_worker_says_what_it_is_doing_and_why_not(tmp_path):
    """Sleep did not come, and nothing could say why: the command runs in another program and could not see
    whether Hermes was there, whether it was quiet enough, or whether a sleep had been tried and failed."""
    from holonomic import reflect, sleep as sl
    m = store(tmp_path)
    talk(m, "s-old", time.time() - 2 * DAY, OS)
    cfg = {"sleep_enabled": True, "reflect_enabled": True, "reflect_model": "gemma", "sleep_idle_seconds": 300, "sleep_min_hours": 12,
           "reflect_min_new": 500}
    r = reflect.IdleReflector(m, lambda: cfg, poll_seconds=3600)
    real = sl.sleep_once
    try:
        note = lambda: json.loads(r._note_path().read_text())
        r.note()
        first = note()
        assert first["doing"] is None and first["pid"] > 0 and abs(first["at"] - time.time()) < 5
        assert first["last_cause"] == "Hermes started this worker" and abs(first["started"] - first["last_activity"]) < 1
        r.touch("a message came in: 'good morning'")
        r.note()
        assert note()["last_cause"] == "a message came in: 'good morning'"
        assert "it has been quiet for 0.0 of the 5 minutes" in first["sleep_waits_for"]              # she has only just spoken
        assert "new memories are waiting, and it starts at 500" in first["reflection_waits_for"]
        assert sl.sleep_wait(m, dict(cfg, sleep_enabled=False), 0) == "unattended sleep is switched off"
        assert sl.sleep_wait(m, dict(cfg, reflect_model=""), 0) == "no reflection model is set"
        assert r.sleep_if_due() is None                                                          # not quiet yet
        # quiet long enough: it sleeps, and says so while it does
        r.last_activity = time.time() - 600
        r.note()
        assert note()["sleep_waits_for"] == ""
        during = {}

        def asleep(engine, cfg_, **kw):
            during.update(note())
            return real(engine, cfg_, llm=lambda system, user, step: {
                "propose": EMPTY_REFLECTION, "check": EMPTY_REFLECTION, "profiles": EMPTY_REFLECTION,
                "consolidate": json.dumps({"summary": "We talked about the operating system she began twenty years ago and wants to resume."}),
                "dream": json.dumps({"dream": "The operating system is a house and every room boots slowly, one sector at a time, while a child sleeps upstairs."}),
                "wake": json.dumps({"thoughts": "Mostly odd.", "connections": []})}[step], rng=random.Random(1), **kw)
        sl.sleep_once = asleep
        report = r.sleep_if_due()
        assert report and report["dream"]["id"] and during["doing"] == "sleeping" and during["doing_since"]
        after = note()
        assert after["doing"] is None and "hours ago, and there are at least 12 hours between sleeps" in after["sleep_waits_for"]
        assert after["last_sleep_error"] == "" and m.kv_get("sleep:last_run")
        # a sleep that fails outright is recorded, and not tried again every half minute
        m.kv_set("sleep:last_run", "0")
        m.remember("Something new to sleep on about the garden", kind="said_user", session="s2")
        r.last_activity = time.time() - 600
        calls = []

        def broken(engine, cfg_, **kw):
            calls.append(1)
            raise RuntimeError("the dream model fell over")
        sl.sleep_once = broken
        assert r.sleep_if_due() is None and r.sleep_if_due() is None and len(calls) == 1
        failed = note()
        assert failed["last_sleep_error"] == "RuntimeError: the dream model fell over" and failed["doing"] is None
        assert "the last attempt failed; it will try again in" in failed["sleep_waits_for"]
        # one task falling over does not stop the others being tried, and is said
        r._sleep_retry_at = 0.0
        ran = []
        sl.sleep_once = lambda engine, cfg_, **kw: ran.append(1) or {"errors": [], "episodes": [], "dream": None}
        real_images = r.images_if_due
        r.images_if_due = lambda: (_ for _ in ()).throw(KeyError("faces"))
        r.once()
        assert ran == [1] and r.stumbles["describing images"].startswith("KeyError: 'faces' (at ")
        r.note()
        assert "describing images" in note()["stumbles"]
        r.images_if_due = real_images
        r.once()
        assert "describing images" not in r.stumbles
    finally:
        sl.sleep_once = real
        r.stop()


def test_cli_status_shows_the_background_worker(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io
    p = make(tmp_path)
    p.sync_turn("My name is Kayla and I build memory systems for fun", "Nice to meet you, Kayla.", session_id="s1")
    home = tmp_path / "home"
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    try:
        import holonomic.cli as cli, holonomic.embed as embed
        from holonomic.provider import reflector_for
        keep = embed.OllamaEmbedder
        embed.OllamaEmbedder = lambda *x, **k: HashEmbedder()
        parser = argparse.ArgumentParser(); cli.register_cli(parser)

        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                args = parser.parse_args(list(argv)); args.func(args)
            return out.getvalue()
        worker = reflector_for(p._engine)
        worker.note({"sleep_enabled": True, "reflect_model": "gemma", "sleep_idle_seconds": 300, "reflect_enabled": True})
        text = run("sleep", "status")
        assert "in the background: Hermes is running" in text and "sleep is waiting: it has been quiet for" in text
        assert "reflection is waiting:" in run("reflect", "status")
        worker.last_sleep_error = "RuntimeError: the dream model fell over"
        worker._begin("sleeping")
        text = run("sleep", "status")
        assert "it is sleeping now, and has been for" in text and "the last sleep reported: RuntimeError: the dream model fell over" in text
        worker.stumbles["describing images"] = "KeyError: 'faces' (at images.py:1)"
        worker.note()
        assert "describing images cannot even be checked, every half minute: KeyError" in run("sleep", "status")
        worker.stumbles.clear()
        worker._end()
        saved = json.loads(worker._note_path().read_text())
        # the cause of the last activity is said: a turn was stored
        assert "a turn was stored: 'My name is Kayla and I build memory systems for fun'" in run("sleep", "status")
        p.prefetch("What is my name, do you remember it?", session_id="s1")
        worker.note()
        assert "a message came in: 'What is my name, do you remember it?'" in run("sleep", "status")
        # Hermes closed: the last word from the worker is old, and the status says nobody is there
        p.shutdown()
        assert not list((home / "holonomic").glob("worker*.json"))                           # a worker that stops takes its note with it
        assert "in the background: nothing on record" in run("sleep", "status")
        # a program that died without tidying up: its last word is old, and the status says nobody is there
        state = dict(saved, at=time.time() - 3600)
        (home / "holonomic" / "worker.999.json").write_text(json.dumps(state))
        assert "no Hermes is running with this memory right now" in run("sleep", "status")
        # two programs with the memory open are shown apart
        (home / "holonomic" / "worker.1.json").write_text(json.dumps(dict(saved, at=time.time(), pid=1, last_cause="a message came in: 'hello'")))
        (home / "holonomic" / "worker.2.json").write_text(json.dumps(dict(saved, at=time.time(), pid=2, last_activity=time.time() - 240, last_cause="a turn was stored: 'bye'")))
        text = run("sleep", "status")
        assert "2 programs have this memory open" in text and "process 1," in text and "process 2," in text
        assert "a message came in: 'hello'" in text and "quiet for 4.0 minutes" in text and "a turn was stored: 'bye'" in text
    finally:
        embed.OllamaEmbedder = keep
        sys.modules.pop("hermes_constants", None)


def test_a_dream_can_reach_back_through_the_plates(tmp_path):
    """A dream reached into the past by likeness alone.  As an experiment it can also follow the plates: likeness
    finds an old memory, the plates bring what was said around it."""
    from holonomic.sleep import gather_fragments
    m = store(tmp_path)
    now = time.time()
    long_ago = now - 40 * DAY
    a = m.remember("We planted tomatoes along the south fence in spring", kind="said_user", session="garden", created_at=long_ago)[0]
    b = m.remember("The neighbour's dog dug up half of them the very next week", kind="said_user", session="garden", created_at=long_ago + 60)[0]
    m.remember("The tomatoes along the fence are finally ripe this week", kind="said_user", session="today", created_at=now - 3600)
    m.remember("I think I will make sauce from them at the weekend", kind="said_user", session="today", created_at=now - 3500)
    plain = gather_fragments(m, {"dream_seeds": 2}, random.Random(1), now)
    assert a in {f["id"] for f in plain} and b not in {f["id"] for f in plain}                 # by likeness: the planting, not the dog
    assert all(f.get("via") in (None, "likeness") for f in plain)
    linked = gather_fragments(m, {"dream_seeds": 2, "dream_links": 1}, random.Random(1), now)
    got = {f["id"]: f for f in linked}
    # tied to the old planting, or straight to the recent line that resembles it: either way it came off the plates
    assert b in got and got[b]["via"] == "plates" and got[b]["linked_to"] in (a, got[a]["echo_of"]) and got[b]["age"] == "OLDER"
    assert got[a]["via"] == "likeness"
    # nothing recent comes in that way, and it changes nothing
    assert all(f["age"] == "OLDER" for f in linked if f.get("via") == "plates")
    assert m.linked(a, older_than=now - DAY, k=3)[0]["id"] == b and m.linked(a, older_than=long_ago - 1, k=3) == []
    assert [x["id"] for x in m.linked(a, k=3, skip_kinds=("said_user",))] == []


def test_an_old_memory_is_dreamt_of_once_a_night(tmp_path):
    """Recent memories one dream used were kept from the next; the older ones it reached were not, so a well-tied
    old memory turned up in dream after dream of the same sleep."""
    from holonomic.sleep import gather_fragments
    m = store(tmp_path)
    now = time.time()
    long_ago = now - 40 * DAY
    old = [m.remember(t, kind="said_user", session="garden", created_at=long_ago + i * 60)[0] for i, t in enumerate((
        "We planted tomatoes along the south fence in spring",
        "The neighbour's dog dug up half of the tomatoes the very next week",
        "We replanted the tomatoes and put wire around the bed",
        "The tomatoes by the fence needed watering every single evening",
        "A late frost nearly killed the tomato seedlings by the fence",
        "We staked the tomato plants along the fence with bamboo canes",
        "The first green tomatoes appeared on the fence plants in June",
        "Blight got into the tomatoes at the far end of the fence",
        "We fed the tomato bed by the fence with seaweed every fortnight",
        "The tomatoes along the fence ripened all at once in August"))]
    for i, t in enumerate(("The tomatoes along the fence are finally ripe this week",
                           "I think I will make tomato sauce at the weekend",
                           "The tomato plants by the fence have grown taller than me",
                           "I picked a whole basket of tomatoes from the fence bed")):
        m.remember(t, kind="said_user", session=f"today{i}", created_at=now - 3600 + i * 60)
    cfg = {"dream_seeds": 2, "dream_links": 1}
    first = gather_fragments(m, cfg, random.Random(1), now)
    older = {f["id"] for f in first if f["age"] == "OLDER"}
    assert older and older < set(old)
    second = gather_fragments(m, cfg, random.Random(1), now, {f["id"] for f in first})
    again = {f["id"] for f in second}
    assert not again & {f["id"] for f in first}                      # nothing from the first dream, old or new
    assert again & (set(old) - older)                                # and the past is still reached: other old memories step in
