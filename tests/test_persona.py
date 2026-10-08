"""Alongside a persona service (persona.py): nothing written in her voice, her accounts stored as hers, her idle
work first, and sleep that gives way when someone starts talking.  Without a service, nothing changes."""
import contextlib, json, os, random, time

from test_provider import HAVE_HERMES, make, tool
from test_reflect import answer, seeded
from test_sleep import DAY, NOW, OS, GARDEN, EMPTY_REFLECTION, store, talk


@contextlib.contextmanager
def service(value="thymos/0.3.0 accounts=1 idle=1"):
    from holonomic.persona import ENV
    old = os.environ.get(ENV)
    os.environ[ENV] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old


def account(home, session, text, **extra):
    folder = home / "plugin-data" / "thymos" / "accounts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{int(time.time() * 1000)}-{session}.json"
    path.write_text(json.dumps(dict({"session_id": session, "account": text, "entry_hash": "abc123",
                                     "model": "ollama/gemma3:12b"}, **extra)), encoding="utf-8")
    return path


def test_the_service_is_found_in_the_process_and_can_be_overridden():
    from holonomic import persona
    os.environ.pop(persona.ENV, None)
    assert persona.service({}) is None and not persona.on({}) and not persona.gives_way({})
    assert persona.on({"persona_service": "on"}) and persona.gives_way({"sleep_gives_way": "on"})
    with service():
        s = persona.service({})
        assert s["name"] == "thymos" and s["version"] == "0.3.0" and s["features"] == {"accounts": "1", "idle": "1"}
        assert persona.on({}) and persona.gives_way({}) and not persona.gives_way({"sleep_gives_way": "off"})
        assert not persona.on({"persona_service": "off"})


def test_reflection_writes_nothing_in_her_voice_with_a_service(tmp_path):
    from holonomic.reflect import reflect_once, SELF_NOTE, BOND_NOTE, FACT
    m, ids = seeded(tmp_path)
    m.set_profile("self", "Written by the reflection model before thymos.")
    seen = {}

    def llm(system, user, step):
        seen[step] = user
        return answer(ids, relationship_notes=[{"text": "We have agreed to build an operating system together.", "sources": [ids["os"]]}],
                      relationship_profile="We have just met.")
    report = reflect_once(m, {}, llm=llm, foundation="You are Athena.", persona=True)
    kinds = {r["kind"] for r in m.recent(20)}
    assert FACT in kinds and SELF_NOTE not in kinds and BOND_NOTE not in kinds
    assert "self_notes" not in seen["propose"] and "relationship_notes" not in seen["propose"] and "insights" in seen["propose"]
    assert "SELF PROFILE" not in seen["profiles"] and "self_profile" not in seen["profiles"] and "USER PROFILE" in seen["profiles"]
    assert report["profiles_updated"] == ["user"] and m.profile("self") == "Written by the reflection model before thymos."
    assert m.profile("us") == ""


def test_her_accounts_are_stored_as_hers_and_let_their_conversation_fade(tmp_path):
    from holonomic import persona
    from holonomic.sleep import EPISODE, sleep_once
    m = store(tmp_path)
    home = tmp_path / "home"
    old = talk(m, "s-old", NOW - 3 * DAY, OS)
    talk(m, "s-other", NOW - 3 * DAY, GARDEN)          # she wrote no account of this one
    path = account(home, "s-old", "We talked about the operating system I have been hearing about, and why it waited.",
                   conversation_ended_at=NOW - 3 * DAY + 4)
    assert persona.waiting_accounts(home) == [path]
    stored = persona.take_accounts(m, home)
    assert len(stored) == 1 and stored[0]["covers"] == 5 and not path.exists()
    ep = m.get(stored[0]["id"])
    assert ep["kind"] == EPISODE and ep["meta"]["author"] == "self" and ep["meta"]["persona"]["entry_hash"] == "abc123"
    assert abs(ep["created_at"] - (NOW - 3 * DAY + 4)) < 1
    assert {old[0], old[-1]} <= {h.id for h in m.associates(stored[0]["id"], k=8)}
    kept = json.loads((path.parent / "stored" / path.name).read_text(encoding="utf-8"))
    assert kept["holonomic"] == dict(kept["holonomic"], how="stored", memory_id=stored[0]["id"])
    assert persona.take_accounts(m, home) == []                                         # each is stored once
    # The conversation she accounted for fades; the one she did not stays as it was said.
    sleep_once(m, {}, steps=["fade"], now=NOW)
    sleep_once(m, {}, steps=["fade"], now=NOW + 10 * DAY)
    strengths = {r["session"]: r["strength"] for r in m.recent(30) if r["kind"] == "said_user"}
    assert strengths["s-old"] < 0.5 and strengths["s-other"] == 1.0


def test_an_empty_account_is_filed_away_and_stores_nothing(tmp_path):
    from holonomic import persona
    m = store(tmp_path)
    home = tmp_path / "home"
    path = account(home, "s1", "   ")
    assert persona.take_accounts(m, home) == [] and not path.exists() and m.stats()["memories"] == 0
    assert json.loads((path.parent / "stored" / path.name).read_text())["holonomic"]["how"] == "empty"


def test_sleep_with_a_service_writes_no_account_and_no_waking_thoughts(tmp_path):
    from holonomic.sleep import DREAM, sleep_once
    m = store(tmp_path)
    talk(m, "s-old", NOW - 2 * DAY, OS)
    steps = []

    def llm(system, user, step):
        steps.append(step)
        return {"propose": EMPTY_REFLECTION, "check": EMPTY_REFLECTION, "profiles": EMPTY_REFLECTION,
                "dream": json.dumps({"dream": "The operating system is a house and every room boots slowly, one sector at a time, while a child sleeps upstairs."})}[step]
    took = []
    report = sleep_once(m, {}, llm=llm, rng=random.Random(1), now=NOW, persona=True,
                        accounts=lambda: took.append(1) or [{"session": "s-old", "id": 99, "covers": 5, "account": "Mine."}])
    assert steps == ["propose", "dream"] and took == [1] and not report["errors"]
    assert report["episodes"] == [{"session": "s-old", "id": 99, "covers": 5, "account": "Mine.", "by": "her"}]
    d = m.get(report["dream"]["id"])
    assert d["kind"] == DREAM and d["meta"]["thoughts"] == "" and d["meta"]["by"] == "memory system"
    assert float(m.kv_get("sleep:last_run")) == NOW


def test_sleep_stops_between_steps_when_someone_starts_talking(tmp_path):
    from holonomic.sleep import sleep_once
    m = store(tmp_path)
    talk(m, "s-old", NOW - 2 * DAY, OS)
    steps, talking = [], []

    def llm(system, user, step):
        steps.append(step)
        talking.append(True)                            # a message arrives while the first step is running
        return {"propose": EMPTY_REFLECTION, "check": EMPTY_REFLECTION, "profiles": EMPTY_REFLECTION}[step]
    report = sleep_once(m, {}, llm=llm, rng=random.Random(1), now=NOW, should_stop=lambda: bool(talking))
    assert steps == ["propose"] and "someone started talking" in report["interrupted"]
    assert report["fade"] is None and report["dream"] is None and report["episodes"] == []
    assert m.kv_get("sleep:last_run") is None            # not counted: the next quiet stretch finishes it


def test_memorys_idle_work_waits_for_hers(tmp_path):
    from holonomic import persona
    home = tmp_path / "home"
    folder = persona.folder(home)
    folder.mkdir(parents=True)
    (folder / "idle.json").write_text(json.dumps({"at": NOW, "due": 2, "running": ""}))
    assert persona.her_turn(home, {}, now=NOW) == ""                                   # no service, no waiting
    with service():
        assert "2 waiting" in persona.her_turn(home, {}, now=NOW)
        (folder / "idle.json").write_text(json.dumps({"at": NOW, "due": 0, "running": "conversation s1"}))
        assert "running now" in persona.her_turn(home, {}, now=NOW)
        assert persona.her_turn(home, {}, now=NOW + 3600) == ""                          # left by a Hermes that is gone
        (folder / "idle.json").write_text(json.dumps({"at": NOW, "due": 0, "running": ""}))
        assert persona.her_turn(home, {}, now=NOW) == ""
    with service("thymos/0.3.0 accounts=1"):                                           # a service that does not say
        (folder / "idle.json").write_text(json.dumps({"at": NOW, "due": 2}))
        assert persona.her_turn(home, {}, now=NOW) == ""


def test_the_worker_stores_her_accounts_and_keeps_its_note_fresh_while_busy(tmp_path):
    from holonomic.reflect import IdleReflector
    m = store(tmp_path / "home")                         # the store lives in home/m, so home is its parent
    home = tmp_path / "home"
    talk(m, "s1", time.time() - DAY, OS)
    account(home, "s1", "We talked about the operating system and why it waited for her daughter.")
    r = IdleReflector(m, lambda: {}, poll_seconds=0.05, settle_seconds=0, spawn=lambda target, name: __import__("threading").Thread(target=lambda: None))
    try:
        assert r.accounts_if_due() is None                                            # no service: left where it is
        with service():
            stored = r.accounts_if_due()
            assert len(stored) == 1 and m.get(stored[0]["id"])["meta"]["author"] == "self"
            r._begin("sleeping")
            first = json.loads(r._note_path().read_text())["at"]
            time.sleep(0.3)
            assert json.loads(r._note_path().read_text())["at"] > first               # still written while it works
            r._end()
    finally:
        r.stop("test over")


def test_without_a_service_her_profiles_and_dreams_are_shown_as_before(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.persona import ENV
    os.environ.pop(ENV, None)
    p = make(tmp_path)
    e = p._engine
    e.set_profile("self", "I like dry humour.")
    e.set_profile("us", "We build things together.")
    assert "Who you have become" in p.system_prompt_block() and "Your relationship with the user" in p.system_prompt_block()


def test_with_a_service_other_models_words_are_not_shown_as_hers(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.sleep import sleep_once
    p = make(tmp_path)
    e = p._engine
    e.set_profile("self", "I like dry humour.")
    e.set_profile("us", "We build things together.")
    e.set_profile("user", "Kayla works in IT.")
    now = time.time()
    talk(e, "s-new", now - 3600, GARDEN)
    sleep_once(e, p._cfg, llm=lambda system, user, step: {
        "dream": json.dumps({"dream": "I am kneeling in the garden and the herbs are growing in rows of assembly, each leaf a line of code."}),
        "wake": json.dumps({"thoughts": "It was odd to see the garden turn into code.", "connections": []})}[step],
        steps=["dream"], rng=random.Random(2))
    note = e.remember("I suggested tomatoes and she has not said yet", kind="self_note", session="reflection", chain=False)[0]
    with service():
        block = p.system_prompt_block()
        assert "Who you have become" not in block and "Your relationship with the user" not in block
        assert "What you know about the user" in block
        assert "composed this from your memories" in block and "you did not write it" in block and "What you made of it" not in block
        asked = p.prefetch("Did you have any dreams last night?", session_id="s9")
        assert "not written by you" in asked and "What you made of it" not in asked
        listed = tool(p, action="dreams")
        assert "what_you_made_of_it" not in listed["dreams"][0] and "did not write them" in listed["note"]
        found = tool(p, action="recall", query="I suggested tomatoes", subject="self")
        assert any(h["id"] == note and h["written_by"].startswith("the memory system") for h in found["results"])
