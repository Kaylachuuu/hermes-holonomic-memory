"""Sleep: the long-idle cycle that turns recent experience into lasting memory.

Four steps, each of which can be switched off:

  1. reflect      facts, notes and profiles from what is new (see reflect.py).
  2. consolidate  for each finished conversation, write a short account of it and store
                  that in long-term memory, linked to the conversation it summarises.
  3. fade         conversation that has been summarised loses strength with time.  Nothing
                  is deleted: a faded memory leaves everyday recall and stays reachable by
                  deep recall, which strengthens it again.  What the agent knows about the
                  user does not fade.
  4. dream        recent memories and older ones they echo, including faded ones, are woven
                  into a dream.  The agent then rereads it soberly and notes any real
                  connection it sees.  Dreams and what was made of them live in their own
                  realm: they never enter factual recall and they strengthen nothing, but
                  the agent can recall them and talk about them.

Only stdlib imports here: Hermes executes this file when it loads the plugin.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

from . import reflect as _reflect
from .reflect import DERIVED_KINDS, ReflectionError, _parse, _speaker

logger = logging.getLogger(__name__)

RAW_KINDS = ("said_user", "said_assistant", "asked_user")     # short-term: conversation as it happened
EPISODE = "episode"                                            # long-term: an account of one conversation
DREAM, DREAM_INSIGHT, DREAM_REALM = "dream", "dream_insight", "dream"
LASTING_KINDS = DERIVED_KINDS + (EPISODE, "core", "note")     # never faded

SLEEP_DEFAULTS: Dict[str, Any] = {
    "sleep_enabled": False,           # run the cycle unattended when idle
    "sleep_idle_seconds": 3600,       # the conversation must have been quiet this long
    "sleep_min_hours": 12,            # and this long must have passed since the last sleep
    "consolidate_enabled": True,
    "consolidate_min_memories": 4,    # shorter conversations are not worth an account
    "consolidate_quiet_minutes": 30,  # a conversation counts as finished after this
    "fade_enabled": True,
    "fade_half_life_days": 5.0,       # summarised conversation halves in strength every this many days
    "fade_threshold": 0.35,           # below this a memory is out of everyday recall
    "dream_enabled": True,
    "dream_model": "",                # empty = the reflection model
    "dream_host": "",                 # empty = the reflection server
    "dream_temperature": 1.0,
    "dream_seeds": 4,                 # recent memories a dream starts from
    "dream_days": 3,                  # how far back "recent" reaches
    "dream_in_prompt_days": 3,        # how long the latest dream stays in view
    # Whether a dream strengthens the old memories it touches.  Off: dreams stay in their own realm
    # and bring nothing faded back to the surface.
    "dream_reinforce": False,
}

_ITEMS = {"type": "array", "items": {"type": "object", "properties": {
    "text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
    "required": ["text", "sources"]}}
_EPISODE_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
_DREAM_SCHEMA = {"type": "object", "properties": {"dream": {"type": "string"}}, "required": ["dream"]}
_WAKE_SCHEMA = {"type": "object", "properties": {"thoughts": {"type": "string"}, "connections": _ITEMS},
                "required": ["thoughts", "connections"]}

_EPISODE = """\
Below is one conversation between an AI assistant and its user, oldest line first. USER lines were said by the \
user, ASSISTANT lines by the assistant.

Write "summary": the assistant's own account of this conversation for its long-term memory, in the first person \
("We talked about ...", "She told me ...", "I suggested ..."). Say what was discussed, what the user revealed or \
decided, what the assistant suggested, and what was left open. Use the user's name if it appears.

{rules}

Keep who said what straight: a suggestion the assistant made stays the assistant's suggestion. Plain prose, at \
most 90 words, no list."""

_DREAM = """\
You are an AI assistant, asleep. Below are fragments of your memory: things from the last few days, and older \
things they faintly echo. In a dream these run together.

Each fragment is marked with whose words it is. Things "said to me" are the user's own life, not yours, though \
in a dream you may find yourself inside them.

Write "dream": the dream you have, in the first person and the present tense, 100 to 180 words. Let the fragments \
blend, shift and stand in for one another the way they do in dreams: places change, one thing becomes another, an \
old memory walks into a new one. It should feel like a dream, not a summary, and it does not have to make sense. \
Do not explain it and do not list the fragments."""

_WAKE = """\
You are an AI assistant, now awake, rereading a dream you just had. Below are the dream and the numbered memory \
fragments it was made from. RECENT fragments are from the last few days; OLDER ones are from before. Each is marked \
with whose words it is: things "said to me" are the user's own life and work, not yours. Awake, keep that straight, \
and never write "the user": call them by their name, which is in what you know about them below.

Write:
- "thoughts": one or two plain sentences, first person, on what you make of the dream now that you are awake. No \
grand claims; it is fine to find it merely odd.
- "connections": real links between a RECENT fragment and an OLDER one that the dream put side by side, where the \
two really do bear on each other. One sentence each, citing both in "sources". A dream mostly throws things \
together by accident, so this list is usually empty or has one item. Do not invent a link to have something to say.

{rules}"""


# Whose words a fragment is.  Without this the dreamer reads the user's "I started a project
# twenty years ago" as its own history, and wakes up talking about "my technical work".
_VOICE = {"said_user": "said to me", "asked_user": "asked of me", "said_assistant": "I said", "fact": "I know this about her or him",
          EPISODE: "I remember", "self_note": "about myself", "bond_note": "about the two of us", "insight": "I noticed",
          "note": "I noted", "core": "I noted"}


def _voice(kind: str) -> str:
    return _VOICE.get(kind, "I recall")


def sleep_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(SLEEP_DEFAULTS)
    out.update({k: cfg[k] for k in SLEEP_DEFAULTS if k in cfg})
    return out


def _model_caller(cfg: Dict[str, Any], report: Dict[str, Any], *, dream: bool = False) -> Callable[..., str]:
    rc = _reflect.reflect_config(cfg)
    sc = sleep_config(cfg)
    model = (sc["dream_model"] if dream else "") or rc["reflect_model"]
    host = ((sc["dream_host"] if dream else "") or rc["reflect_host"]).rstrip("/")
    if not model:
        raise ReflectionError("No model is set. Run: hermes holonomic reflect on --model NAME")

    def call(step: str, system: str, user: str, schema: dict, max_tokens: int, temperature: Optional[float] = None) -> str:
        out = _reflect.ollama_chat(host, model, system, user, timeout=float(rc["reflect_timeout"]), max_tokens=max_tokens,
                                   temperature=float(rc["reflect_temperature"]) if temperature is None else temperature,
                                   think=False, schema=schema)
        report["calls"].append(dict(_reflect.LAST_CALL, step=step))
        return out
    return call


def _wrap_test_llm(llm: Callable[..., str]) -> Callable[..., str]:
    return lambda step, system, user, schema, max_tokens, temperature=None: llm(system, user, step)


def _lines(memories: List[dict], max_chars: int = 9000) -> str:
    out, used = [], 0
    for m in memories:
        line = f"{_speaker(m['kind'])}: {' '.join(m['text'].split())}"
        if used + len(line) > max_chars:
            out.append("(the conversation went on; the rest is not shown)")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


# ---------------------------------------------------------------- consolidate

def unfinished_business(engine, cfg: Dict[str, Any], now: Optional[float] = None) -> List[dict]:
    """Conversations that have gone quiet and have something not yet summarised."""
    sc = sleep_config(cfg)
    now = time.time() if now is None else now
    todo = []
    for s in engine.sessions(RAW_KINDS):
        done = int(engine.kv_get("episode:" + s["session"], "0") or 0)
        if s["last_id"] <= done or now - s["ended"] < float(sc["consolidate_quiet_minutes"]) * 60:
            continue
        fresh = engine.session_memories(s["session"], after_id=done, kinds=RAW_KINDS)
        if len(fresh) >= int(sc["consolidate_min_memories"]):
            todo.append(dict(s, memories=fresh))
    return todo


def consolidate(engine, cfg: Dict[str, Any], call: Callable[..., str], report: Dict[str, Any], *, dry_run: bool,
                key_fn: Optional[Callable[[str], List[str]]] = None, now: Optional[float] = None) -> None:
    for s in unfinished_business(engine, cfg, now):
        memories = s["memories"]
        prompt = _EPISODE.format(rules=_reflect._RULES) + "\n\nCONVERSATION:\n" + _lines(memories)
        try:
            summary = " ".join(str(_parse(call("consolidate", _reflect._SYSTEM, prompt, _EPISODE_SCHEMA, 400)).get("summary") or "").split())
        except ReflectionError as exc:
            report["errors"].append(f"consolidate {s['session']}: {exc}")
            continue
        if len(summary) < 30:
            continue
        summary = summary[:900]
        entry = {"session": s["session"], "memories": len(memories), "summary": summary, "when": memories[-1]["created_at"]}
        report["episodes"].append(entry)
        if dry_run:
            continue
        # Linked to the opening and closing of the conversation and to what reflection drew from it.
        ids = {m["id"] for m in memories}
        derived = [m["id"] for m in engine.recent(200) if m["kind"] in DERIVED_KINDS
                   and set((engine.get(m["id"]) or {}).get("meta", {}).get("sources", [])) & ids][:4]
        links = list(dict.fromkeys([memories[0]["id"], memories[-1]["id"]] + derived))
        new = engine.remember(summary, kind=EPISODE, session="episodes", chain=True, links=links,
                              keys=key_fn(summary) if key_fn else [], trust=0.7, salience=1.2,
                              created_at=memories[-1]["created_at"], meta={"session": s["session"], "sources": sorted(ids)[:40]})
        entry["id"] = new[0] if new else None
        engine.kv_set("episode:" + s["session"], str(memories[-1]["id"]))


# ----------------------------------------------------------------------- fade

def fade(engine, cfg: Dict[str, Any], report: Dict[str, Any], *, dry_run: bool, now: Optional[float] = None) -> None:
    """Summarised conversation loses strength by elapsed time, so running the cycle twice
    in an hour fades nothing extra."""
    sc = sleep_config(cfg)
    now = time.time() if now is None else now
    last = float(engine.kv_get("sleep:last_fade", "0") or 0)
    if not last:                                     # first sleep: start the clock, fade nothing yet
        if not dry_run:
            engine.kv_set("sleep:last_fade", str(now))
        return
    factor = 0.5 ** ((now - last) / 86400.0 / max(float(sc["fade_half_life_days"]), 0.1))
    ids = []
    for s in engine.sessions(RAW_KINDS):
        done = int(engine.kv_get("episode:" + s["session"], "0") or 0)
        ids += [m["id"] for m in engine.session_memories(s["session"], kinds=RAW_KINDS) if m["id"] <= done]
    report["fade"] = {"factor": round(factor, 3), "eligible": len(ids), "changed": 0}
    if dry_run or factor > 0.999 or not ids:
        return
    report["fade"]["changed"] = engine.fade(ids, factor)
    engine.kv_set("sleep:last_fade", str(now))


# ---------------------------------------------------------------------- dream

def gather_fragments(engine, cfg: Dict[str, Any], rng: random.Random, now: Optional[float] = None) -> List[dict]:
    """Recent memories to dream from, each with older memories it echoes."""
    sc = sleep_config(cfg)
    now = time.time() if now is None else now
    since = now - float(sc["dream_days"]) * 86400
    recent = [m for m in engine.recent(120, since=since) if m["kind"] not in (DREAM, DREAM_INSIGHT, "asked_user")
              and len(m["text"]) >= 25]
    if len(recent) < 2:
        return []
    # Favour what was strongly written and what the agent or the user actually stated.
    weights = [m["strength"] * (1.5 if m["kind"] in ("said_user", "fact", EPISODE) else 1.0) for m in recent]
    seeds: List[dict] = []
    pool = list(zip(recent, weights))
    while pool and len(seeds) < int(sc["dream_seeds"]):
        pick = rng.choices(range(len(pool)), weights=[w for _, w in pool])[0]
        seeds.append(pool.pop(pick)[0])
    fragments, seen = [], set()
    for seed in seeds:
        if seed["id"] in seen:
            continue
        seen.add(seed["id"])
        fragments.append({"id": seed["id"], "text": seed["text"], "age": "RECENT", "strength": seed["strength"],
                          "kind": seed["kind"]})
        for echo in engine.echoes(seed["id"], older_than=since, k=2):
            if echo["id"] not in seen:
                seen.add(echo["id"])
                fragments.append({"id": echo["id"], "text": echo["text"], "age": "OLDER", "strength": echo["strength"],
                                  "kind": echo["kind"], "echo_of": seed["id"]})
    return fragments


def dream(engine, cfg: Dict[str, Any], call: Callable[..., str], report: Dict[str, Any], *, dry_run: bool,
          rng: Optional[random.Random] = None, now: Optional[float] = None, user_profile: str = "") -> None:
    sc = sleep_config(cfg)
    rng = rng or random.Random()
    fragments = gather_fragments(engine, cfg, rng, now)
    if len(fragments) < 2:
        report["dream"] = {"skipped": "not enough recent memories to dream from"}
        return
    listing = "\n".join(f"- ({_voice(f['kind'])}) {' '.join(f['text'].split())}" for f in rng.sample(fragments, len(fragments)))
    who = engine.profile("user") or user_profile      # in a dry run the profile from this cycle is not stored yet
    try:
        text = " ".join(str(_parse(call("dream", "You are dreaming. Reply with JSON only.", _DREAM + "\n\nFRAGMENTS:\n" + listing,
                                        _DREAM_SCHEMA, 700, float(sc["dream_temperature"]))).get("dream") or "").split())
        if len(text) < 60:
            raise ReflectionError("the model returned no dream")
        numbered = "\n".join(f"[{f['id']}] {f['age']} ({_voice(f['kind'])}): {' '.join(f['text'].split())}" for f in fragments)
        woke = _parse(call("wake", _reflect._SYSTEM, _WAKE.format(rules=_reflect._RULES)
                           + (f"\n\nWHAT YOU KNOW ABOUT THE USER:\n{who}" if who else "")
                           + f"\n\nDREAM:\n{text}\n\nFRAGMENTS:\n{numbered}", _WAKE_SCHEMA, 500, 0.2))
    except ReflectionError as exc:
        report["errors"].append(f"dream: {exc}")
        return
    recent_ids = {f["id"] for f in fragments if f["age"] == "RECENT"}
    older_ids = {f["id"] for f in fragments if f["age"] == "OLDER"}
    connections = [c for c in _reflect._clean_items(woke.get("connections"), recent_ids | older_ids, 2)
                   if set(c["sources"]) & recent_ids and set(c["sources"]) & older_ids]      # must bridge new and old
    thoughts = " ".join(str(woke.get("thoughts") or "").split())[:500]
    report["dream"] = {"text": text[:1800], "thoughts": thoughts, "connections": connections,
                       "fragments": [{"id": f["id"], "age": f["age"], "faded": f["strength"] < float(sc["fade_threshold"])}
                                     for f in fragments]}
    if dry_run:
        return
    now = time.time() if now is None else now
    # Everything a dream produces is stored in the dream realm.  Its links to waking memories are
    # written on dream plates, which waking recall never reads.
    ids = engine.remember(text[:1800], kind=DREAM, realm=DREAM_REALM, session="dreams", chain=True,
                          links=sorted(recent_ids | older_ids)[:6], salience=1.0, trust=0.5, created_at=now,
                          meta={"thoughts": thoughts, "fragments": sorted(recent_ids | older_ids)})
    report["dream"]["id"] = ids[0] if ids else None
    for c in connections:
        engine.remember(c["text"], kind=DREAM_INSIGHT, realm=DREAM_REALM, session="dreams", chain=False,
                        links=(ids[:1] + c["sources"]), trust=0.4, salience=1.0, created_at=now, meta={"sources": c["sources"]})
    if sc["dream_reinforce"] and older_ids:
        engine.reinforce(sorted(older_ids), 0.1)
    if ids:
        engine.kv_set("dream:latest", str(ids[0]))
        engine.kv_set("dream:latest_at", str(now))


def latest_dream(engine, cfg: Dict[str, Any], now: Optional[float] = None) -> Optional[dict]:
    """The most recent dream, if it is recent enough to still be on the agent's mind."""
    try:
        at = float(engine.kv_get("dream:latest_at", "0") or 0)
        mid = int(engine.kv_get("dream:latest", "0") or 0)
    except ValueError:
        return None
    now = time.time() if now is None else now
    if not mid or now - at > float(sleep_config(cfg)["dream_in_prompt_days"]) * 86400:
        return None
    found = engine.get(mid)
    return dict(found, at=at) if found else None


def dreams(engine, n: int = 5) -> List[dict]:
    out = []
    for d in engine.recent(n, realm=DREAM_REALM, kind=DREAM):
        full = engine.get(d["id"]) or {}
        insights = [h.text for h in engine.associates(d["id"], k=4, realms=(DREAM_REALM,), min_score=0.2) if h.kind == DREAM_INSIGHT]
        out.append(dict(d, thoughts=(full.get("meta") or {}).get("thoughts", ""), connections=insights))
    return out


# ---------------------------------------------------------------------- cycle

def sleep_once(engine, cfg: Dict[str, Any], *, llm: Optional[Callable[..., str]] = None, dry_run: bool = False,
               key_fn: Optional[Callable[[str], List[str]]] = None, foundation: str = "",
               steps: Optional[List[str]] = None, rng: Optional[random.Random] = None,
               now: Optional[float] = None) -> Dict[str, Any]:
    """One sleep cycle.  `steps` limits it to some of: reflect, consolidate, fade, dream."""
    sc = sleep_config(cfg)
    wanted = set(steps or ["reflect", "consolidate", "fade", "dream"])
    report: Dict[str, Any] = {"dry_run": dry_run, "calls": [], "errors": [], "reflections": [], "episodes": [],
                              "fade": None, "dream": None}
    if "reflect" in wanted:
        for _ in range(4):                               # catch up on a backlog, a batch at a time
            if _reflect.pending(engine) == 0:
                break
            try:
                r = _reflect.reflect_once(engine, cfg, llm=llm, dry_run=dry_run, key_fn=key_fn, foundation=foundation)
            except ReflectionError as exc:
                report["errors"].append(f"reflect: {exc}")
                break
            report["reflections"].append(r)
            report["calls"] += r.get("calls", [])
            if dry_run:                                  # a dry run does not advance, so one batch is all it can show
                break
    if "consolidate" in wanted and sc["consolidate_enabled"]:
        call = _wrap_test_llm(llm) if llm else _model_caller(cfg, report)
        consolidate(engine, cfg, call, report, dry_run=dry_run, key_fn=key_fn, now=now)
    if "fade" in wanted and sc["fade_enabled"]:
        fade(engine, cfg, report, dry_run=dry_run, now=now)
    if "dream" in wanted and sc["dream_enabled"]:
        call = _wrap_test_llm(llm) if llm else _model_caller(cfg, report, dream=True)
        fresh = next((r["profiles"]["user"] for r in reversed(report["reflections"]) if (r.get("profiles") or {}).get("user")), "")
        dream(engine, cfg, call, report, dry_run=dry_run, rng=rng, now=now, user_profile=fresh)
    if not dry_run:
        engine.kv_set("sleep:last_run", str(time.time() if now is None else now))
    return report


def sleep_due(engine, cfg: Dict[str, Any], last_activity: float, now: Optional[float] = None) -> bool:
    sc = sleep_config(cfg)
    rc = _reflect.reflect_config(cfg)
    now = time.time() if now is None else now
    if not sc["sleep_enabled"] or not rc["reflect_model"]:
        return False
    last = float(engine.kv_get("sleep:last_run", "0") or 0)
    if now - last_activity < float(sc["sleep_idle_seconds"]) or now - last < float(sc["sleep_min_hours"]) * 3600:
        return False
    newest = engine.recent(1)
    return bool(newest) and newest[0]["created_at"] > last          # something has happened since the last sleep
