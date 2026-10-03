"""Reflection: a language model reads recent memories and distils what is worth keeping.

One pass takes the memories stored since the last pass and works in up to three
small steps, each a separate model call with one job:

  1. propose   read the memories; write facts about the user, the agent's notes about
               itself, notes about the relationship, insights, and any stored fact the
               user has contradicted.  Every item cites the memories it rests on.
  2. check     (depth 3) re-read each proposed item against only the lines it cites;
               keep it, rewrite it, or drop it.
  3. profiles  rewrite the short profiles of the user, the agent and the two together
               from the items that survived, never from the raw conversation.

Asking for all of this in one call overloaded a small model: it recorded its own
phrases as the user's values, and with reasoning switched on it deliberated until
it ran out of tokens without answering.  Small steps keep each call simple.

Surviving items are stored as memories linked (through the plates) to the memories
they came from.  The profiles are always in view (system prompt), so identity never
depends on a search matching.  The user's SOUL.md is shown to the model as a fixed
foundation: the agent's view of itself grows within it and never rewrites it.

Off by default.  Enable with `hermes holonomic reflect on --model NAME`.
Only stdlib imports here: Hermes executes this file when it loads the plugin.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Kinds produced by reflection.  They are never fed back in as input.
FACT, SELF_NOTE, BOND_NOTE, INSIGHT = "fact", "self_note", "bond_note", "insight"
DERIVED_KINDS = (FACT, SELF_NOTE, BOND_NOTE, INSIGHT, "dream")
# Subjects the agent can ask its memory about, and the profile and kind that belong to each.
SUBJECTS = {"user": FACT, "self": SELF_NOTE, "us": BOND_NOTE}
FOUNDATION_MAX_CHARS = 20000      # a real SOUL.md ran to 10,000 characters; 6,000 cut it off mid-section
WATERMARK = "reflect:last_id"

REFLECT_DEFAULTS: Dict[str, Any] = {
    "reflect_enabled": False,
    "reflect_model": "",              # e.g. the Ollama name of your chat model
    "reflect_host": "",               # empty = same server as ollama_host
    # 1 = facts about the user and the user profile only
    # 2 = everything, proposed and stored unchecked
    # 3 = everything, with each item checked against the lines it cites before it is stored
    "reflect_depth": 3,
    "reflect_min_new": 12,            # wait until this many new memories exist
    "reflect_idle_seconds": 300,      # and the conversation has been quiet this long
    "reflect_batch": 60,              # memories read per pass
    "reflect_timeout": 600.0,
    "reflect_max_tokens": 2000,       # cap on each reply; without one a model can generate until its context is full
    # Let a reasoning model think first.  On a real 43-memory batch this took 98 s instead of 24 s and
    # found 8 well-separated facts instead of 3.  If thinking uses up the token budget without producing
    # an answer, that step is repeated without it.
    "reflect_think": True,
    "reflect_temperature": 0.3,
    "profile_max_chars": 1200,
}

_ITEMS = {"type": "array", "items": {"type": "object", "properties": {
    "text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
    "required": ["text", "sources"]}}
_SUPERSEDED = {"type": "array", "items": {"type": "object", "properties": {
    "fact": {"type": "integer"}, "replacement": {"type": "string"},
    "sources": {"type": "array", "items": {"type": "integer"}}}, "required": ["fact", "replacement", "sources"]}}

_PROPOSE_SCHEMA = {"type": "object", "properties": {
    "user_facts": _ITEMS, "self_notes": _ITEMS, "relationship_notes": _ITEMS, "insights": _ITEMS, "superseded": _SUPERSEDED},
    "required": ["user_facts", "self_notes", "relationship_notes", "insights", "superseded"]}
_PROPOSE_BASIC_SCHEMA = {"type": "object", "properties": {"user_facts": _ITEMS, "superseded": _SUPERSEDED},
                         "required": ["user_facts", "superseded"]}
_CHECK_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "item": {"type": "integer"}, "verdict": {"type": "string", "enum": ["keep", "rewrite", "drop"]},
    "text": {"type": "string"}}, "required": ["item", "verdict", "text"]}}}, "required": ["verdicts"]}
_PROFILE_SCHEMA = {"type": "object", "properties": {
    "user_profile": {"type": "string"}, "self_profile": {"type": "string"}, "relationship_profile": {"type": "string"}},
    "required": ["user_profile", "self_profile", "relationship_profile"]}
# Used when a caller asks for no particular step (kept for tests and older callers).
_SCHEMA = _PROPOSE_SCHEMA

_SYSTEM = (
    "You are the reflective process of an AI assistant's long-term memory. You never invent anything: everything "
    "you write must be supported by the numbered lines you are shown. Reply with JSON only."
)

_RULES = """\
Words and ideas belong to whoever said them. A phrase the ASSISTANT coined, a suggestion it made or a judgement it \
offered is not the user's value, plan or opinion unless a USER line takes it up. Never put the assistant's phrases \
in quotation marks as though the user had said them.

State what is, not how admirable it is. No praise and no flattering adjectives about either of them ("highly \
skilled", "dedicated", "ambitious", "nuanced", "keen"), and no conclusions the lines do not state."""

_FACTS_RULE = """\
- user_facts: durable facts about the user as a person: their history, work, skills, projects, family, tastes, \
plans, what is going on in their life. One standalone sentence each, in the third person, using the user's name \
if it is known. Be specific and keep the details and the reasons: write "Kayla stopped working on her operating \
system when her daughter was born", not "Kayla has a daughter" or "Kayla has a project". Write a fact for every \
concrete thing the user revealed about themselves. Every fact must rest on at least one USER line. Do not write \
facts about the conversation itself: that the user said hello, is excited to talk, or asked a question is not a \
fact about them. At most 10.
- superseded: facts from EXISTING FACTS that are no longer true because the USER has plainly said so in these \
memories. Give "fact" (its number), "replacement" (the fact that is true now, one sentence) and "sources" (the \
USER memories that say so). People rarely change: a bad day, a one-off exception, a joke, or anything the \
ASSISTANT said is not a contradiction. This list is usually empty."""

_PROPOSE = """\
Below are the assistant's foundation and its newest memories, oldest first. Lines marked USER were said by the \
user, ASSISTANT by the assistant itself, NOTE were stored deliberately.

The FOUNDATION was written by the user. It is fixed: it defines who the assistant is at its core. Never restate \
it, and never write anything about the assistant that contradicts it.

{rules}

Write:
{facts}
- self_notes: what the assistant learned about itself, written in the first person ("I ..."): an opinion or taste \
of its own that it voiced, something it found interesting, a commitment it made, a mistake or something that went \
well. Not a log of what it said, and not anything about the user. If there is nothing of that kind, leave it \
empty. At most 4.
- relationship_notes: what matters about the two of them together, written from the assistant's side ("We ..."): \
shared plans, running jokes, trust, friction, how they work together. At most 3.
- insights: connections or patterns across several memories that no single memory states. At most 3.

For every item give "sources": the numbers of the memories that support it. Empty lists are a good answer when \
there is nothing worth keeping. Do not guess and do not pad."""

_PROPOSE_BASIC = """\
Below are an AI assistant's newest memories, oldest first. Lines marked USER were said by the user, ASSISTANT by \
the assistant itself, NOTE were stored deliberately.

{rules}

Write:
{facts}

For every item give "sources": the numbers of the memories that support it. Empty lists are a good answer when \
there is nothing worth keeping. Do not guess and do not pad."""

_CHECK = """\
Below are statements an assistant's memory is about to store, each followed by the only lines it cites as support. \
Check each statement against its own lines and nothing else.

{rules}

For each statement give a verdict:
- "keep": every part of it is supported by its lines, and it is attributed to the right person.
- "rewrite": part of it is supported. Give the corrected sentence in "text", keeping only what the lines support, \
with the same subject and the same grammatical person.
- "drop": its lines do not support it; or it is a fact about the user that rests only on ASSISTANT lines; or it is \
about the conversation itself and not about the person; or, for a self note, it is really about the user.

Give one verdict per statement, with "item" set to its number. For "keep" and "drop", "text" may be empty."""

_PROFILES = """\
Rewrite an AI assistant's three short profiles. You are given the assistant's foundation, the current profiles, and \
the statements its memory has just accepted. Work only from those; you are not shown the conversation.

The FOUNDATION was written by the user and is fixed. Never restate it, and never write anything about the \
assistant that contradicts it.

{rules}

Write:
- user_profile: a portrait of the user as a person, including what is new. Describe the person, not the \
conversation, and never call them "a user". Keep everything already in it unless a new statement contradicts it. \
Plain prose, at most 150 words.
- self_profile: who the assistant has become beyond the FOUNDATION, first person: its own opinions, tastes, \
interests, humour and habits, as they actually showed up. Leave out its name, its role and anything else the \
FOUNDATION already says. Do not describe the assistant by what it understands about the user or how it supports \
the user. If the statements show nothing of its own, repeat the current self profile exactly, or return an empty \
string if there is none. At most 120 words.
- relationship_profile: the state of the relationship, from the assistant's side. If it is too early to say \
anything specific, repeat the current one exactly, or return an empty string. At most 100 words."""

_KIND_TITLE = {FACT: "fact about the user", SELF_NOTE: "the assistant's note about itself",
               BOND_NOTE: "note about the two of them", INSIGHT: "insight"}


class ReflectionError(RuntimeError):
    pass


def reflect_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(REFLECT_DEFAULTS)
    out.update({k: cfg[k] for k in REFLECT_DEFAULTS if k in cfg})
    out["reflect_host"] = (out["reflect_host"] or cfg.get("ollama_host") or "http://localhost:11434").rstrip("/")
    out["reflect_depth"] = max(1, min(int(out["reflect_depth"] or 3), 3))
    return out


LAST_CALL: Dict[str, Any] = {}      # timing and token counts of the most recent model call, for diagnostics


def ollama_chat(host: str, model: str, system: str, user: str, *, timeout: float, temperature: float,
                max_tokens: int = 2000, think: bool = False, schema: Optional[dict] = None) -> str:
    payload = {"model": model, "stream": False, "format": schema or _SCHEMA, "think": think,
               "options": {"temperature": temperature, "num_predict": int(max_tokens)},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}

    def post(body: dict) -> dict:
        req = urllib.request.Request(host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    started = time.time()
    try:
        try:
            data = post(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 400 and "think" in detail.lower():      # model has no thinking switch
                payload.pop("think")
                data = post(payload)
            else:
                raise ReflectionError(f"Ollama returned {exc.code} for model '{model}': {detail}") from exc
    except TimeoutError as exc:
        raise ReflectionError(f"Model '{model}' did not finish within {timeout:.0f} s") from exc
    except urllib.error.HTTPError as exc:
        raise ReflectionError(f"Ollama returned {exc.code} for model '{model}': {exc.read().decode(errors='replace')[:300]}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        if "timed out" in str(exc).lower():
            raise ReflectionError(f"Model '{model}' did not finish within {timeout:.0f} s") from exc
        raise ReflectionError(f"Could not reach model '{model}' at {host}: {exc}") from exc
    content = (data.get("message") or {}).get("content") or ""
    LAST_CALL.clear()
    LAST_CALL.update({"seconds": time.time() - started, "prompt_tokens": data.get("prompt_eval_count"),
                      "reply_tokens": data.get("eval_count"), "done_reason": data.get("done_reason"),
                      "reply_chars": len(content), "think": think})
    if data.get("done_reason") == "length":
        raise ReflectionError(f"The model hit the {max_tokens}-token reply limit without finishing. "
                              f"Its reply began: {content[:300]!r}")
    return content


def _parse(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except ValueError:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except ValueError:
                pass
    raise ReflectionError(f"The model did not return valid JSON. Its reply began: {(raw or '')[:300]!r}")


def _speaker(kind: str) -> str:
    return {"said_user": "USER", "asked_user": "USER", "said_assistant": "ASSISTANT"}.get(kind, "NOTE")


def _line(memory: dict) -> str:
    return f"[{memory['id']}] {_speaker(memory['kind'])}: {' '.join(memory['text'].split())}"


def _foundation(text: str) -> str:
    return text.strip()[:FOUNDATION_MAX_CHARS] or "(none)"


def build_prompt(memories: List[dict], profiles: Optional[Dict[str, str]] = None, foundation: str = "",
                 existing_facts: Optional[List[dict]] = None, depth: int = 3) -> str:
    """The propose step.  (`profiles` is unused here; profiles are written in their own step.)"""
    facts = "\n".join(f"[{f['id']}] {' '.join(f['text'].split())}" for f in existing_facts or []) or "(none)"
    tail = (f"EXISTING FACTS ABOUT THE USER (already stored; related to these memories):\n{facts}\n\n"
            "MEMORIES:\n" + "\n".join(_line(m) for m in memories))
    if depth <= 1:
        return _PROPOSE_BASIC.format(rules=_RULES, facts=_FACTS_RULE) + "\n\n" + tail
    return _PROPOSE.format(rules=_RULES, facts=_FACTS_RULE) + f"\n\nFOUNDATION:\n{_foundation(foundation)}\n\n" + tail


def build_check_prompt(items: List[dict], by_id: Dict[int, dict]) -> str:
    blocks = []
    for n, item in enumerate(items, 1):
        cited = "\n".join("    " + _line(by_id[s]) for s in item["sources"] if s in by_id)
        blocks.append(f"STATEMENT {n} ({_KIND_TITLE[item['kind']]}): {item['text']}\n  cites:\n{cited}")
    return _CHECK.format(rules=_RULES) + "\n\n" + "\n\n".join(blocks)


def build_profile_prompt(accepted: List[dict], superseded: List[dict], profiles: Dict[str, str], foundation: str,
                         depth: int = 3) -> str:
    lines = [f"- ({_KIND_TITLE[item['kind']]}) {item['text']}" for item in accepted]
    lines += [f"- (no longer true) {s['was']}  Now: {s['replacement']}" for s in superseded]
    return (_PROFILES.format(rules=_RULES)
            + f"\n\nFOUNDATION:\n{_foundation(foundation) if depth > 1 else '(not shown at this depth; leave the self and relationship profiles as they are)'}"
            + f"\n\nCURRENT USER PROFILE:\n{profiles.get('user') or '(empty)'}"
            + f"\n\nCURRENT SELF PROFILE:\n{profiles.get('self') or '(empty)'}"
            + f"\n\nCURRENT RELATIONSHIP PROFILE:\n{profiles.get('us') or '(empty)'}"
            + "\n\nNEWLY ACCEPTED STATEMENTS:\n" + ("\n".join(lines) or "(none)"))


def read_foundation(hermes_home) -> str:
    """The user's SOUL.md: the fixed identity Hermes loads into every session.  Read-only here."""
    try:
        from pathlib import Path
        return (Path(str(hermes_home)) / "SOUL.md").read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


def _clean_items(items: Any, valid_ids: set, limit: int) -> List[dict]:
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").split())
        sources = [int(s) for s in item.get("sources") or [] if isinstance(s, (int, float)) and int(s) in valid_ids]
        # An item with no valid source is unsupported by anything the model was shown.
        if not (15 <= len(text) <= 400) or not sources or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append({"text": text, "sources": list(dict.fromkeys(sources))[:4]})
        if len(out) >= limit:
            break
    return out


def _make_llm(rc: Dict[str, Any], report: Dict[str, Any]) -> Callable[[str, str, str, dict, int], str]:
    """Model caller for one named step.  If reasoning eats the whole token budget
    without an answer, the step is repeated once with reasoning off."""
    def call(step: str, system: str, user: str, schema: dict, max_tokens: int) -> str:
        kwargs = dict(timeout=float(rc["reflect_timeout"]), temperature=float(rc["reflect_temperature"]), schema=schema)
        think = bool(rc["reflect_think"])
        try:
            out = ollama_chat(rc["reflect_host"], rc["reflect_model"], system, user, think=think,
                              max_tokens=max_tokens * (4 if think else 1), **kwargs)
            report["calls"].append(dict(LAST_CALL, step=step))
            return out
        except ReflectionError as exc:
            report["calls"].append(dict(LAST_CALL, step=step, failed=str(exc)[:120]))
            if not think or "reply limit" not in str(exc):
                raise
        out = ollama_chat(rc["reflect_host"], rc["reflect_model"], system, user, think=False, max_tokens=max_tokens, **kwargs)
        report["calls"].append(dict(LAST_CALL, step=step + " (repeated without thinking)"))
        return out
    return call


def _wrap_test_llm(llm: Callable[..., str]) -> Callable[[str, str, str, dict, int], str]:
    """Accept a plain `llm(system, user)` or `llm(system, user, step)` callable."""
    takes_step = len(inspect.signature(llm).parameters) >= 3
    return lambda step, system, user, schema, max_tokens: llm(system, user, step) if takes_step else llm(system, user)


def reflect_once(engine, cfg: Dict[str, Any], *, llm: Optional[Callable[..., str]] = None,
                 dry_run: bool = False, key_fn: Optional[Callable[[str], List[str]]] = None,
                 foundation: str = "") -> Dict[str, Any]:
    """One reflection pass.  Returns a report; stores nothing when dry_run is set."""
    rc = reflect_config(cfg)
    depth = rc["reflect_depth"]
    report: Dict[str, Any] = {"read": 0, "stored": [], "reinforced": [], "profiles_updated": [], "dry_run": dry_run,
                              "depth": depth, "calls": [], "checked": []}
    if llm is None:
        if not rc["reflect_model"]:
            raise ReflectionError("No reflection model is set. Run: hermes holonomic reflect on --model NAME")
        call = _make_llm(rc, report)
    else:
        call = _wrap_test_llm(llm)
    budget = int(rc["reflect_max_tokens"])
    last = int(engine.kv_get(WATERMARK, "0") or 0)
    batch = engine.memories_after(last, int(rc["reflect_batch"]), exclude_kinds=DERIVED_KINDS)
    report["read"] = len(batch)
    if not batch:
        return report
    by_id = {m["id"]: m for m in batch}
    valid = set(by_id)
    current = {who: engine.profile(who) for who in SUBJECTS}
    user_said = {m["id"] for m in batch if m["kind"] == "said_user"}
    existing = engine.related_of_kind(sorted(user_said), FACT)

    # ---- step 1: propose
    data = _parse(call("propose", _SYSTEM, build_prompt(batch, current, foundation, existing, depth),
                       _PROPOSE_BASIC_SCHEMA if depth <= 1 else _PROPOSE_SCHEMA, budget))
    # A stored fact is retired only by something the user said, and only if it was actually shown.
    shown = {f["id"]: f["text"] for f in existing}
    superseded = []
    for item in data.get("superseded") if isinstance(data.get("superseded"), list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("fact"), (int, float)) or int(item["fact"]) not in shown:
            continue
        sources = [int(s) for s in item.get("sources") or [] if isinstance(s, (int, float)) and int(s) in user_said]
        replacement = " ".join(str(item.get("replacement") or "").split())
        if sources and 15 <= len(replacement) <= 400:
            superseded.append({"fact": int(item["fact"]), "was": shown[int(item["fact"])], "replacement": replacement,
                               "sources": list(dict.fromkeys(sources))[:4]})
    report["superseded"] = superseded
    # A fact about the user must rest on something that did not come from the assistant's own mouth.
    not_assistant = {m["id"] for m in batch if m["kind"] != "said_assistant"}
    facts = _clean_items(data.get("user_facts"), valid, 10)
    report["dropped_assistant_only"] = [f["text"] for f in facts if not set(f["sources"]) & not_assistant]
    facts = [f for f in facts if set(f["sources"]) & not_assistant]
    groups = [(FACT, facts)]
    if depth > 1:
        groups += [(SELF_NOTE, _clean_items(data.get("self_notes"), valid, 4)),
                   (BOND_NOTE, _clean_items(data.get("relationship_notes"), valid, 3)),
                   (INSIGHT, _clean_items(data.get("insights"), valid, 3))]
    items = [dict(item, kind=kind) for kind, group in groups for item in group]

    # ---- step 2: check each item against only the lines it cites
    if depth >= 3 and items:
        verdicts = _parse(call("check", _SYSTEM, build_check_prompt(items, by_id), _CHECK_SCHEMA, budget)).get("verdicts")
        decided: Dict[int, dict] = {}
        for v in verdicts if isinstance(verdicts, list) else []:
            if isinstance(v, dict) and isinstance(v.get("item"), (int, float)) and 1 <= int(v["item"]) <= len(items):
                decided[int(v["item"])] = v
        kept = []
        for n, item in enumerate(items, 1):
            v = decided.get(n) or {}
            verdict = str(v.get("verdict") or "keep").lower()      # an item the checker skipped is left as proposed
            new_text = " ".join(str(v.get("text") or "").split())
            if verdict == "drop":
                report["checked"].append({"verdict": "drop", "kind": item["kind"], "was": item["text"]})
                continue
            if verdict == "rewrite" and 15 <= len(new_text) <= 400 and new_text != item["text"]:
                report["checked"].append({"verdict": "rewrite", "kind": item["kind"], "was": item["text"], "now": new_text})
                item = dict(item, text=new_text)
            kept.append(item)
        items = kept
    report["proposed"] = {kind: [{"text": i["text"], "sources": i["sources"]} for i in items if i["kind"] == kind]
                          for kind in (FACT, SELF_NOTE, BOND_NOTE, INSIGHT)}

    # ---- step 3: profiles, written from the accepted items only
    limit = int(rc["profile_max_chars"])
    new_profiles = dict(current)
    if items or superseded:
        pdata = _parse(call("profiles", _SYSTEM, build_profile_prompt(items, superseded, current, foundation, depth),
                            _PROFILE_SCHEMA, budget))
        wanted = {"user": "user_profile"} if depth <= 1 else {"user": "user_profile", "self": "self_profile",
                                                              "us": "relationship_profile"}
        for who, key in wanted.items():
            text = " ".join(str(pdata.get(key) or "").split())[:limit]
            if len(text) >= 20:
                new_profiles[who] = text
    report["profiles"] = new_profiles
    if dry_run:
        return report

    for item in items:
        # The same conclusion reached again strengthens the existing memory instead of duplicating it.
        near = engine.recall(item["text"], k=1, min_score=0.0, only_kinds=(item["kind"],))
        if near and near[0].direct >= 0.88:
            engine.reinforce([near[0].id], 0.2)
            report["reinforced"].append(near[0].id)
            continue
        ids = engine.remember(item["text"], kind=item["kind"], session="reflection", chain=False, links=item["sources"],
                              keys=key_fn(item["text"]) if key_fn else [], trust=0.6, salience=1.1,
                              meta={"sources": item["sources"]})
        report["stored"].extend(ids)
    for item in superseded:
        new_ids = engine.remember(item["replacement"], kind=FACT, session="reflection", chain=False,
                                  links=item["sources"] + [item["fact"]], keys=key_fn(item["replacement"]) if key_fn else [],
                                  trust=0.6, salience=1.1, meta={"sources": item["sources"], "replaces": item["fact"]})
        engine.supersede(item["fact"], new_ids[0] if new_ids else None, reason=item["replacement"])
        report["stored"].extend(new_ids)
    for who, text in new_profiles.items():
        if text != current[who]:
            engine.set_profile(who, text)
            report["profiles_updated"].append(who)
    engine.kv_set(WATERMARK, str(batch[-1]["id"]))
    engine.kv_set("reflect:last_run", str(time.time()))
    return report


def pending(engine) -> int:
    return engine.count_after(int(engine.kv_get(WATERMARK, "0") or 0), exclude_kinds=DERIVED_KINDS)


class IdleReflector:
    """Runs reflection in the background once enough new memories have piled up and
    the conversation has gone quiet, so it does not compete with a reply for the GPU."""

    def __init__(self, engine, load_cfg: Callable[[], Dict[str, Any]], key_fn: Optional[Callable[[str], List[str]]] = None,
                 spawn: Optional[Callable[..., threading.Thread]] = None, poll_seconds: float = 30.0,
                 foundation_fn: Optional[Callable[[], str]] = None):
        self.engine, self.load_cfg, self.key_fn = engine, load_cfg, key_fn
        self.foundation_fn = foundation_fn or (lambda: "")
        self.poll_seconds = poll_seconds
        self.last_activity = time.time()
        self.last_error = ""
        self._stop = threading.Event()
        self._busy = threading.Lock()
        maker = spawn or (lambda target, name: threading.Thread(target=target, name=name, daemon=True))
        self._thread = maker(target=self._loop, name="holonomic-reflect")
        self._thread.start()

    def touch(self) -> None:
        self.last_activity = time.time()

    def stop(self) -> None:
        self._stop.set()

    def due(self, cfg: Dict[str, Any]) -> bool:
        rc = reflect_config(cfg)
        return (bool(rc["reflect_enabled"]) and bool(rc["reflect_model"])
                and time.time() - self.last_activity >= float(rc["reflect_idle_seconds"])
                and pending(self.engine) >= int(rc["reflect_min_new"]))

    def run_if_due(self) -> Optional[Dict[str, Any]]:
        cfg = self.load_cfg()
        if not self.due(cfg) or not self._busy.acquire(blocking=False):
            return None
        try:
            report = reflect_once(self.engine, cfg, key_fn=self.key_fn, foundation=self.foundation_fn())
            self.last_error = ""
            logger.info("holonomic: reflection read %d memories, stored %d", report["read"], len(report["stored"]))
            return report
        except Exception as exc:
            self.last_error = str(exc)
            self.last_activity = time.time()          # back off for one idle period before retrying
            logger.warning("holonomic: reflection failed: %s", exc)
            return None
        finally:
            self._busy.release()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.run_if_due()
            except Exception as exc:                   # never let the worker die
                logger.debug("holonomic: reflector loop error: %s", exc)
