"""Reflection: a language model reads recent memories and distils what is worth keeping.

One pass takes the memories stored since the last pass and produces
  * facts about the user,
  * the agent's notes about itself,
  * insights connecting several memories,
each stored as a memory linked (through the plates) to the memories it came from,
plus notes about the relationship, and a rewritten short profile of the user, of
the agent and of the two together.  The profiles are always in view (system prompt),
so identity never depends on a search matching.

The user's SOUL.md is shown to the model as a fixed foundation: the agent's view
of itself grows within it and never rewrites it.

Off by default.  Enable with `hermes holonomic reflect on --model NAME`.
Only stdlib imports here: Hermes executes this file when it loads the plugin.
"""

from __future__ import annotations

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
FOUNDATION_MAX_CHARS = 6000
WATERMARK = "reflect:last_id"

REFLECT_DEFAULTS: Dict[str, Any] = {
    "reflect_enabled": False,
    "reflect_model": "",              # e.g. the Ollama name of your chat model
    "reflect_host": "",               # empty = same server as ollama_host
    "reflect_min_new": 12,            # wait until this many new memories exist
    "reflect_idle_seconds": 300,      # and the conversation has been quiet this long
    "reflect_batch": 60,              # memories read per pass
    "reflect_timeout": 300.0,
    "reflect_max_tokens": 2000,       # hard cap on the model's reply; without one a model can generate until its context is full
    "reflect_think": False,           # let a reasoning model think first (much slower)
    "reflect_temperature": 0.3,
    "profile_max_chars": 1200,
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "user_facts": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
            "required": ["text", "sources"]}},
        "self_notes": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
            "required": ["text", "sources"]}},
        "relationship_notes": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
            "required": ["text", "sources"]}},
        "insights": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
            "required": ["text", "sources"]}},
        "superseded": {"type": "array", "items": {"type": "object", "properties": {
            "fact": {"type": "integer"}, "replacement": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "integer"}}},
            "required": ["fact", "replacement", "sources"]}},
        "user_profile": {"type": "string"},
        "self_profile": {"type": "string"},
        "relationship_profile": {"type": "string"},
    },
    "required": ["user_facts", "self_notes", "relationship_notes", "insights", "superseded", "user_profile",
                 "self_profile", "relationship_profile"],
}

_SYSTEM = (
    "You are the reflective process of an AI assistant's long-term memory. You read a log of the assistant's recent "
    "conversation memories and decide what is worth keeping. You never invent anything: every item you write must be "
    "supported by the numbered memories you cite. Reply with JSON only."
)

_INSTRUCTIONS = """\
Below are the assistant's foundation, the current profiles and the newest memories, oldest first. Lines marked \
USER were said by the user, ASSISTANT by the assistant itself, NOTE were stored deliberately.

The FOUNDATION was written by the user. It is fixed: it defines who the assistant is at its core and the guidelines \
it grows within. Never restate or summarise it, and never write anything about the assistant that contradicts it. \
What you write about the assistant is what it has become beyond the foundation, through these conversations.

Write:
- user_facts: durable facts about the user: who they are, what they do, what they want, what they prefer, what is \
going on in their life. One standalone sentence each, in the third person, using the user's name if it is known. \
Skip small talk and anything that only mattered in the moment. At most 8.
- self_notes: what the assistant should remember about itself, written in the first person ("I ..."): commitments \
it made, how it responded and whether that went well, tastes and opinions it has formed. At most 4.
- relationship_notes: what matters about the two of them together, written from the assistant's side ("We ..."): \
shared plans, running jokes, trust, friction, how they work together. At most 3.
- insights: connections or patterns across several memories that no single memory states. At most 3.
- superseded: facts from EXISTING FACTS that are no longer true because the USER has plainly said so in these \
memories. Give "fact" (its number), "replacement" (the fact that is true now, one sentence) and "sources" (the \
USER memories that say so). People rarely change: a bad day, a one-off exception, a joke, or anything the \
ASSISTANT said is not a contradiction. This list is usually empty.
- user_profile: the user profile rewritten to include what is new. Keep everything already in it unless a memory \
contradicts it. Plain prose, at most 150 words.
- self_profile: the assistant's description of who it has become, first person, rewritten the same way. \
At most 120 words.
- relationship_profile: the state of the relationship, from the assistant's side, rewritten the same way. \
At most 100 words.

For every item give "sources": the numbers of the memories that support it. If the memories contain nothing worth \
keeping, return empty lists and repeat the current profiles unchanged. Do not guess, do not flatter, do not pad."""


class ReflectionError(RuntimeError):
    pass


def reflect_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(REFLECT_DEFAULTS)
    out.update({k: cfg[k] for k in REFLECT_DEFAULTS if k in cfg})
    out["reflect_host"] = (out["reflect_host"] or cfg.get("ollama_host") or "http://localhost:11434").rstrip("/")
    return out


LAST_CALL: Dict[str, Any] = {}      # timing and token counts of the most recent model call, for diagnostics


def ollama_chat(host: str, model: str, system: str, user: str, *, timeout: float, temperature: float,
                max_tokens: int = 2000, think: bool = False) -> str:
    payload = {"model": model, "stream": False, "format": _SCHEMA, "think": think,
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
                      "reply_chars": len(content)})
    if data.get("done_reason") == "length":
        raise ReflectionError(f"The model hit the {max_tokens}-token reply limit without finishing. "
                              f"Its reply began: {content[:300]!r}")
    return content


def _parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except ValueError:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                pass
    raise ReflectionError(f"The model did not return valid JSON. Its reply began: {(raw or '')[:300]!r}")


def _speaker(kind: str) -> str:
    return {"said_user": "USER", "asked_user": "USER", "said_assistant": "ASSISTANT"}.get(kind, "NOTE")


def build_prompt(memories: List[dict], profiles: Dict[str, str], foundation: str = "",
                 existing_facts: Optional[List[dict]] = None) -> str:
    lines = [f"[{m['id']}] {_speaker(m['kind'])}: {' '.join(m['text'].split())}" for m in memories]
    facts = "\n".join(f"[{f['id']}] {' '.join(f['text'].split())}" for f in existing_facts or []) or "(none)"
    return (f"{_INSTRUCTIONS}\n\nFOUNDATION:\n{foundation.strip()[:FOUNDATION_MAX_CHARS] or '(none)'}\n\n"
            f"CURRENT USER PROFILE:\n{profiles.get('user') or '(empty)'}\n\n"
            f"CURRENT SELF PROFILE:\n{profiles.get('self') or '(empty)'}\n\n"
            f"CURRENT RELATIONSHIP PROFILE:\n{profiles.get('us') or '(empty)'}\n\n"
            f"EXISTING FACTS ABOUT THE USER (already stored; related to these memories):\n{facts}\n\n"
            "MEMORIES:\n" + "\n".join(lines))


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


def reflect_once(engine, cfg: Dict[str, Any], *, llm: Optional[Callable[[str, str], str]] = None,
                 dry_run: bool = False, key_fn: Optional[Callable[[str], List[str]]] = None,
                 foundation: str = "") -> Dict[str, Any]:
    """One reflection pass.  Returns a report; stores nothing when dry_run is set."""
    rc = reflect_config(cfg)
    if llm is None:
        if not rc["reflect_model"]:
            raise ReflectionError("No reflection model is set. Run: hermes holonomic reflect on --model NAME")
        llm = lambda system, user: ollama_chat(rc["reflect_host"], rc["reflect_model"], system, user,   # noqa: E731
                                               timeout=float(rc["reflect_timeout"]), temperature=float(rc["reflect_temperature"]),
                                               max_tokens=int(rc["reflect_max_tokens"]), think=bool(rc["reflect_think"]))
    last = int(engine.kv_get(WATERMARK, "0") or 0)
    batch = engine.memories_after(last, int(rc["reflect_batch"]), exclude_kinds=DERIVED_KINDS)
    report: Dict[str, Any] = {"read": len(batch), "stored": [], "reinforced": [], "profiles_updated": [], "dry_run": dry_run}
    if not batch:
        return report
    current = {who: engine.profile(who) for who in SUBJECTS}
    user_said = {m["id"] for m in batch if m["kind"] == "said_user"}
    existing = engine.related_of_kind(sorted(user_said), FACT)
    data = _parse(llm(_SYSTEM, build_prompt(batch, current, foundation, existing)))
    valid = {m["id"] for m in batch}
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
    groups = ((FACT, _clean_items(data.get("user_facts"), valid, 8)),
              (SELF_NOTE, _clean_items(data.get("self_notes"), valid, 4)),
              (BOND_NOTE, _clean_items(data.get("relationship_notes"), valid, 3)),
              (INSIGHT, _clean_items(data.get("insights"), valid, 3)))
    report["proposed"] = {kind: items for kind, items in groups}
    limit = int(rc["profile_max_chars"])
    new_profiles = {"user": " ".join(str(data.get("user_profile") or "").split())[:limit],
                    "self": " ".join(str(data.get("self_profile") or "").split())[:limit],
                    "us": " ".join(str(data.get("relationship_profile") or "").split())[:limit]}
    report["profiles"] = new_profiles
    if dry_run:
        return report
    for kind, items in groups:
        for item in items:
            # The same conclusion reached again strengthens the existing memory instead of duplicating it.
            near = engine.recall(item["text"], k=1, min_score=0.0, only_kinds=(kind,))
            if near and near[0].direct >= 0.88:
                engine.reinforce([near[0].id], 0.2)
                report["reinforced"].append(near[0].id)
                continue
            ids = engine.remember(item["text"], kind=kind, session="reflection", chain=False, links=item["sources"],
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
        if len(text) >= 20 and text != current[who]:
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
