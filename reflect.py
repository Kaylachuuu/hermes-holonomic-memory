"""Reflection: a language model reads recent memories and distils what is worth keeping.

One pass takes the memories stored since the last pass and produces
  * facts about the user,
  * the agent's notes about itself,
  * insights connecting several memories,
each stored as a memory linked (through the plates) to the memories it came from,
plus a rewritten short profile of the user and of the agent.  The profiles are
always in view (system prompt), so identity never depends on a search matching.

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
FACT, SELF_NOTE, INSIGHT = "fact", "self_note", "insight"
DERIVED_KINDS = (FACT, SELF_NOTE, INSIGHT, "dream")
WATERMARK = "reflect:last_id"

REFLECT_DEFAULTS: Dict[str, Any] = {
    "reflect_enabled": False,
    "reflect_model": "",              # e.g. the Ollama name of your chat model
    "reflect_host": "",               # empty = same server as ollama_host
    "reflect_min_new": 12,            # wait until this many new memories exist
    "reflect_idle_seconds": 300,      # and the conversation has been quiet this long
    "reflect_batch": 60,              # memories read per pass
    "reflect_timeout": 600.0,
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
        "insights": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
            "required": ["text", "sources"]}},
        "user_profile": {"type": "string"},
        "self_profile": {"type": "string"},
    },
    "required": ["user_facts", "self_notes", "insights", "user_profile", "self_profile"],
}

_SYSTEM = (
    "You are the reflective process of an AI assistant's long-term memory. You read a log of the assistant's recent "
    "conversation memories and decide what is worth keeping. You never invent anything: every item you write must be "
    "supported by the numbered memories you cite. Reply with JSON only."
)

_INSTRUCTIONS = """\
Below are the current profiles and the newest memories, oldest first. Lines marked USER were said by the user, \
ASSISTANT by the assistant itself, NOTE were stored deliberately.

Write:
- user_facts: durable facts about the user: who they are, what they do, what they want, what they prefer, what is \
going on in their life. One standalone sentence each, in the third person, using the user's name if it is known. \
Skip small talk and anything that only mattered in the moment. At most 8.
- self_notes: what the assistant should remember about itself, written in the first person ("I ..."): commitments \
it made, how it responded and whether that went well, how the relationship with the user is developing. At most 4.
- insights: connections or patterns across several memories that no single memory states. At most 3.
- user_profile: the user profile rewritten to include what is new. Keep everything already in it unless a memory \
contradicts it. Plain prose, at most 150 words.
- self_profile: the assistant's description of itself, first person, rewritten the same way. At most 120 words.

For every item give "sources": the numbers of the memories that support it. If the memories contain nothing worth \
keeping, return empty lists and repeat the current profiles unchanged. Do not guess, do not flatter, do not pad."""


class ReflectionError(RuntimeError):
    pass


def reflect_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(REFLECT_DEFAULTS)
    out.update({k: cfg[k] for k in REFLECT_DEFAULTS if k in cfg})
    out["reflect_host"] = (out["reflect_host"] or cfg.get("ollama_host") or "http://localhost:11434").rstrip("/")
    return out


def ollama_chat(host: str, model: str, system: str, user: str, *, timeout: float, temperature: float) -> str:
    payload = {"model": model, "stream": False, "format": _SCHEMA, "options": {"temperature": temperature},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(host + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["message"]["content"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise ReflectionError(f"Ollama returned {exc.code} for model '{model}': {detail}") from exc
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        raise ReflectionError(f"Could not reach model '{model}' at {host}: {exc}") from exc


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
    raise ReflectionError("The model did not return valid JSON")


def _speaker(kind: str) -> str:
    return {"said_user": "USER", "asked_user": "USER", "said_assistant": "ASSISTANT"}.get(kind, "NOTE")


def build_prompt(memories: List[dict], user_profile: str, self_profile: str) -> str:
    lines = [f"[{m['id']}] {_speaker(m['kind'])}: {' '.join(m['text'].split())}" for m in memories]
    return (f"{_INSTRUCTIONS}\n\nCURRENT USER PROFILE:\n{user_profile or '(empty)'}\n\n"
            f"CURRENT SELF PROFILE:\n{self_profile or '(empty)'}\n\nMEMORIES:\n" + "\n".join(lines))


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
                 dry_run: bool = False, key_fn: Optional[Callable[[str], List[str]]] = None) -> Dict[str, Any]:
    """One reflection pass.  Returns a report; stores nothing when dry_run is set."""
    rc = reflect_config(cfg)
    if llm is None:
        if not rc["reflect_model"]:
            raise ReflectionError("No reflection model is set. Run: hermes holonomic reflect on --model NAME")
        llm = lambda system, user: ollama_chat(rc["reflect_host"], rc["reflect_model"], system, user,   # noqa: E731
                                               timeout=float(rc["reflect_timeout"]), temperature=float(rc["reflect_temperature"]))
    last = int(engine.kv_get(WATERMARK, "0") or 0)
    batch = engine.memories_after(last, int(rc["reflect_batch"]), exclude_kinds=DERIVED_KINDS)
    report: Dict[str, Any] = {"read": len(batch), "stored": [], "reinforced": [], "profiles_updated": [], "dry_run": dry_run}
    if not batch:
        return report
    user_profile, self_profile = engine.profile("user"), engine.profile("self")
    data = _parse(llm(_SYSTEM, build_prompt(batch, user_profile, self_profile)))
    valid = {m["id"] for m in batch}
    groups = ((FACT, _clean_items(data.get("user_facts"), valid, 8)),
              (SELF_NOTE, _clean_items(data.get("self_notes"), valid, 4)),
              (INSIGHT, _clean_items(data.get("insights"), valid, 3)))
    report["proposed"] = {kind: items for kind, items in groups}
    limit = int(rc["profile_max_chars"])
    new_profiles = {"user": " ".join(str(data.get("user_profile") or "").split())[:limit],
                    "self": " ".join(str(data.get("self_profile") or "").split())[:limit]}
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
    for who, text in new_profiles.items():
        if len(text) >= 20 and text != engine.profile(who):
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
                 spawn: Optional[Callable[..., threading.Thread]] = None, poll_seconds: float = 30.0):
        self.engine, self.load_cfg, self.key_fn = engine, load_cfg, key_fn
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
            report = reflect_once(self.engine, cfg, key_fn=self.key_fn)
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
