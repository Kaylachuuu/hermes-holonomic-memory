"""Living alongside a persona service (thymos): nothing writes as her except her.

When a persona service is running in the same Hermes, holonomic stops writing in her voice: no self notes, no
relationship notes, no "who you have become" or relationship profile, no account of a conversation and no waking
thoughts on a dream.  Facts about the user, the checks on them and the profiles of the user and their projects stay
as they are: they are bookkeeping about the user, not words of hers (persona-provider.md 9.7 to 9.9).

What she writes instead reaches holonomic through a folder.  The service leaves each account of a conversation she
chose to store in `plugin-data/thymos/accounts/` as a small JSON file, and holonomic stores it as an episode, dated
as the conversation and linked to it on the plates, then moves the file to `stored/`.  If she stores none, there is
none, and nothing fills in for her.

The same folder carries the rest of what passes between them (persona-provider.md 4.2 and 4.6).  After a sleep that
made a dream, holonomic says so in `slept/`, with the dream's text; whatever she writes about it comes back through
`dream-thoughts/` and is kept with the dream as hers.  Once, the notes another model wrote in her voice before the
service ran are offered to her in `old-notes.json`, as dated text; holonomic leaves them as they are.

Without a persona service nothing here changes anything.

Only stdlib imports here: Hermes executes this file when it loads the plugin.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Set by the persona service when Hermes loads it, in the same process, e.g. "thymos/0.3.0 accounts=1 idle=1".
# An environment variable because the two plugins cannot import each other, and because a file would outlive the
# service: a thymos that is uninstalled leaves no variable behind, and holonomic goes back to what it did before.
ENV = "HERMES_PERSONA_SERVICE"
ACCOUNTS = "accounts"           # her accounts of conversations; holonomic stores them
IDLE = "idle.json"              # what she has waiting for idle time; holonomic's own work waits for it
SLEPT = "slept"                 # holonomic says here that it slept and what it made; she may write about it
DREAM_THOUGHTS = "dream-thoughts"   # what she wrote about a dream; holonomic keeps it with the dream as hers
OLD_NOTES = "old-notes.json"    # the notes another model wrote in her voice, offered to her once
IDLE_FRESH_SECONDS = 600        # a file older than this was left by a Hermes that is gone, and is not waited for

PERSONA_DEFAULTS: Dict[str, Any] = {
    # auto: follow the persona service when one is running.  on / off: as if one were, or were not.
    "persona_service": "auto",
    # Unattended sleep stops between steps when someone starts talking, and picks up where it was next time.
    # auto: when a persona service is running.  on / off: always, never.
    "sleep_gives_way": "auto",
}


def service(cfg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The persona service running alongside, as {"name", "version", "features"}, or None when there is none."""
    setting = str((cfg or {}).get("persona_service", "auto")).strip().lower()
    if setting in ("off", "false", "0", "no"):
        return None
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return {"name": "configured", "version": "", "features": {}} if setting in ("on", "true", "1", "yes") else None
    head, *rest = raw.split()
    name, _, version = head.partition("/")
    features = {}
    for item in rest:
        key, _, value = item.partition("=")
        features[key] = value or "1"
    return {"name": name or "persona", "version": version, "features": features}


def on(cfg: Optional[Dict[str, Any]] = None) -> bool:
    """Whether holonomic should stop writing in her voice."""
    return service(cfg) is not None


def gives_way(cfg: Optional[Dict[str, Any]] = None) -> bool:
    setting = str((cfg or {}).get("sleep_gives_way", "auto")).strip().lower()
    if setting in ("on", "true", "1", "yes"):
        return True
    if setting in ("off", "false", "0", "no"):
        return False
    return on(cfg)


def folder(home) -> Path:
    return Path(str(home)) / "plugin-data" / "thymos"


def her_turn(home, cfg: Optional[Dict[str, Any]] = None, now: Optional[float] = None) -> str:
    """Why memory's own idle work should wait for her, in words; empty when it need not.

    Her reflections and her accounts of quiet conversations come first at idle (persona-provider.md 9.10).  The
    service says what it has waiting in `idle.json`, rewritten every half minute while it runs."""
    s = service(cfg)
    if s is None or s["features"].get("idle") in (None, "0"):
        return ""
    now = time.time() if now is None else now
    try:
        st = json.loads((folder(home) / IDLE).read_text(encoding="utf-8"))
        at = float(st.get("at") or 0)
    except (OSError, ValueError, TypeError):
        return ""
    if now - at > IDLE_FRESH_SECONDS:
        return ""
    if st.get("running"):
        return f"her own reflection comes first, and one is running now ({st.get('running')})"
    due = int(st.get("due") or 0)
    if due:
        return f"her own reflections come first: {due} waiting for idle time"
    return ""


def waiting_accounts(home) -> List[Path]:
    try:
        return sorted(p for p in (folder(home) / ACCOUNTS).glob("*.json") if p.is_file())
    except OSError:
        return []


def take_accounts(engine, home, *, key_fn: Optional[Callable[[str], List[str]]] = None,
                  raw_kinds: tuple = (), limit: int = 20) -> List[Dict[str, Any]]:
    """Store the accounts she has written as episodes.  Returns what was stored.

    Each is linked to the opening and close of the stretch of conversation it covers, and marks that stretch as
    summarised, which is what lets it fade (sleep step 3) as a summary written by holonomic did.  Conversation she
    did not account for keeps no mark, and so does not fade: it stays as it was said."""
    from .sleep import EPISODE, RAW_KINDS
    kinds = raw_kinds or RAW_KINDS
    out: List[Dict[str, Any]] = []
    for path in waiting_accounts(home)[:limit]:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            text = " ".join(str(item.get("account") or "").split())
            session = str(item.get("session_id") or "")
        except (OSError, ValueError) as exc:
            logger.warning("holonomic: an account from the persona service could not be read (%s): %s", path.name, exc)
            continue
        if not text:
            _file_away(path, item if isinstance(item, dict) else {}, None, "empty")
            continue
        done = int(engine.kv_get("episode:" + session, "0") or 0) if session else 0
        memories = engine.session_memories(session, after_id=done, kinds=kinds, limit=2000) if session else []
        links = list(dict.fromkeys([memories[0]["id"], memories[-1]["id"]])) if memories else []
        when = float(item.get("conversation_ended_at") or (memories[-1]["created_at"] if memories else 0) or time.time())
        meta = {"session": session, "sources": [m["id"] for m in memories][:40], "author": "self",
                "persona": {k: item.get(k) for k in ("entry_hash", "model", "written_at", "service") if item.get(k)}}
        ids = engine.remember(text[:4000], kind=EPISODE, session="episodes", chain=True, links=links, whole=True,
                              keys=key_fn(text) if key_fn else [], trust=0.8, salience=1.2, created_at=when, meta=meta)
        if memories:
            engine.kv_set("episode:" + session, str(memories[-1]["id"]))
        stored = {"session": session, "id": ids[0] if ids else None, "covers": len(memories), "account": text}
        out.append(stored)
        _file_away(path, item, stored["id"], "stored")
    return out


def _file_away(path: Path, item: Dict[str, Any], memory_id: Optional[int], how: str) -> None:
    """Moved, not deleted: the service, and anyone checking, can see what became of each one."""
    dest = path.parent / "stored"
    try:
        dest.mkdir(exist_ok=True)
        record = dict(item, holonomic={"how": how, "memory_id": memory_id, "at": time.time()})
        tmp = dest / (path.name + ".part")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, dest / path.name)
        path.unlink()
    except OSError as exc:
        logger.warning("holonomic: account %s was %s but could not be filed away: %s", path.name, how, exc)


def written_by_her(memory: Dict[str, Any]) -> bool:
    return ((memory.get("meta") or {}).get("author") == "self")


def wants(cfg: Optional[Dict[str, Any]], name: str) -> bool:
    """Whether the service running alongside says it takes part in `name` (e.g. "slept=1")."""
    s = service(cfg)
    return s is not None and s["features"].get(name) not in (None, "0")


def _write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def tell_slept(home, cfg: Optional[Dict[str, Any]], report: Dict[str, Any], now: Optional[float] = None) -> Optional[Path]:
    """Tell the persona service that memory slept, and what it made (persona-provider.md 4.2, `MEMORY_SLEPT`).

    Facts only: when, the dreams with their text (she has no way to look them up in the moment she is offered), how
    many of her accounts were stored and how many facts were learned.  Nothing here says what any of it means.
    Only a sleep that made a dream is told: without one there is nothing for her to write about."""
    if not wants(cfg, "slept") or report.get("dry_run"):
        return None
    dreams = [d for d in (report.get("dreams") or []) if d.get("id")]
    if not dreams:
        return None
    from .backup import plugin_version     # the command line's package has no __version__ (backup.py)
    now = time.time() if now is None else now
    item = {"slept_at": now,
            "dreams": [{"id": d["id"], "text": d["text"], "pictures": len(d.get("pictures") or [])} for d in dreams],
            "her_accounts_stored": len(report.get("episodes") or []),
            "facts_learned": sum(len(r.get("stored") or []) for r in report.get("reflections") or []),
            "memory": f"holonomic/{plugin_version()}"}
    path = folder(home) / SLEPT / f"{int(now * 1000)}.json"
    try:
        _write(path, item)
        return path
    except OSError as exc:
        logger.warning("holonomic: could not tell the persona service that memory slept: %s", exc)
        return None


def waiting_dream_thoughts(home) -> List[Path]:
    try:
        return sorted(p for p in (folder(home) / DREAM_THOUGHTS).glob("*.json") if p.is_file())
    except OSError:
        return []


def take_dream_thoughts(engine, home, limit: int = 20) -> List[Dict[str, Any]]:
    """Keep what she wrote about a dream with the dream, as hers.  The dream itself stays the memory system's."""
    out: List[Dict[str, Any]] = []
    for path in waiting_dream_thoughts(home)[:limit]:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            text = " ".join(str(item.get("thoughts") or "").split())
            dream_id = int(item.get("dream_id") or 0)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("holonomic: her words on a dream could not be read (%s): %s", path.name, exc)
            continue
        found = engine.get(dream_id) if dream_id else None
        if not text or not found:
            _file_away(path, item, None, "empty" if not text else "no such dream")
            continue
        engine.update_meta(dream_id, {"her_thoughts": text[:4000], "her_thoughts_at": item.get("written_at") or time.time(),
                                      "her_thoughts_entry": item.get("entry_hash", "")})
        out.append({"dream": dream_id, "thoughts": text})
        _file_away(path, item, dream_id, "stored")
    return out


# What another model wrote in her voice before the persona service ran: holonomic's reflection model's notes about
# her and about the two of them, and the "who you have become" and relationship profiles.
_OLD_KINDS = ("self_note", "bond_note")
_OLD_OFFERED = "persona:old_notes_offered"
OLD_NOTES_MAX_CHARS = 12000


def old_notes(engine) -> Dict[str, Any]:
    notes = []
    for kind in _OLD_KINDS:
        for r in engine.recent(500, kind=kind):
            full = engine.get(r["id"]) or {}
            if not written_by_her(full):
                notes.append({"id": r["id"], "kind": kind, "at": r["created_at"], "text": " ".join(r["text"].split())})
    notes.sort(key=lambda n: n["at"])
    profiles = {k: engine.profile(k) for k in ("self", "us") if engine.profile(k)}
    return {"notes": notes, "profiles": profiles}


def offer_old_notes(engine, home, cfg: Optional[Dict[str, Any]], now: Optional[float] = None) -> Optional[Path]:
    """Once: the notes another model wrote in her voice, as dated text, for her to keep in her own words or not
    (persona-provider.md 4.6, `OLD_SELF_NOTES`).  Holonomic changes nothing in them and keeps them, labelled as
    that model's.  With nothing to offer, it is marked as offered and nothing is written."""
    if not wants(cfg, "old_notes") or engine.kv_get(_OLD_OFFERED):
        return None
    now = time.time() if now is None else now
    found = old_notes(engine)
    if not found["notes"] and not found["profiles"]:
        engine.kv_set(_OLD_OFFERED, f"{now} nothing to offer")
        return None
    shown, budget = [], OLD_NOTES_MAX_CHARS - sum(len(t) for t in found["profiles"].values())
    for n in reversed(found["notes"]):                  # newest kept when there are too many
        if len(n["text"]) + 40 > budget:
            break
        shown.append(n)
        budget -= len(n["text"]) + 40
    shown.reverse()
    notes = found["notes"]
    from .backup import plugin_version     # the command line's package has no __version__ (backup.py)
    item = {"written_by": str((cfg or {}).get("reflect_model") or "the memory system's reflection model"),
            "count": len(notes), "shown": len(shown),
            "first_at": notes[0]["at"] if notes else None, "last_at": notes[-1]["at"] if notes else None,
            "notes": [{"at": n["at"], "kind": n["kind"], "text": n["text"]} for n in shown],
            "profiles": found["profiles"], "memory": f"holonomic/{plugin_version()}", "offered_at": now}
    path = folder(home) / OLD_NOTES
    try:
        _write(path, item)
    except OSError as exc:
        logger.warning("holonomic: could not offer her the old notes: %s", exc)
        return None
    engine.kv_set(_OLD_OFFERED, str(now))
    return path
