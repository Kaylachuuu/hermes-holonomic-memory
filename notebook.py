"""A notebook in each reference library: her own notes on the material, kept with it.

A library is a shelf: what is on it belongs to nobody and never changes.  The notebook is what she has learned by
using it: how a passage bears on the project in hand, what worked and what did not, what she is still unsure of,
and where to look for a kind of problem.  The reference material is never written to.

A note lives in the library's own store, in a realm of its own ("notebook"), so its plates are apart from the
plates of the reference material and can be read with them or without.  A note is bound on those plates to the
passages it is about: the passage brings the note up, and the note leads back to the passage, whether or not the
two resemble each other.  ("This example failed because our target uses a different calling convention" looks
nothing like the example.)

Every note says where it came from, and that is carried into recall:

    inference   her own reasoning, not checked (the default)
    tested      she tried it; the note must say what was run and what happened
    user        the user said so; the note must carry the user's words, and they must be words the user wrote

A note may also point at memories in her own store (the project conversation it came out of).  Those are kept as
numbers, not on the plates: the library and her memory are separate stores, and nothing from a library reaches a
conversation in which it is not open.

When the library's folder is read again, a passage may be replaced.  Each note remembers the file, the section
and the text of what it was about, and is bound again to the passage that now stands there; if the text changed
the note says so, and if the passage is gone the note is kept and says that.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REALM = "notebook"
NOTE = "library_note"
TYPES = {"connection": "how it bears on the project", "lesson": "lesson learned", "question": "open question",
         "guidance": "where to look"}
PROVENANCE = ("inference", "tested", "user")

NOTE_DEFAULTS: Dict[str, Any] = {
    "library_note_chars": 700,            # a note is short: longer is cut at a sentence
    "library_notes_per_passage": 2,       # notes given with a passage that is given
    "library_notes_k": 3,                 # notes given because they answer the message itself
    "library_notes_min_score": 0.35,
    "library_notes_context_chars": 1400,  # budget for notes in a message, beside the reference material
}


def _cfg(cfg: Dict[str, Any], key: str):
    return cfg.get(key, NOTE_DEFAULTS[key])


def _error(message: str):
    from .library import LibraryError
    return LibraryError(message)


def digest(text: str) -> str:
    return hashlib.sha1(" ".join(str(text).split()).encode("utf-8")).hexdigest()[:16]


def _source(meta: dict) -> str:
    return (meta.get("file") or "") + (f" > {meta['section']}" if meta.get("section") else "")


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9_]+", str(text).lower()))


# ----------------------------------------------------------------- what a note is about

def _piece(lib, piece_id: Any) -> Optional[dict]:
    from .library import PIECE
    try:
        got = lib.get(int(piece_id))
    except (TypeError, ValueError):
        return None
    return got if got and got["kind"] == PIECE else None


def anchor(piece: dict) -> dict:
    meta = piece.get("meta") or {}
    return {"id": int(piece["id"]), "file": meta.get("file", ""), "section": meta.get("section", ""), "n": int(meta.get("n") or 0),
            "digest": digest(piece["text"]), "state": ""}


def resolve(lib, files: Dict[str, dict], about: List[Any]) -> List[dict]:
    """The passages a note is about, from what she gave: a passage's number, or its source as search results
    print it ('KERNEL.ASM > Load_FAT').  Anything that cannot be found is an error that says what there is."""
    anchors: List[dict] = []
    for item in about or []:
        text = str(item).strip()
        if not text:
            continue
        if re.fullmatch(r"#?\d+", text):
            piece = _piece(lib, text.lstrip("#"))
            if piece is None:
                raise _error(f"There is no passage numbered {text} in this library. Use the number or the source a search result gives.")
            found = [piece]
        else:
            name, _, section = (part.strip() for part in text.partition(">"))
            rels = [r for r in files if r.lower() == name.lower()] or [r for r in files if Path(r).name.lower() == Path(name).name.lower()]
            if not rels:
                raise _error(f"No file called '{name}' is in this library. Give the source as a search result prints it: FILE > SECTION.")
            pieces = [p for r in rels for p in (_piece(lib, i) for i in (files[r].get("ids") or [])) if p]
            if section:
                same = [p for p in pieces if (p["meta"].get("section") or "").lower() == section.lower()]
                found = same or [p for p in pieces if section.lower() in (p["meta"].get("section") or "").lower()]
                if not found:
                    have = sorted({p["meta"].get("section") or "" for p in pieces} - {""})[:12]
                    raise _error(f"'{name}' has no section called '{section}'." + (f" It has: {'; '.join(have)}." if have else ""))
            else:
                found = pieces
            if len(found) > 6:
                raise _error(f"'{text}' is {len(found)} passages. Say which part: give the section (FILE > SECTION) or a passage's number.")
        for piece in found:
            if piece["id"] not in {a["id"] for a in anchors}:
                anchors.append(anchor(piece))
    if len(anchors) > 8:
        raise _error(f"A note can be about at most 8 passages; this names {len(anchors)}. Write a note for each group.")
    return anchors


# ----------------------------------------------------------------- whose word it is

def said_by_user(quote: str, messages: List[str]) -> bool:
    """Whether these are words the user wrote.  A model asked to say where a note came from will call its own
    guess 'confirmed'; a note may only be marked as the user's if it carries the user's words, and they are
    looked for in what the user actually wrote in this conversation."""
    wanted = _plain(quote)
    return len(wanted) >= 8 and any(wanted in _plain(m) for m in messages if m)


def _settle(provenance: str, evidence: str, quote: str, messages: Optional[List[str]], by_command: bool) -> Dict[str, Any]:
    provenance = (provenance or "inference").strip().lower()
    if provenance not in PROVENANCE:
        raise _error(f"provenance is one of: {', '.join(PROVENANCE)}")
    evidence, quote = " ".join(str(evidence or "").split())[:400], " ".join(str(quote or "").split())[:300]
    if provenance == "tested":
        if len(evidence) < 20:
            raise _error("NOT SAVED as tested. A tested note must say in 'evidence' what was run and what happened "
                         "(for example: assembled with NASM, booted in Bochs, the FAT loaded). If you have not tried it, "
                         "save it as provenance 'inference'.")
        return {"provenance": "tested", "evidence": evidence}
    if provenance == "user":
        if by_command:
            return {"provenance": "user", "by_command": True, **({"quote": quote} if quote else {})}
        if not said_by_user(quote, messages or []):
            raise _error("NOT SAVED as the user's. A note marked provenance 'user' must carry in 'quote' the user's own words "
                         "from this conversation, copied exactly, and those words were not found in what the user wrote here. "
                         "If this is your own conclusion, save it as provenance 'inference'.")
        return {"provenance": "user", "quote": quote}
    return {"provenance": "inference"}


# ----------------------------------------------------------------- reading notes

def note_of(lib, memory: dict) -> dict:
    meta = memory.get("meta") or {}
    about = [dict(a) for a in meta.get("about") or []]
    return {"id": int(memory["id"]), "text": memory["text"], "type": meta.get("type", "lesson"),
            "provenance": meta.get("provenance", "inference"), "evidence": meta.get("evidence", ""), "quote": meta.get("quote", ""),
            "by_command": bool(meta.get("by_command")), "about": about, "memories": [int(i) for i in meta.get("memories") or []],
            "when": float(memory.get("created_at") or 0), "version": int(meta.get("version") or 1),
            "revises": meta.get("revises"), "retired": bool(meta.get("superseded_at")) or float(memory.get("trust", 1)) <= 0,
            "retired_because": meta.get("superseded_reason", ""), "replaced_by": meta.get("superseded_by"),
            "conversation": meta.get("conversation", "")}


def get_note(lib, note_id: Any) -> Optional[dict]:
    try:
        got = lib.get(int(note_id))
    except (TypeError, ValueError):
        return None
    return note_of(lib, got) if got and got["kind"] == NOTE else None


def list_notes(lib, *, retired: bool = False) -> List[dict]:
    out = []
    for m in lib.recent(100000, realm=REALM, kind=NOTE):
        full = lib.get(m["id"])
        if full:
            n = note_of(lib, full)
            if retired or not n["retired"]:
                out.append(n)
    return sorted(out, key=lambda n: n["id"])


def has_notes(lib) -> bool:
    try:
        return bool(lib.recent(1, realm=REALM, kind=NOTE))
    except Exception:
        return False


def where_from(note: dict) -> str:
    """Where a note came from, in the words she is given it in.  A guess is never worded like a result."""
    if note["provenance"] == "tested":
        return "you tested this: " + (note["evidence"] or "no record of how")
    if note["provenance"] == "user":
        return f"the user told you: \"{note['quote']}\"" if note["quote"] and not note["by_command"] else "from the user"
    return "your own inference, not checked"


def label(note: dict) -> str:
    out = f"note {note['id']}, {TYPES.get(note['type'], note['type'])}, {where_from(note)}"
    states = {a.get("state") for a in note["about"]}
    if "changed" in states:
        out += "; the passage it was written about has since changed"
    if "gone" in states:
        out += "; a passage it was written about is no longer in the library"
    return out


def line(note: dict, main=None, *, sources: bool = False) -> str:
    out = f"- ({label(note)}) {' '.join(note['text'].split())}"
    if sources and note["about"]:
        out += "\n  It is about: " + "; ".join(_source(a) + (" [changed]" if a.get("state") == "changed" else " [gone]" if a.get("state") == "gone" else "")
                                               for a in note["about"])
    if note["memories"] and main is not None:
        said = []
        for mid in note["memories"][:3]:
            got = main.get(int(mid))
            if got and got.get("text"):
                said.append(f"#{mid} \"{' '.join(got['text'].split())[:140]}\"")
        if said:
            out += "\n  It came out of what you remember: " + "; ".join(said)
    return out


def for_passages(lib, passage_ids: List[int], cfg: Dict[str, Any]) -> Dict[int, List[dict]]:
    """The notes bound to each of these passages, read off the notebook's plates with the passage as the cue.  What
    the plates bring is checked against the note's own record of what it is about, so a note is never shown under
    a passage it was not written on."""
    out: Dict[int, List[dict]] = {}
    per = int(_cfg(cfg, "library_notes_per_passage"))
    if per < 1:
        return out
    for pid in passage_ids:
        kept: List[dict] = []
        for hit in lib.associates(int(pid), k=per * 4 + 4, realms=(REALM,), min_score=0.1):
            if hit.kind != NOTE or hit.trust <= 0:
                continue
            note = get_note(lib, hit.id)
            if note and not note["retired"] and any(a.get("id") == int(pid) for a in note["about"]):
                kept.append(note)
        if kept:
            out[int(pid)] = sorted(kept, key=_rank)[:per]
    return out


def _rank(note: dict) -> tuple:
    # what the user said, then what was tested, then her own inference; newest first within each
    return ({"user": 0, "tested": 1}.get(note["provenance"], 2), -note["when"])


def matching(lib, query: str, cfg: Dict[str, Any], *, skip: Tuple[int, ...] = (), k: Optional[int] = None,
             floor: Optional[float] = None) -> List[dict]:
    """Notes that answer the message itself, whatever passages came up: 'for this kind of problem, look there'."""
    query = " ".join(str(query or "").split())
    k = int(_cfg(cfg, "library_notes_k")) if k is None else int(k)
    if not query or k < 1:
        return []
    floor = float(_cfg(cfg, "library_notes_min_score")) if floor is None else float(floor)
    out = []
    for hit in lib.recall(query[:2000], k=k * 3, realms=(REALM,), only_kinds=(NOTE,), min_score=floor, min_trust=0.01,
                          lexical=float(cfg.get("library_lexical_weight", 0.35))):
        if hit.id in skip:
            continue
        note = get_note(lib, hit.id)
        if note and not note["retired"]:
            out.append(dict(note, score=round(float(hit.score), 3)))
    return out[:k]


# ----------------------------------------------------------------- writing notes

def _clean(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:stop + 1] if stop > limit * 0.5 else cut.rstrip() + " [...]"


def _twin(lib, text: str, kind: str) -> Optional[dict]:
    """A note that already says this.  Told from the words alone, so nothing is asked of any model."""
    words = _words(text)
    for note in list_notes(lib):
        other = _words(note["text"])
        if note["type"] == kind and len(words & other) >= 0.8 * max(1, len(words | other)):
            return note
    return None


def _write(lib, text: str, meta: dict, anchors: List[dict]) -> int:
    ids = lib.remember(text, kind=NOTE, realm=REALM, session="notebook", chain=False, whole=True, trust=1.0,
                       links=[a["id"] for a in anchors if a.get("id")], meta=meta)
    if not ids:
        raise _error("The note could not be stored.")
    return int(ids[0])


def add(lib, files: Dict[str, dict], cfg: Dict[str, Any], *, text: str, kind: str = "lesson", about: Optional[List[Any]] = None,
        memories: Optional[List[int]] = None, provenance: str = "inference", evidence: str = "", quote: str = "",
        messages: Optional[List[str]] = None, by_command: bool = False, conversation: str = "", main=None) -> dict:
    kind = (kind or "lesson").strip().lower()
    if kind not in TYPES:
        raise _error(f"type is one of: {', '.join(TYPES)}")
    text = _clean(text, int(_cfg(cfg, "library_note_chars")))
    if len(text) < 15:
        raise _error("A note needs 'text': a sentence or two of what you learned.")
    whose = _settle(provenance, evidence, quote, messages, by_command)
    anchors = resolve(lib, files, about or [])
    twin = _twin(lib, text, kind)
    if twin is not None:
        raise _error(f"NOT SAVED: note {twin['id']} already says this (\"{twin['text'][:160]}\"). To change or add to it, use "
                     f"action 'revise_note' with note_id {twin['id']}; to say it is now tested or was confirmed, use 'confirm_note'.")
    kept = [int(i) for i in (memories or []) if main is None or main.get(int(i))][:6]
    meta = {"type": kind, **whose, "about": anchors, "memories": kept, "version": 1, "conversation": conversation}
    return get_note(lib, _write(lib, text, meta, anchors))


def revise(lib, files: Dict[str, dict], cfg: Dict[str, Any], note_id: Any, *, text: Optional[str] = None, kind: Optional[str] = None,
           about: Optional[List[Any]] = None, memories: Optional[List[int]] = None, provenance: Optional[str] = None,
           evidence: str = "", quote: str = "", messages: Optional[List[str]] = None, by_command: bool = False,
           conversation: str = "", main=None) -> dict:
    """A better version of a note takes its place.  The earlier one is retired, not deleted: it leaves recall and
    stays on record.  A note whose words change is her inference again unless she says otherwise: what was tested
    or confirmed was the earlier wording."""
    old = get_note(lib, note_id)
    if old is None or old["retired"]:
        raise _error(f"There is no note {note_id} to revise" + (" (it has been retired)." if old else "."))
    new_text = _clean(text, int(_cfg(cfg, "library_note_chars"))) if text else old["text"]
    if len(new_text) < 15:
        raise _error("A note needs 'text': a sentence or two of what you learned.")
    kind = (kind or old["type"]).strip().lower()
    if kind not in TYPES:
        raise _error(f"type is one of: {', '.join(TYPES)}")
    same_words = _plain(new_text) == _plain(old["text"])
    if provenance:
        whose = _settle(provenance, evidence, quote, messages, by_command)
    elif same_words:
        whose = {k: old[k] for k in ("provenance", "evidence", "quote") if old.get(k)}
        if old["by_command"]:
            whose["by_command"] = True
    else:
        whose = {"provenance": "inference"}
    anchors = resolve(lib, files, about) if about else [a for a in old["about"]]
    kept = [int(i) for i in memories if main is None or main.get(int(i))][:6] if memories else old["memories"]
    meta = {"type": kind, **whose, "about": anchors, "memories": kept, "version": old["version"] + 1, "revises": old["id"],
            "conversation": conversation}
    new_id = _write(lib, new_text, meta, anchors)
    lib.supersede(old["id"], new_id, "revised")
    return get_note(lib, new_id)


def confirm(lib, note_id: Any, *, provenance: str, evidence: str = "", quote: str = "", messages: Optional[List[str]] = None,
            by_command: bool = False) -> dict:
    """Say where a note now stands, its words unchanged: it was tested, or the user said so."""
    old = get_note(lib, note_id)
    if old is None or old["retired"]:
        raise _error(f"There is no note {note_id}" + (" (it has been retired)." if old else "."))
    whose = _settle(provenance, evidence, quote, messages, by_command)
    if whose["provenance"] == "inference":
        raise _error("confirm_note marks a note as 'tested' (with 'evidence') or as the user's (with 'quote').")
    history = list((lib.get(old["id"])["meta"] or {}).get("history") or [])
    history.append({"at": time.time(), "was": old["provenance"], "now": whose["provenance"]})
    # Confirmed as it stands now, so against the passage as it stands now: "has since changed" no longer applies.
    about = [dict(a, state="") if a.get("state") == "changed" else a for a in old["about"]]
    lib.update_meta(old["id"], {"evidence": None, "quote": None, "by_command": None, **whose, "history": history[-10:], "about": about})
    return get_note(lib, old["id"])


def retire(lib, note_id: Any, reason: str = "") -> dict:
    old = get_note(lib, note_id)
    if old is None:
        raise _error(f"There is no note {note_id}.")
    if not old["retired"]:
        lib.supersede(old["id"], None, " ".join(str(reason or "retired").split())[:300])
    return get_note(lib, old["id"])


def restore(lib, note_id: Any) -> dict:
    old = get_note(lib, note_id)
    if old is None:
        raise _error(f"There is no note {note_id}.")
    if old["retired"]:
        lib.unsupersede(old["id"], 1.0)
        lib.update_meta(old["id"], {"superseded_by": None, "superseded_at": None, "superseded_reason": None})
    return get_note(lib, old["id"])


# ----------------------------------------------------------------- when the library is read again

def rebind(lib, files: Dict[str, dict]) -> dict:
    """After the folder has been read again: bind each note to the passages that now stand where its passages
    stood.  Unchanged text is found again and nothing is said.  Changed text is bound and the note says the
    passage changed.  A passage that is gone leaves the note standing, saying so."""
    report = {"notes": 0, "rebound": 0, "changed": 0, "gone": 0}
    for note in list_notes(lib):
        report["notes"] += 1
        touched = False
        for a in note["about"]:
            if a.get("id") and _piece(lib, a["id"]) is not None:
                continue                                           # still there, as it was
            now = [p for p in (_piece(lib, i) for i in ((files.get(a.get("file") or "") or {}).get("ids") or []))
                   if p and (p["meta"].get("section") or "") == (a.get("section") or "")]
            touched = True
            if not now:
                if a.get("state") != "gone":
                    report["gone"] += 1
                a.update(id=None, state="gone")
                continue
            same = [p for p in now if digest(p["text"]) == a.get("digest")]
            pick = (same or [p for p in now if int(p["meta"].get("n") or 0) == int(a.get("n") or 0)] or now)[0]
            a.update(id=int(pick["id"]), n=int(pick["meta"].get("n") or 0), state="" if same else "changed")
            if not same:
                a.update(was=a.get("digest"), digest=digest(pick["text"]))
                report["changed"] += 1
            else:
                report["rebound"] += 1
            lib.link(note["id"], int(pick["id"]), realm=REALM)
        if touched:
            lib.update_meta(note["id"], {"about": note["about"]})
    return report


def check(lib) -> dict:
    """Whether the plates do what the notes' own records say: from each passage, is the note read back; from each
    note, the passage?  Read-only."""
    out = {"notes": 0, "bindings": 0, "passage_to_note": 0, "note_to_passage": 0, "unbound": 0, "stale": 0}
    for note in list_notes(lib):
        out["notes"] += 1
        live = [a for a in note["about"] if a.get("id")]
        out["stale"] += sum(1 for a in note["about"] if a.get("state"))
        if not live:
            out["unbound"] += 1
            continue
        back = {h.id for h in lib.associates(note["id"], k=16, realms=(REALM, "waking"), min_score=0.1)}
        for a in live:
            out["bindings"] += 1
            out["note_to_passage"] += int(a["id"] in back)
            out["passage_to_note"] += int(note["id"] in {h.id for h in lib.associates(a["id"], k=16, realms=(REALM,), min_score=0.1)})
    return out
