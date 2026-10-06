"""Image memory: what the agent has been shown.

An image is kept three ways:
  the file       the original bytes, in <data>/images/, never changed.  This is the record.
  a description  what a vision model saw in it, stored as an ordinary memory (kind 'image'), so the
                 image is found by meaning, fades and is recalled like anything else she remembers
  its sections   the image cut into overlapping parts, each looked at on its own (kind 'image_part'),
                 so small things and things at the edges are noticed too

Everything named in the image is also filed as a label, which makes "every image with a cat in it"
an exact lookup and not a search.

Each image and each section has an empty place for a vector.  That is where fingerprints from a
vision embedding model will go, to recognise similar pictures without words.

Only the standard library is imported at module level: Hermes executes every top-level module in
this folder when it loads the plugin.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .reflect import IMAGE, IMAGE_KINDS, IMAGE_PART, NO_DOUBLE_QUOTES, ReflectionError, is_complete, trim_to_sentence

logger = logging.getLogger(__name__)

IMAGE_DEFAULTS: Dict[str, Any] = {
    "image_enabled": False,          # keep and describe images shown in conversation
    "image_model": "",               # vision model; empty = the reflection model
    "image_host": "",                # Ollama server for it; empty = the reflection server
    "image_view_width": 2048,        # the copy she looks at and shows fits inside this, keeping its shape
    "image_view_height": 1536,       # (the box is turned on its side for a portrait image)
    "image_sections": True,          # also look at the image part by part
    "image_grid": 3,                 # 3 = nine parts, each half the width and height, overlapping by half
    "image_section_min_side": 640,   # smaller images are not cut up: there is nothing more to see
    "image_section_max_side": 1024,  # a part is shrunk to this before the model sees it
    "image_from_original": True,     # cut the parts from the original, which may hold more detail than the copy
    "image_sections_when": "idle",   # 'idle': when the conversation has gone quiet; 'now': straight away
    "image_idle_seconds": 120,
    "image_timeout": 180,
    "image_max_tokens": 1000,
    "image_temperature": 0.2,
    "image_max_labels": 12,
    "image_max_bytes": 30_000_000,
}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS images (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256       TEXT NOT NULL UNIQUE,
    file         TEXT NOT NULL,              -- the original, relative to the data folder
    view         TEXT NOT NULL DEFAULT '',   -- the standard-size copy (may be the original itself)
    width        INTEGER NOT NULL DEFAULT 0,
    height       INTEGER NOT NULL DEFAULT 0,
    bytes        INTEGER NOT NULL DEFAULT 0,
    format       TEXT NOT NULL DEFAULT '',
    realm        TEXT NOT NULL DEFAULT 'waking',
    source       TEXT NOT NULL DEFAULT 'user',
    session      TEXT NOT NULL DEFAULT '',
    origin       TEXT NOT NULL DEFAULT '',   -- the file name it arrived with
    caption      TEXT NOT NULL DEFAULT '',   -- what was said when it was shown
    created_at   REAL NOT NULL,
    seen         INTEGER NOT NULL DEFAULT 1,
    last_seen    REAL,
    memory_id    INTEGER,                    -- the memory holding the description of the whole image
    described_at REAL,
    sections_at  REAL,                       -- when the last section was looked at
    claimed_at   REAL,                       -- a process is describing it right now
    vec          BLOB,                       -- reserved: fingerprint of the whole image
    vec_model    TEXT,
    forgotten    INTEGER NOT NULL DEFAULT 0,
    meta         TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS image_sections (
    image_id     INTEGER NOT NULL,
    idx          INTEGER NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,     -- as fractions of the image
    place        TEXT NOT NULL DEFAULT '',
    notable      INTEGER,                    -- NULL not looked at yet; 0 nothing worth recording; 1 described
    memory_id    INTEGER,
    described_at REAL,
    claimed_at   REAL,
    vec          BLOB,                       -- reserved: fingerprint of this part
    vec_model    TEXT,
    PRIMARY KEY (image_id, idx)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS image_labels (
    image_id INTEGER NOT NULL,
    label    TEXT NOT NULL,
    section  INTEGER NOT NULL DEFAULT -1,    -- -1: named in the description of the whole image
    PRIMARY KEY (image_id, label, section)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_image_labels ON image_labels(label);
"""

_CLAIM_SECONDS = 900.0          # a description that was started this long ago and never finished is tried again

_SYSTEM = ("You are the visual memory of an AI companion. You look at an image and record what is visible in it, accurately "
           "and plainly, so that it can be found and recalled later. Record only what can be seen. Do not guess who a "
           "person is, where a place is, or what happened before or after, unless it is written in the image or stated "
           "in a note supplied with it. " + NO_DOUBLE_QUOTES)

_WHOLE = """\
Look at this image.{note}

Write:
- description: three to six sentences saying what the image shows: what kind of image it is (photo, screenshot, drawing, diagram, document), its main subject, the setting, and notable details such as colours, positions, expressions and anything unusual.
- labels: up to {n} short lowercase names for the distinct things visible, each a singular noun or short noun phrase (for example: cat, sofa, window, laptop, mountain). Use an adjective only when it is needed to tell two things apart.
- text: writing in the image that you can read clearly, copied exactly, a few lines at most. Leave out anything you cannot make out for certain: do not guess at what a sign probably says. An empty string if there is none.
- people: true if one or more real people can be seen in the image, otherwise false."""

_NOTE = ("\nThe person who showed it said: {caption}\nIf those words name someone or something that is visible, use that name. "
         "Take nothing else from them.")
_CORRECTED = ("\nAn earlier description of this image got something wrong, and the person who showed it corrected it: {correction}\n"
              "Treat what they say as true and describe the image accordingly.")

_PART = """\
This is one part of a larger image: the {place}.
The whole image was described as: {whole}
That description is only there to tell you where you are; it can be mistaken about what things are. Go by what you can see in this part.{said}

Write:
- notable: false if this part shows nothing worth recording on its own (plain background, sky, wall, floor, blur, or only the edge of something); otherwise true.
- description: one to three sentences on what is visible in this part, giving detail that the description of the whole lacks. An empty string if not notable. Quote writing only if you can read it clearly; do not guess at what a sign says.
- labels: up to 6 short lowercase names for the distinct things visible in this part, each a singular noun or short noun phrase. An empty list if not notable."""

_LOOK = """\
Look at this image and answer the question from what is visible. If the image does not show the answer, say so.

Question: {question}"""

_LABELS = {"type": "array", "items": {"type": "string"}}
_WHOLE_SCHEMA = {"type": "object", "properties": {"description": {"type": "string"}, "labels": _LABELS, "text": {"type": "string"},
                                                  "people": {"type": "boolean"}},
                 "required": ["description", "labels", "text", "people"]}
# For images described before the model was asked whether people are visible.
# Words for a person only: 'face', 'head' and 'hand' were here once, and a photo of a cat was taken to have people in it
# because its parts were labelled 'face' and 'head'.
_PEOPLE_WORDS = frozenset("person people woman women man men girl girls boy boys child children baby toddler kid kids selfie "
                          "daughter son mother father family crowd".split())
_PART_SCHEMA = {"type": "object", "properties": {"notable": {"type": "boolean"}, "description": {"type": "string"}, "labels": _LABELS},
                "required": ["notable", "description", "labels"]}
_LOOK_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

# Hermes marks a message that carries an image in one of two ways.  For a model that can see, the picture
# goes in the message and the text gets `[Image attached at: <path>]`.  For one it believes cannot, it has
# another model describe the picture and puts that description, and the path, in the text instead.
# The desktop app writes a third form into the message itself: `@image:<path>`, the path quoted if it has spaces.
_REF = r"@image:(?:`([^`\n]+)`|\"([^\"\n]+)\"|'([^'\n]+)'|(\S+))"
_HINT_RE = re.compile(r"\[Image attached at: ([^\]\r\n]+)\]|\bimage_url: ([^\]\r\n]+)\]|" + _REF)
_MARKER_RE = re.compile(r"\[Image attached(?: at)?: [^\]\r\n]+\]|\[\d+ images?\]|\[screenshot\]|" + _REF
                        + r"|\[The user attached an image.*?\bimage_url: [^\]\r\n]+\]", re.DOTALL)
# The desktop app only treats a few formats as pictures.  Anything else, a phone's HEIC photo included, is attached
# as a file: the message gets an `@file:` token and a footer saying where the file is on disk, and the model is
# not shown the picture at all.
IMAGE_SUFFIXES = (".heic", ".heif", ".avif", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")
_ON_DISK_RE = re.compile(r"It is available on disk at `([^`\r\n]+)`")
_FILE_REF_RE = re.compile(r"@file:(?:`([^`\n]+)`|\"([^\"\n]+)\"|'([^'\n]+)'|(\S+))")
_FOOTER_RE = re.compile(r"\n*--- (?:Context Warnings|Attached Context) ---\s*\n.*", re.DOTALL)
_DEFAULT_CAPTION = "What do you see in this image?"          # Hermes supplies this when an image is sent without words
_EXT = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tif",
        "HEIF": ".heic", "AVIF": ".avif"}
_SHOWN_AS_IS = ("JPEG", "PNG", "WEBP")       # formats a chat window can display without conversion


class ImageError(RuntimeError):
    pass


def image_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(IMAGE_DEFAULTS)
    out.update({k: cfg[k] for k in IMAGE_DEFAULTS if k in cfg})
    out["image_model"] = out["image_model"] or cfg.get("reflect_model") or ""
    out["image_host"] = (out["image_host"] or cfg.get("reflect_host") or cfg.get("ollama_host") or "http://localhost:11434").rstrip("/")
    out["image_grid"] = max(1, min(int(out["image_grid"] or 3), 6))
    return out


def _db(engine):
    if not getattr(engine, "_images_ready", False):
        with engine._lock:
            engine._db.executescript(_SCHEMA_SQL)
        engine._images_ready = True
    return engine._db


# ------------------------------------------------------------------ pictures

def _pil():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImageError("Pillow is not installed in Hermes' Python environment, so images cannot be read.") from exc
    try:                                     # phone photos; Hermes ships this alongside Pillow
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    return Image, ImageOps


def open_image(data: bytes):
    """Decode bytes to an upright RGB picture.  Returns (picture, format name)."""
    Image, ImageOps = _pil()
    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").upper()
        img = ImageOps.exif_transpose(img)           # a phone stores a sideways photo plus a note to turn it
        img.load()
    except ImageError:
        raise
    except Exception as exc:
        raise ImageError(f"That is not an image that can be read ({exc}).") from exc
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[-1])
        img = flat
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img, fmt


def _jpeg(img, box: Tuple[int, int]) -> bytes:
    Image, _ = _pil()
    copy = img.copy()
    copy.thumbnail(box, Image.LANCZOS)               # only ever shrinks
    out = io.BytesIO()
    copy.save(out, "JPEG", quality=88)
    return out.getvalue()


def view_box(width: int, height: int, ic: Dict[str, Any]) -> Tuple[int, int]:
    long, short = int(ic["image_view_width"]), int(ic["image_view_height"])
    return (long, short) if width >= height else (short, long)


def section_grid(grid: int) -> List[Tuple[int, float, float, float, float, str]]:
    """Overlapping parts: each is 2/(grid+1) of the width and height, and each step moves by half a part,
    so whatever one part cuts in two lies near the middle of its neighbour.  grid=3 on a 1024x768 image
    gives nine parts of 512x384."""
    if grid < 2:
        return []
    size, step = 2.0 / (grid + 1), 1.0 / (grid + 1)
    rows = _names(grid, ("top", "middle", "bottom"))
    cols = _names(grid, ("left", "centre", "right"))
    out = []
    for r in range(grid):
        for c in range(grid):
            place = "centre" if (rows[r], cols[c]) == ("middle", "centre") else f"{rows[r]} {cols[c]}"
            out.append((r * grid + c, c * step, r * step, size, size, place))
    return out


def _names(grid: int, three: Tuple[str, str, str]) -> List[str]:
    if grid == 2:
        return [three[0], three[2]]
    if grid == 3:
        return list(three)
    return [three[0] if i == 0 else three[2] if i == grid - 1 else f"{three[1]}-{i}" for i in range(grid)]


def _crop(img, x: float, y: float, w: float, h: float, max_side: int) -> bytes:
    W, H = img.size
    box = (int(round(x * W)), int(round(y * H)), int(round((x + w) * W)), int(round((y + h) * H)))
    return _jpeg(img.crop(box), (max_side, max_side))


# ----------------------------------------------------- images in a conversation

def strip_image_markers(text: str) -> str:
    """Remove what Hermes adds to a message that carries an image, leaving the user's own words."""
    text = _FOOTER_RE.sub("", text or "")               # Hermes' note about attached files is not something the user said
    text = _FILE_REF_RE.sub(lambda m: "" if next(g for g in m.groups() if g).lower().endswith(IMAGE_SUFFIXES) else m.group(0), text)
    text = _MARKER_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text.replace(_DEFAULT_CAPTION, "")).strip()


def attached_paths(text: str) -> List[str]:
    found = [next(g for g in groups if g).strip() for groups in _HINT_RE.findall(text or "")]
    return list(dict.fromkeys(found + attached_as_files(text)))


def attached_as_files(text: str) -> List[str]:
    """Pictures that were attached as plain files, which the model was told about but not shown."""
    return list(dict.fromkeys(p.strip() for p in _ON_DISK_RE.findall(text or "") if p.strip().lower().endswith(IMAGE_SUFFIXES)))


def _data_url_bytes(url: str) -> Optional[bytes]:
    if not isinstance(url, str) or not url.startswith("data:image/") or "," not in url:
        return None
    try:
        return base64.b64decode(url.split(",", 1)[1], validate=False)
    except Exception:
        return None


def images_in_turn(user_text: str, messages: Optional[List[Dict[str, Any]]] = None,
                   max_bytes: int = 30_000_000) -> List[Tuple[bytes, str]]:
    """The images the user attached to the message just answered, as (bytes, where it came from).
    Hermes leaves a `[Image attached at: <path>]` note in the text and, for a model that can see, puts the
    picture itself in the message.  The file is preferred: it is the original, not a re-encoded copy."""
    found: List[Tuple[bytes, str]] = []
    paths = attached_paths(user_text)
    for raw in paths:
        try:
            p = Path(raw)
            if p.is_file() and 0 < p.stat().st_size <= max_bytes:
                found.append((p.read_bytes(), raw))
        except OSError:
            continue
    if found or not messages:
        return found
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            n = 0
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in ("image_url", "input_image"):
                    continue
                url = part.get("image_url")
                data = _data_url_bytes(url.get("url") if isinstance(url, dict) else url)
                if data and len(data) <= max_bytes:
                    found.append((data, paths[n] if n < len(paths) else "attached image"))
                    n += 1
        break                                       # only the message that was just answered
    return found


# ------------------------------------------------------------------- storing

def _row(engine, image_id: int) -> Optional[sqlite3.Row]:
    with engine._lock:
        return _db(engine).execute("SELECT * FROM images WHERE id = ?", (int(image_id),)).fetchone()


def known(engine, data: bytes) -> Optional[dict]:
    """The stored image with exactly these bytes, if she has been shown it before."""
    with engine._lock:
        row = _db(engine).execute("SELECT id FROM images WHERE sha256 = ? AND forgotten = 0",
                                  (hashlib.sha256(data).hexdigest(),)).fetchone()
    return get_image(engine, row["id"]) if row else None


def add_image(engine, data: bytes, cfg: Dict[str, Any], *, origin: str = "", caption: str = "", session: str = "",
              source: str = "user", realm: str = "waking", links: Tuple[int, ...] | List[int] = (), count: bool = True) -> dict:
    """Keep an image.  Showing the same file again does not store it twice: it counts as seen again.
    Returns the image with 'new' saying which happened.  Nothing is described here."""
    ic = image_config(cfg)
    if not data:
        raise ImageError("The image is empty.")
    if len(data) > int(ic["image_max_bytes"]):
        raise ImageError(f"The image is larger than the {int(ic['image_max_bytes']) // 1_000_000} MB limit.")
    sha = hashlib.sha256(data).hexdigest()
    db, now = _db(engine), time.time()
    with engine._lock:
        row = db.execute("SELECT id, forgotten FROM images WHERE sha256 = ?", (sha,)).fetchone()
        if row and not row["forgotten"]:
            if count:
                db.execute("UPDATE images SET seen = seen + 1, last_seen = ? WHERE id = ?", (now, row["id"]))
            return dict(get_image(engine, row["id"]), new=False)
    img, fmt = open_image(data)
    width, height = img.size
    folder = engine.path / "images"
    folder.mkdir(parents=True, exist_ok=True)
    name = sha[:24]
    file = f"images/{name}{_EXT.get(fmt, '.img')}"
    (engine.path / file).write_bytes(data)
    box = view_box(width, height, ic)
    if fmt in _SHOWN_AS_IS and width <= box[0] and height <= box[1]:
        view = file
    else:
        view = f"images/{name}.view.jpg"
        (engine.path / view).write_bytes(_jpeg(img, box))
    meta = json.dumps({"links": [int(i) for i in links if i]})
    caption = " ".join(strip_image_markers(caption).split())[:600]
    values = (file, view, width, height, len(data), fmt, realm, source, session, Path(origin).name if origin else "", caption, now, now, meta)
    with engine._lock:
        try:
            if row:                                  # forgotten once, shown again: it starts over
                db.execute("DELETE FROM image_labels WHERE image_id = ?", (row["id"],))
                db.execute("DELETE FROM image_sections WHERE image_id = ?", (row["id"],))
                db.execute("UPDATE images SET file=?, view=?, width=?, height=?, bytes=?, format=?, realm=?, source=?, session=?, "
                           "origin=?, caption=?, created_at=?, last_seen=?, meta=?, seen=1, memory_id=NULL, described_at=NULL, "
                           "sections_at=NULL, claimed_at=NULL, forgotten=0 WHERE id=?", values + (row["id"],))
                image_id = int(row["id"])
            else:
                image_id = int(db.execute(
                    "INSERT INTO images (file, view, width, height, bytes, format, realm, source, session, origin, caption, "
                    "created_at, last_seen, meta, sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values + (sha,)).lastrowid)
        except sqlite3.IntegrityError:               # another process stored it a moment ago
            other = db.execute("SELECT id FROM images WHERE sha256 = ?", (sha,)).fetchone()
            return dict(get_image(engine, other["id"]), new=False)
        if ic["image_sections"] and max(width, height) >= int(ic["image_section_min_side"]):
            db.executemany("INSERT OR IGNORE INTO image_sections (image_id, idx, x, y, w, h, place) VALUES (?,?,?,?,?,?,?)",
                           [(image_id, i, x, y, w, h, place) for i, x, y, w, h, place in section_grid(ic["image_grid"])])
    return dict(get_image(engine, image_id), new=True)


# ---------------------------------------------------------------- describing

def _seer(ic: Dict[str, Any], report: Dict[str, Any]) -> Callable[..., str]:
    from . import reflect as _reflect
    if not ic["image_model"]:
        raise ImageError("No vision model is set. Run: hermes holonomic images on --model NAME")

    def see(step: str, system: str, prompt: str, jpeg: bytes, schema: dict, max_tokens: int) -> str:
        out = _reflect.ollama_chat(ic["image_host"], ic["image_model"], system, prompt, timeout=float(ic["image_timeout"]),
                                   temperature=float(ic["image_temperature"]), max_tokens=max_tokens, think=False,
                                   schema=schema, images=[base64.b64encode(jpeg).decode()])
        report.setdefault("calls", []).append(dict(_reflect.LAST_CALL, step=step))
        return out
    return see


_ATTEMPT = """\
This picture is one attempt to show this moment from a dream you had:
{scene}{remembered}

Look at it closely and write:
- shows: one or two sentences on what is in the picture, and anything the moment describes that is missing from it.
- faults: anything that looks wrong: malformed faces, hands, animals or objects, garbled writing, a signature or watermark, parts that \
do not belong. An empty string if nothing looks wrong.
- score: from 1 to 10, how well it shows the moment. Keep 9 and 10 for a picture with everything in it and nothing wrong."""

_ATTEMPT_SCHEMA = {"type": "object", "properties": {"shows": {"type": "string"}, "faults": {"type": "string"}, "score": {"type": "integer"}},
                   "required": ["shows", "faults", "score"]}

_CHOOSE = """\
{n} pictures were drawn of this moment from a dream you had:
{scene}{remembered}

You looked at each one in turn. These are your notes:
{notes}

Choose the one to keep: what the moment describes should be there, things should look right, and it should feel like the dream.

Write:
- best: the number of the attempt you choose, from 1 to {n}.
- why: one sentence on why that one and not the others."""

_CHOOSE_SCHEMA = {"type": "object", "properties": {"best": {"type": "integer"}, "why": {"type": "string"}}, "required": ["best", "why"]}


def _chooser(ic: Dict[str, Any], report: Dict[str, Any]) -> Callable[..., str]:
    """Ask the vision model about dream pictures: with one picture to look at it, with none to compare her notes."""
    from . import reflect as _reflect
    if not ic["image_model"]:
        raise ImageError("No vision model is set.")

    def look_at(step: str, prompt: str, jpegs: List[bytes], schema: dict, max_tokens: int, think: bool = False) -> str:
        _reflect.LAST_CALL.clear()
        try:
            out = _reflect.ollama_chat(ic["image_host"], ic["image_model"], "You are choosing between pictures of your own dream. "
                                       "Reply with JSON only. " + NO_DOUBLE_QUOTES, prompt, timeout=float(ic["image_timeout"]) * (3 if think else 1),
                                       temperature=0.2, max_tokens=max_tokens, think=think, schema=schema,
                                       images=[base64.b64encode(j).decode() for j in jpegs] or None)
        except Exception as exc:
            report.setdefault("calls", []).append(dict(_reflect.LAST_CALL, step=step, think=think, failed=str(exc)[:120]))
            raise
        report.setdefault("calls", []).append(dict(_reflect.LAST_CALL, step=step))
        return out
    return look_at


def pick_best(cfg: Dict[str, Any], scene: str, pictures: List[bytes], *, remembered: str = "",
              report: Optional[Dict[str, Any]] = None, noted: Optional[Dict[int, Dict[str, Any]]] = None,
              earlier: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
    """Which of several attempts at a dream picture she keeps: (index, her reason).

    She looks at each attempt on its own and notes what it shows, what is wrong with it and a score; then she
    reads her notes side by side and chooses.  Shown all the attempts in one message, the model this was tuned
    with reported seeing one picture, or two that were the same, and kept the first.  One picture per look is
    what every vision model can do.

    If the comparison fails the best-scored attempt is kept; if she cannot look at all (no vision model, the
    server is off) it is the first.  With `dream_image_choose_think` she reasons before the comparison; that
    has a fixed allowance, and if she uses it up she is asked once more without it.

    `noted`, if given, is filled with her notes on each attempt she could look at, by index.

    `earlier` is her note on an attempt she already chose from a first drawing.  It joins the comparison as one
    more attempt without being looked at again, and if she prefers it the index returned is len(pictures).
    Comparing the two drawings by score alone made her give up a detail she had just chosen a picture for."""
    from .reflect import _parse
    if not pictures or (len(pictures) < 2 and noted is None):         # one picture is still looked at when its score is wanted
        return 0, ""
    try:
        look_at = _chooser(image_config(cfg), report if report is not None else {})
        note = f"\nIt draws on something you remember: {remembered}" if remembered else ""
    except Exception as exc:
        logger.debug("holonomic: choosing between dream pictures failed: %s", exc)
        return 0, ""
    notes: Dict[int, Dict[str, Any]] = noted if noted is not None else {}
    notes.clear()
    for i, p in enumerate(pictures):
        try:
            data = _parse(look_at(f"look at attempt {i + 1}", _ATTEMPT.format(scene=scene, remembered=note),
                                  [_jpeg(open_image(p)[0], (1024, 1024))], _ATTEMPT_SCHEMA, 300))
            notes[i] = {"shows": _sentence(data.get("shows"))[:400], "faults": _sentence(data.get("faults"))[:300],
                        "score": max(1, min(10, int(data.get("score") or 1)))}
            if report is not None and report.get("calls"):       # shown with the timings, so her choice can be followed
                report["calls"][-1]["noted"] = f"{notes[i]['score']} of 10. {notes[i]['shows']} Faults: {notes[i]['faults'] or 'none seen.'}"
        except Exception as exc:
            logger.debug("holonomic: looking at dream picture %d failed: %s", i + 1, exc)
    if not notes:
        return (len(pictures), "") if earlier else (0, "")
    known = dict(notes)
    if earlier:
        known[len(pictures)] = earlier
    top = max(known, key=lambda i: (known[i]["score"], i == len(pictures), -i))      # best score; on a tie the one already chosen, then the first
    fallback = (top, f"It scored highest of the attempts she could look at ({known[top]['score']} of 10).")
    if len(known) < 2:
        return fallback
    written = "\n".join(f"Attempt {i + 1}{' (the one you chose from the first drawing)' if earlier and i == len(pictures) else ''} "
                        f"(score {n['score']} of 10): {n['shows']} Faults: {n['faults'] or 'none seen.'}" for i, n in sorted(known.items()))
    prompt = _CHOOSE.format(n=len(pictures) + bool(earlier), scene=scene, remembered=note, notes=written)
    for think in ([True, False] if cfg.get("dream_image_choose_think") else [False]):
        try:
            data = _parse(look_at("choose", prompt, [], _CHOOSE_SCHEMA, int(cfg.get("dream_image_choose_think_tokens") or 3000), think=True)
                          if think else look_at("choose", prompt, [], _CHOOSE_SCHEMA, 200))
            best = int(data.get("best") or 0)
            if best - 1 in known:
                return best - 1, _sentence(data.get("why"))[:300]
        except Exception as exc:
            logger.debug("holonomic: choosing between dream pictures failed (thinking %s): %s", think, exc)
    return fallback


_BRIEFLY = ("\n\nYour last answer ran too long and was cut off. Be brief this time: at most four sentences of description, and for "
            "writing in the image only the few most prominent words.")


def _look_once_more(see: Callable[..., str], step: str, prompt: str, jpeg: bytes, schema: dict, max_tokens: int) -> str:
    """Ask the vision model, and if its answer ran past the limit ask again for a shorter one with more room.
    A night street full of signs made the model copy out every sign and run out of space before it finished."""
    try:
        return see(step, _SYSTEM, prompt, jpeg, schema, max_tokens)
    except ReflectionError as exc:
        if "reply limit" not in str(exc):
            raise
        return see(step + " (again, briefly)", _SYSTEM, prompt + _BRIEFLY, jpeg, schema, max_tokens * 2)


def _unreachable(exc: Exception) -> bool:
    """Whether a failure means the vision server itself is the problem, so there is no point trying the next image."""
    return isinstance(exc, OSError) or "Could not reach" in str(exc) or "No vision model" in str(exc)


def clean_labels(labels: Any, limit: int) -> List[str]:
    out: List[str] = []
    for raw in labels if isinstance(labels, list) else []:
        label = " ".join(re.sub(r"[^\w\s'\-]", " ", str(raw).lower()).split())[:40]
        if len(label) >= 2 and len(label.split()) <= 4 and label not in out:
            out.append(label)
    return out[:limit]


def _sentence(text: Any) -> str:
    """A reply that was cut off mid-sentence loses the unfinished sentence, unless that would leave nothing."""
    text = " ".join(str(text or "").split())
    return text if is_complete(text) else (trim_to_sentence(text) or text)


def _claim(engine, table: str, where: str, params: tuple) -> bool:
    now = time.time()
    with engine._lock:
        cur = _db(engine).execute(f"UPDATE {table} SET claimed_at = ? WHERE {where} AND (claimed_at IS NULL OR claimed_at < ?)",
                                  (now,) + params + (now - _CLAIM_SECONDS,))
        return cur.rowcount == 1


def describe(engine, cfg: Dict[str, Any], image_id: int, *, see: Optional[Callable[..., str]] = None,
             key_fn: Optional[Callable[[str], List[str]]] = None, report: Optional[Dict[str, Any]] = None) -> Optional[dict]:
    """Look at the whole image and store what is in it.  Returns the image, or None if there was nothing to do."""
    from .reflect import _parse
    ic = image_config(cfg)
    report = report if report is not None else {}
    row = _row(engine, image_id)
    if row is None or row["forgotten"] or row["memory_id"]:
        return None
    see = see or _seer(ic, report)
    if not _claim(engine, "images", "id = ? AND memory_id IS NULL", (int(image_id),)):
        return None
    try:
        img, _ = open_image((engine.path / row["file"]).read_bytes())
        note = _NOTE.format(caption=row["caption"]) if row["caption"] else ""
        corrections = [c for c in json.loads(row["meta"] or "{}").get("corrections", []) if c]
        if corrections:
            note += _CORRECTED.format(correction=" ".join(corrections))
        prompt = _WHOLE.format(note=note, n=int(ic["image_max_labels"]))
        data = _parse(_look_once_more(see, "image", prompt, _jpeg(img, view_box(row["width"], row["height"], ic)), _WHOLE_SCHEMA,
                                      int(ic["image_max_tokens"])))
        description = _sentence(data.get("description"))
        if len(description) < 20:
            raise ImageError("The description came back empty or cut short; it will be tried again.")
        labels = clean_labels(data.get("labels"), int(ic["image_max_labels"]))
        written = " ".join(str(data.get("text") or "").split())[:300]
        text = description + (f" Writing in the image: {written}" if written else "")
        named = key_fn(" ".join([row["caption"]] + corrections)) if key_fn and (row["caption"] or corrections) else []
        links = [i for i in json.loads(row["meta"] or "{}").get("links", []) if engine.get(int(i))]
        ids = engine.remember(text, kind=IMAGE, realm=row["realm"], session=row["session"], chain=False, whole=True,
                              keys=list(dict.fromkeys(labels[:8] + named)), links=links, trust=0.6,
                              meta={"image_id": int(image_id)})
        with engine._lock:
            db = _db(engine)
            meta = json.loads(row["meta"] or "{}")
            if isinstance(data.get("people"), bool):
                meta["people"] = data["people"]
            db.execute("UPDATE images SET memory_id = ?, described_at = ?, claimed_at = NULL, meta = ? WHERE id = ?",
                       (ids[0], time.time(), json.dumps(meta), int(image_id)))
            db.executemany("INSERT OR IGNORE INTO image_labels (image_id, label, section) VALUES (?, ?, -1)",
                           [(int(image_id), label) for label in labels])
    except BaseException:
        with engine._lock:
            _db(engine).execute("UPDATE images SET claimed_at = NULL WHERE id = ?", (int(image_id),))
        raise
    return get_image(engine, image_id)


def describe_sections(engine, cfg: Dict[str, Any], image_id: int, *, see: Optional[Callable[..., str]] = None,
                      should_stop: Optional[Callable[[], bool]] = None, report: Optional[Dict[str, Any]] = None) -> int:
    """Look at the parts of an image that have not been looked at yet.  Returns how many were looked at.
    Stops early, leaving the rest for later, when `should_stop` says the conversation has resumed."""
    from .reflect import _parse
    ic = image_config(cfg)
    report = report if report is not None else {}
    row = _row(engine, image_id)
    if row is None or row["forgotten"] or not row["memory_id"]:
        return 0
    # Writing quoted in the description of the whole is kept from the parts: each part must read a sign for
    # itself, so that two looks agreeing on what it says means something.
    whole = _without_writing((engine.get(int(row["memory_id"])) or {}).get("text") or "")
    # Only what the user affirmed is passed on.  Telling a model that there is no Tiger Beer sign puts the word
    # in front of it, and the next thing it reports seeing is a sign that says Tiger.
    told = what_was_stated([row["caption"]] + json.loads(row["meta"] or "{}").get("corrections", []))[0]
    said = (f"\nThe person who showed the image said this about the whole image, which is true: {told[:600]}\n"
            "It is about the whole image. Most of it will not be in this part: use it to name what you can see here, and do not mention "
            "anything from it that this part does not show.") if told else ""
    with engine._lock:
        todo = _db(engine).execute("SELECT * FROM image_sections WHERE image_id = ? AND notable IS NULL ORDER BY idx",
                                   (int(image_id),)).fetchall()
    if not todo:
        return 0
    see = see or _seer(ic, report)
    source = row["file"] if ic["image_from_original"] else row["view"]
    img, _ = open_image((engine.path / source).read_bytes())
    done = 0
    for s in todo:
        if should_stop and should_stop():
            break
        if not _claim(engine, "image_sections", "image_id = ? AND idx = ? AND notable IS NULL", (int(image_id), s["idx"])):
            continue
        try:
            jpeg = _crop(img, s["x"], s["y"], s["w"], s["h"], int(ic["image_section_max_side"]))
            data = _parse(_look_once_more(see, "image_part", _PART.format(place=s["place"], whole=whole, said=said), jpeg, _PART_SCHEMA,
                                          int(ic["image_max_tokens"])))
            description = _sentence(data.get("description")) if data.get("notable") else ""
            labels = clean_labels(data.get("labels"), 6) if description else []
            with engine._lock:            # 'whiskers' when the image already has 'whisker' is filed under the one it has
                have = [r["label"] for r in _db(engine).execute("SELECT DISTINCT label FROM image_labels WHERE image_id = ?", (int(image_id),))]
            labels = list(dict.fromkeys(next((h for h in have if h in label_forms(l)), l) for l in labels))
            mid = None
            if len(description) >= 15:
                mid = engine.remember(description, kind=IMAGE_PART, realm=row["realm"], session=row["session"], chain=False,
                                      whole=True, keys=labels[:4], links=[int(row["memory_id"])], salience=0.7, trust=0.6,
                                      meta={"image_id": int(image_id), "section": int(s["idx"]), "place": s["place"]})[0]
            with engine._lock:
                db = _db(engine)
                db.execute("UPDATE image_sections SET notable = ?, memory_id = ?, described_at = ?, claimed_at = NULL "
                           "WHERE image_id = ? AND idx = ?", (1 if mid else 0, mid, time.time(), int(image_id), s["idx"]))
                db.executemany("INSERT OR IGNORE INTO image_labels (image_id, label, section) VALUES (?, ?, ?)",
                               [(int(image_id), label, int(s["idx"])) for label in labels])
            done += 1
        except BaseException:
            with engine._lock:
                _db(engine).execute("UPDATE image_sections SET claimed_at = NULL WHERE image_id = ? AND idx = ?",
                                    (int(image_id), s["idx"]))
            raise
    with engine._lock:
        db = _db(engine)
        finished = not db.execute("SELECT 1 FROM image_sections WHERE image_id = ? AND notable IS NULL", (int(image_id),)).fetchone()
        if finished:
            db.execute("UPDATE images SET sections_at = ? WHERE id = ?", (time.time(), int(image_id)))
    if finished:
        report["unclear"] = report.get("unclear", 0) + len(settle_writing(engine, image_id))
    return done


# ---------------------------------------------------------- writing in images

UNCLEAR = "[writing I could not read for certain]"
_WRITTEN = " Writing in the image: "


# Quoted writing, down to a single letter ('C'); an apostrophe inside a word (the cat's face) is not a quote.
_WRITING_RE = re.compile(r"(?<!\w)'([^'\n]{1,60})'(?!\w)|\"([^\"\n]{1,80})\"|\u2018([^\u2019\n]{1,60})\u2019|\u201c([^\u201d\n]{1,80})\u201d")
_DENIAL_WORD_RE = re.compile(r"\b(?:no|not|never|without|isn't|aren't|wasn't|doesn't|don't)\b", re.IGNORECASE)


def _quotes(text: str) -> List[str]:
    return [next(g for g in m.groups() if g) for m in _WRITING_RE.finditer(text or "")]


def what_was_stated(texts: List[str]) -> Tuple[str, set]:
    """What the user affirmed about an image, and the words of what they denied.  In 'the sign says Bridgestone
    Arena, not Country Music Square' the first name is affirmed and the second denied; in 'there is no Tiger
    Beer sign' the whole of it is denied.  A denial must not be taken for a statement that the thing is there:
    that kept an invented 'Tiger' sign in a description because the correction had said there was no such sign."""
    affirmed: List[str] = []
    denied: set = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", " ".join(t for t in texts if t)):
        m = _DENIAL_WORD_RE.search(sentence)
        affirmed.append(sentence[:m.start()] if m else sentence)
        if m:
            denied |= {w for w in re.findall(r"[a-z0-9]+", sentence[m.end():].lower()) if len(w) >= 3}
    return " ".join(a.strip() for a in affirmed if a.strip()), denied - {"sign", "signs", "the", "and", "any", "there", "that", "says"}


def _letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _without_writing(text: str) -> str:
    """A description with the writing it quotes taken out."""
    return _WRITING_RE.sub("[writing]", (text or "").split(_WRITTEN.strip())[0]).strip()


def settle_writing(engine, image_id: int) -> List[str]:
    """Keep only the writing that two separate looks at the image agree on, or that the user stated.

    A vision model asked what a small sign says will often supply something plausible instead of saying it
    cannot read it: one street photo was given a 'Country Music Square', a 'Tiger Beer', a 'Budweiser' and a
    'T.J. Maxx' that were not there, each reported by a single look.  The real signs were each read the same
    way by two overlapping parts.  So writing seen in only one look is replaced by a note that it could not be
    read for certain, and labels made from it are dropped.  Returns what was taken out."""
    row = _row(engine, image_id)
    if row is None or row["forgotten"] or not row["memory_id"]:
        return []
    meta = json.loads(row["meta"] or "{}")
    if meta.get("writing_settled"):
        return []
    with engine._lock:
        parts = _db(engine).execute("SELECT idx, place, memory_id FROM image_sections WHERE image_id = ? AND memory_id IS NOT NULL "
                                    "ORDER BY idx", (row["id"],)).fetchall()
    whole = engine.get(int(row["memory_id"])) or {}
    whole_text = whole.get("text") or ""
    described, _, written = whole_text.partition(_WRITTEN)
    items = [w.strip() for w in re.split(r"[,;\n]", written) if w.strip()]
    texts = {p["idx"]: (engine.get(int(p["memory_id"])) or {}).get("text") or "" for p in parts}
    seen: Dict[str, set] = {}
    for q in _quotes(described) + items:
        seen.setdefault(_letters(q), set()).add("whole")
    for idx, text in texts.items():
        for q in _quotes(text):
            seen.setdefault(_letters(q), set()).add(idx)
    affirmed, denied = what_was_stated([row["caption"]] + meta.get("corrections", []))
    said_words = set(re.findall(r"[a-z0-9]+", affirmed.lower()))

    def stated_by_user(original: str) -> bool:           # every word of it is among the words the user affirmed
        words = re.findall(r"[a-z0-9]+", original.lower())
        return bool(words) and set(words) <= said_words
    spelled: Dict[str, str] = {}
    for q in _quotes(described) + items + [q for t in texts.values() for q in _quotes(t)]:
        spelled.setdefault(_letters(q), q)
    refused = {q for q, original in spelled.items() if denied & set(re.findall(r"[a-z0-9]+", original.lower()))}
    # Doubtful: the user said it is not there, or only one look reported it and the user did not say it is.
    doubtful = {q for q, where in seen.items() if q and (q in refused or (len(where) < 2 and not stated_by_user(spelled.get(q, ""))))}
    removed: List[str] = []

    def clean(text: str) -> str:
        def swap(m):
            quote = next(g for g in m.groups() if g)
            if _letters(quote) in doubtful:
                removed.append(quote)
                return UNCLEAR
            return m.group(0)
        return _WRITING_RE.sub(swap, text)
    new_described = clean(described)
    kept = [w for w in items if _letters(w) not in doubtful]
    removed += [w for w in items if _letters(w) in doubtful and w not in removed]
    new_whole = new_described + (_WRITTEN + ", ".join(kept) if kept else "")
    whole_id = int(row["memory_id"])
    with engine._lock:
        labels = [r["label"] for r in _db(engine).execute("SELECT DISTINCT label FROM image_labels WHERE image_id = ? AND section = -1", (row["id"],))]
    if new_whole != whole_text:
        engine.forget(whole_id)
        links = [i for i in meta.get("links", []) if engine.get(int(i))]
        whole_id = engine.remember(new_whole, kind=IMAGE, realm=row["realm"], session=row["session"], chain=False, whole=True,
                                   keys=[l for l in labels if not any(d in _letters(l) for d in doubtful)][:8], links=links, trust=0.6,
                                   meta={"image_id": int(image_id)})[0]
    for p in parts:
        new_text = clean(texts[p["idx"]])
        if new_text == texts[p["idx"]] and whole_id == int(row["memory_id"]):
            continue
        engine.forget(int(p["memory_id"]))          # re-stored so that its text, and its link to the whole, are right
        mid = engine.remember(new_text, kind=IMAGE_PART, realm=row["realm"], session=row["session"], chain=False, whole=True,
                              links=[whole_id], salience=0.7, trust=0.6,
                              meta={"image_id": int(image_id), "section": int(p["idx"]), "place": p["place"]})[0]
        with engine._lock:
            _db(engine).execute("UPDATE image_sections SET memory_id = ? WHERE image_id = ? AND idx = ?", (mid, row["id"], p["idx"]))
    meta["writing_settled"] = True
    with engine._lock:
        db = _db(engine)
        for label in [r["label"] for r in db.execute("SELECT DISTINCT label FROM image_labels WHERE image_id = ?", (row["id"],))]:
            if any(d in _letters(label) for d in doubtful):
                db.execute("DELETE FROM image_labels WHERE image_id = ? AND label = ?", (row["id"], label))
        db.execute("UPDATE images SET memory_id = ?, meta = ? WHERE id = ?", (whole_id, json.dumps(meta), row["id"]))
    return list(dict.fromkeys(removed))


def redescribe(engine, cfg: Dict[str, Any], image_id: int, *, correction: str = "", caption: Optional[str] = None,
               see: Optional[Callable[..., str]] = None, key_fn: Optional[Callable[[str], List[str]]] = None,
               report: Optional[Dict[str, Any]] = None) -> Optional[dict]:
    """Look at an image afresh: what was written about it and its parts is forgotten and written again.
    `correction` is something the person said was wrong; it is kept with the image and given to the model
    as true.  `caption` replaces what was said when the image was shown.  The parts are left waiting."""
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        return None
    meta = json.loads(row["meta"] or "{}")
    correction = " ".join((correction or "").split())[:600]
    meta.pop("writing_settled", None)
    if correction:
        meta["corrections"] = (meta.get("corrections", []) + [correction])[-5:]
    with engine._lock:
        db = _db(engine)
        old = [row["memory_id"]] + [r["memory_id"] for r in db.execute("SELECT memory_id FROM image_sections WHERE image_id = ?", (row["id"],))]
    for mid in old:
        if mid:
            engine.forget(int(mid))
    with engine._lock:
        db = _db(engine)
        db.execute("DELETE FROM image_labels WHERE image_id = ?", (row["id"],))
        db.execute("UPDATE image_sections SET notable = NULL, memory_id = NULL, described_at = NULL, claimed_at = NULL WHERE image_id = ?",
                   (row["id"],))
        db.execute("UPDATE images SET memory_id = NULL, described_at = NULL, sections_at = NULL, claimed_at = NULL, meta = ?, caption = ? "
                   "WHERE id = ?", (json.dumps(meta), row["caption"] if caption is None else " ".join(strip_image_markers(caption).split())[:600],
                                    row["id"]))
    return describe(engine, cfg, image_id, see=see, key_fn=key_fn, report=report)


def pending(engine) -> Dict[str, int]:
    with engine._lock:
        db = _db(engine)
        return {"images": db.execute("SELECT COUNT(*) FROM images WHERE forgotten = 0 AND memory_id IS NULL AND source != 'dream'").fetchone()[0],
                "sections": db.execute("SELECT COUNT(*) FROM image_sections s JOIN images i ON i.id = s.image_id "
                                       "WHERE i.forgotten = 0 AND s.notable IS NULL").fetchone()[0]}


def process(engine, cfg: Dict[str, Any], *, see: Optional[Callable[..., str]] = None, sections: bool = True,
            should_stop: Optional[Callable[[], bool]] = None, key_fn: Optional[Callable[[str], List[str]]] = None) -> Dict[str, Any]:
    """Describe everything that is waiting: whole images first, then their parts.  A failure (the vision
    server is off, say) ends the pass; what was not done stays waiting."""
    report: Dict[str, Any] = {"described": [], "sections": 0, "errors": [], "calls": [], "unclear": 0}
    with engine._lock:
        db = _db(engine)
        whole = [r["id"] for r in db.execute("SELECT id FROM images WHERE forgotten = 0 AND memory_id IS NULL AND source != 'dream' ORDER BY id")]
        parts = [r["id"] for r in db.execute(
            "SELECT DISTINCT i.id FROM images i JOIN image_sections s ON s.image_id = i.id "
            "WHERE i.forgotten = 0 AND s.notable IS NULL ORDER BY i.id")]
    # One image that cannot be described must not hold up the rest: it is reported and the pass moves on.
    # Only a failure of the server itself ends the pass.
    for image_id in whole:
        if should_stop and should_stop():
            return report
        try:
            if describe(engine, cfg, image_id, see=see, key_fn=key_fn, report=report):
                report["described"].append(image_id)
        except (ImageError, ReflectionError, OSError) as exc:
            report["errors"].append(f"image #{image_id}: {exc}")
            if _unreachable(exc):
                return report
    for image_id in (parts if sections and image_config(cfg)["image_sections"] else []):
        if should_stop and should_stop():
            return report
        try:
            report["sections"] += describe_sections(engine, cfg, image_id, see=see, should_stop=should_stop, report=report)
        except (ImageError, ReflectionError, OSError) as exc:
            report["errors"].append(f"image #{image_id}: {exc}")
            if _unreachable(exc):
                return report
    with engine._lock:                           # images finished before writing was checked this way
        finished = [r["id"] for r in _db(engine).execute(
            "SELECT id FROM images WHERE forgotten = 0 AND source != 'dream' AND sections_at IS NOT NULL AND meta NOT LIKE '%writing_settled%'")]
    for image_id in finished:
        report["unclear"] = report.get("unclear", 0) + len(settle_writing(engine, image_id))
    return report


def look(engine, cfg: Dict[str, Any], image_id: int, question: str, *, section: Optional[int] = None,
         see: Optional[Callable[..., str]] = None) -> str:
    """Look at a stored image again and answer a question about it."""
    from .reflect import _parse
    ic = image_config(cfg)
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        raise ImageError(f"No image with id {image_id}")
    see = see or _seer(ic, {})
    img, _ = open_image((engine.path / row["file"]).read_bytes())
    if section is None:
        jpeg = _jpeg(img, view_box(row["width"], row["height"], ic))
    else:
        with engine._lock:
            s = _db(engine).execute("SELECT * FROM image_sections WHERE image_id = ? AND idx = ?", (int(image_id), int(section))).fetchone()
        if s is None:
            raise ImageError(f"Image {image_id} has no section {section}")
        jpeg = _crop(img, s["x"], s["y"], s["w"], s["h"], int(ic["image_section_max_side"]))
    data = _parse(_look_once_more(see, "look", _LOOK.format(question=" ".join(question.split())[:600]), jpeg, _LOOK_SCHEMA,
                                  int(ic["image_max_tokens"])))
    return " ".join(str(data.get("answer") or "").split())


# -------------------------------------------------------------------- reading

def get_image(engine, image_id: int, *, sections: bool = False) -> Optional[dict]:
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        return None
    memory = engine.get(int(row["memory_id"])) if row["memory_id"] else None
    with engine._lock:
        db = _db(engine)
        labels = [r["label"] for r in db.execute("SELECT DISTINCT label FROM image_labels WHERE image_id = ? ORDER BY section, label",
                                                 (row["id"],))]
        parts = db.execute("SELECT * FROM image_sections WHERE image_id = ? ORDER BY idx", (row["id"],)).fetchall()
    out = {"id": int(row["id"]), "file": str(engine.path / row["view"]), "original": str(engine.path / row["file"]),
           "width": row["width"], "height": row["height"], "bytes": row["bytes"], "format": row["format"],
           "realm": row["realm"], "source": row["source"], "session": row["session"], "origin": row["origin"],
           "caption": row["caption"], "created_at": row["created_at"], "seen": row["seen"], "last_seen": row["last_seen"],
           "memory_id": row["memory_id"], "description": (memory or {}).get("text", ""), "labels": labels,
           "people": has_people(engine, row["id"]), "dream_from": dream_use(engine, row["id"]), "signature": signature_known(engine, row["id"]),
           "sections_total": len(parts), "sections_waiting": sum(1 for p in parts if p["notable"] is None)}
    if sections:
        out["sections"] = [{"section": p["idx"], "place": p["place"], "memory_id": p["memory_id"],
                            "looked_at": p["notable"] is not None,
                            "description": ((engine.get(int(p["memory_id"])) or {}).get("text", "") if p["memory_id"] else "")}
                           for p in parts]
    return out


def list_images(engine, n: int = 20, *, realm: Optional[str] = "waking") -> List[dict]:
    sql, params = "SELECT id FROM images WHERE forgotten = 0", []
    if realm:
        sql, params = sql + " AND realm = ?", [realm]
    with engine._lock:
        ids = [r["id"] for r in _db(engine).execute(sql + " ORDER BY id DESC LIMIT ?", params + [int(n)])]
    return [img for img in (get_image(engine, i) for i in ids) if img]


def count_images(engine, realm: str = "waking") -> int:
    with engine._lock:
        return _db(engine).execute("SELECT COUNT(*) FROM images WHERE forgotten = 0 AND realm = ?", (realm,)).fetchone()[0]


# ------------------------------------------------------------ dream pictures

def set_people(engine, image_id: int, people: bool) -> bool:
    """Say whether an image has real people in it, overriding what the vision model judged."""
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        return False
    meta = dict(json.loads(row["meta"] or "{}"), people_said=bool(people))
    with engine._lock:
        _db(engine).execute("UPDATE images SET meta = ? WHERE id = ?", (json.dumps(meta), row["id"]))
    return True


def set_dream_use(engine, image_id: int, allowed: Optional[bool]) -> bool:
    """Say whether dream pictures may be drawn from this image: True, False, or None to go back to the general rule."""
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        return False
    meta = json.loads(row["meta"] or "{}")
    meta.pop("dream_from", None)
    if allowed is not None:
        meta["dream_from"] = bool(allowed)
    with engine._lock:
        _db(engine).execute("UPDATE images SET meta = ? WHERE id = ?", (json.dumps(meta), row["id"]))
    return True


def dream_use(engine, image_id: int) -> Optional[bool]:
    row = _row(engine, image_id)
    value = json.loads(row["meta"] or "{}").get("dream_from") if row is not None else None
    return value if isinstance(value, bool) else None


def may_dream_from(engine, image_id: int, people_allowed: bool) -> bool:
    """Whether a dream picture may start from this image.  What was said about this image decides; otherwise an
    image with real people in it is used only if images of people are allowed in general."""
    said = dream_use(engine, image_id)
    return said if said is not None else (people_allowed or not has_people(engine, image_id))


def has_people(engine, image_id: int) -> bool:
    """Whether real people are visible in an image: what the user said if they said, otherwise what the vision
    model judged when it described the image, otherwise a guess from what was noticed in it."""
    row = _row(engine, image_id)
    if row is None:
        return False
    meta = json.loads(row["meta"] or "{}")
    people = meta.get("people_said", meta.get("people"))
    if isinstance(people, bool):
        return people
    with engine._lock:
        labels = [r["label"] for r in _db(engine).execute("SELECT DISTINCT label FROM image_labels WHERE image_id = ?", (int(image_id),))]
    return any(_PEOPLE_WORDS & set(label.split()) for label in labels)


SIGNATURE_PLACES = ("none", "top left", "top right", "bottom left", "bottom right")

_SIGNED = """\
Look at the four corners of this image. Has something been added on top of the picture there: a signature, a \
watermark, a photographer's name or logo, or a date stamp? Writing that is part of the scene itself (a sign, a label, \
a screen) does not count.

Write:
- where: the corner it is in, or none if there is nothing of the kind."""

_SIGNED_SCHEMA = {"type": "object", "properties": {"where": {"type": "string", "enum": list(SIGNATURE_PLACES)}}, "required": ["where"]}


def signature_place(engine, cfg: Dict[str, Any], image_id: int, *, see: Optional[Callable[..., str]] = None,
                    report: Optional[Dict[str, Any]] = None) -> str:
    """The corner of an image that carries a signature or watermark, 'none', or '' if it could not be found out.
    What the user said wins.  Otherwise the vision model is asked once and its answer is kept with the image."""
    from .reflect import _parse
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        return ""
    meta = json.loads(row["meta"] or "{}")
    known = meta.get("signature_said", meta.get("signature"))
    if known in SIGNATURE_PLACES:
        return known
    try:
        see = see or _seer(image_config(cfg), report if report is not None else {})
        img, _ = open_image((engine.path / (row["view"] or row["file"])).read_bytes())
        where = str(_parse(see("signature", _SYSTEM, _SIGNED, _jpeg(img, (1024, 1024)), _SIGNED_SCHEMA, 60)).get("where") or "").lower()
    except Exception as exc:
        logger.debug("holonomic: could not check image %s for a signature: %s", image_id, exc)
        return ""
    if where not in SIGNATURE_PLACES:
        return ""
    with engine._lock:
        row = _row(engine, image_id)
        _db(engine).execute("UPDATE images SET meta = ? WHERE id = ?",
                            (json.dumps(dict(json.loads(row["meta"] or "{}"), signature=where)), int(image_id)))
    return where


def set_signature(engine, image_id: int, place: str) -> bool:
    """Say where an image is signed or watermarked ('none' if it is not), overriding what the vision model judged."""
    row = _row(engine, image_id)
    place = " ".join(str(place or "").lower().replace("-", " ").split())
    if row is None or row["forgotten"] or place not in SIGNATURE_PLACES:
        return False
    with engine._lock:
        _db(engine).execute("UPDATE images SET meta = ? WHERE id = ?",
                            (json.dumps(dict(json.loads(row["meta"] or "{}"), signature_said=place)), int(image_id)))
    return True


def signature_known(engine, image_id: int) -> str:
    """What is on record about an image's signature, without asking anyone: a corner, 'none', or ''."""
    row = _row(engine, image_id)
    meta = json.loads(row["meta"] or "{}") if row else {}
    known = meta.get("signature_said", meta.get("signature"))
    return known if known in SIGNATURE_PLACES else ""


_SIGNED_RE = re.compile(r"\b(signature|signed|watermark\w*|autograph\w*|date stamp)\b", re.I)


def without_signature(text: str) -> str:
    """A description with the sentences about a signature or watermark taken out."""
    return " ".join(x for x in re.split(r"(?<=[.!?])\s+", " ".join((text or "").split())) if x and not _SIGNED_RE.search(x))


def _smooth_corner(img, place: str):
    """Blur one corner of a picture until thin strokes in it are gone, fading into the rest.  The colours stay,
    so an image generator reworking the picture fills the corner with more of what surrounds it."""
    from PIL import Image, ImageDraw, ImageFilter
    w, h = img.size
    pw, ph = int(w * 0.30), int(h * 0.16)
    x0 = 0 if "left" in place else w - pw
    y0 = 0 if "top" in place else h - ph
    feather = max(4, ph // 5)
    mask = Image.new("L", (w, h), 0)
    # the patch runs off the edges of the picture so that the fade is only on its inner sides
    ImageDraw.Draw(mask).rectangle([x0 - (feather * 3 if x0 == 0 else 0), y0 - (feather * 3 if y0 == 0 else 0),
                                    x0 + pw + (feather * 3 if x0 else 0), y0 + ph + (feather * 3 if y0 else 0)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(img.filter(ImageFilter.GaussianBlur(max(8, ph // 3))), img, mask)


def is_portrait(engine, image_id: int) -> bool:
    row = _row(engine, image_id)
    return bool(row) and row["height"] > row["width"]


def blend(engine, image_ids: List[int], size: Tuple[int, int], hide: Optional[Dict[int, str]] = None) -> Optional[bytes]:
    """One picture made of up to two stored images laid over each other, the size a dream picture will be.
    This is what an image generator is given to rework, so a dream can carry the shapes and colours of things
    she has really seen.  The stored images themselves are only read.

    `hide` names, per image, a corner to smooth over first: where it is signed or watermarked.  A photographer's
    signature was otherwise redrawn as a scrawl in the same corner of every dream picture made from the photo."""
    Image, ImageOps = _pil()
    layers = []
    for image_id in image_ids[:2]:
        row = _row(engine, image_id)
        if row is None or row["forgotten"]:
            continue
        try:
            img, _ = open_image((engine.path / (row["view"] or row["file"])).read_bytes())
        except (OSError, ImageError):
            continue
        layer = ImageOps.fit(img.convert("RGB"), size, Image.LANCZOS)
        place = (hide or {}).get(int(image_id), "")
        layers.append(_smooth_corner(layer, place) if place in SIGNATURE_PLACES[1:] else layer)
    if not layers:
        return None
    out = io.BytesIO()
    (layers[0] if len(layers) == 1 else Image.blend(layers[0], layers[1], 0.5)).save(out, "PNG")
    return out.getvalue()


def add_dream_image(engine, data: bytes, cfg: Dict[str, Any], *, dream_id: int, scene: str, sources: List[int]) -> dict:
    """Keep a picture from a dream.  It lives in the dream realm: it is never listed among images she was
    shown, never described as one, and is found only through the dream it belongs to."""
    img = add_image(engine, data, dict(cfg, image_sections=False), caption=scene, session=f"dream:{int(dream_id)}",
                    source="dream", realm="dream")
    with engine._lock:
        _db(engine).execute("UPDATE images SET meta = ? WHERE id = ?",
                            (json.dumps({"dream_id": int(dream_id), "from": [int(i) for i in sources]}), img["id"]))
    return img


def dream_pictures(engine, dream_id: int) -> List[dict]:
    with engine._lock:
        rows = _db(engine).execute("SELECT id, view, caption, meta FROM images WHERE forgotten = 0 AND realm = 'dream' AND session = ? "
                                   "ORDER BY id", (f"dream:{int(dream_id)}",)).fetchall()
    return [{"id": int(r["id"]), "file": str(engine.path / r["view"]), "scene": r["caption"],
             "from": json.loads(r["meta"] or "{}").get("from", [])} for r in rows]


def label_forms(label: str) -> List[str]:
    """'cats' should find what was labelled 'cat', and the other way round."""
    q = " ".join(re.sub(r"[^\w\s'\-]", " ", (label or "").lower()).split())
    forms = [q]
    if q.endswith("ies") and len(q) > 4:
        forms.append(q[:-3] + "y")
    if q.endswith("es") and len(q) > 3:
        forms.append(q[:-2])
    if q.endswith("s") and len(q) > 2:
        forms.append(q[:-1])
    else:
        forms += [q + "s", q + "es"]
    return [f for f in dict.fromkeys(forms) if f]


def find_by_label(engine, label: str, *, realm: Optional[str] = "waking", limit: int = 200) -> List[dict]:
    """Every image in which this thing was noticed, newest first, with the parts it was noticed in.  A label
    matches when it is the word or contains it as a whole word ('black cat' is found by 'cat')."""
    forms = set(label_forms(label))
    if not forms:
        return []
    with engine._lock:
        db = _db(engine)
        every = [r["label"] for r in db.execute("SELECT DISTINCT label FROM image_labels")]
        matching = [l for l in every if l in forms or forms & set(l.split())]
        if not matching:
            return []
        rows = db.execute(
            f"SELECT l.image_id, l.label, l.section FROM image_labels l JOIN images i ON i.id = l.image_id "
            f"WHERE i.forgotten = 0 AND l.label IN ({','.join('?' * len(matching))})" + (" AND i.realm = ?" if realm else "")
            + " ORDER BY l.image_id DESC", matching + ([realm] if realm else [])).fetchall()
    found: Dict[int, dict] = {}
    for r in rows:
        entry = found.setdefault(int(r["image_id"]), {"labels": [], "sections": []})
        if r["label"] not in entry["labels"]:
            entry["labels"].append(r["label"])
        if r["section"] >= 0 and r["section"] not in entry["sections"]:
            entry["sections"].append(int(r["section"]))
    out = []
    for image_id, entry in list(found.items())[:limit]:
        img = get_image(engine, image_id, sections=True)
        if img:
            places = {s["section"]: s["place"] for s in img.pop("sections")}
            out.append(dict(img, matched=entry["labels"], matched_in=[places.get(s, str(s)) for s in sorted(entry["sections"])]))
    return out


def all_labels(engine, limit: int = 60) -> List[Tuple[str, int]]:
    with engine._lock:
        return [(r["label"], r["n"]) for r in _db(engine).execute(
            "SELECT l.label, COUNT(DISTINCT l.image_id) AS n FROM image_labels l JOIN images i ON i.id = l.image_id "
            "WHERE i.forgotten = 0 GROUP BY l.label ORDER BY n DESC, l.label LIMIT ?", (int(limit),))]


def memory_ids(engine, described_before: Optional[float] = None) -> List[int]:
    """Ids of the memories that describe images and their parts (optionally: described before a time)."""
    cut = float("inf") if described_before is None else float(described_before)
    with engine._lock:
        db = _db(engine)
        ids = [r[0] for r in db.execute("SELECT memory_id FROM images WHERE forgotten = 0 AND memory_id IS NOT NULL "
                                        "AND described_at < ?", (cut,))]
        ids += [r[0] for r in db.execute("SELECT s.memory_id FROM image_sections s JOIN images i ON i.id = s.image_id "
                                         "WHERE i.forgotten = 0 AND s.memory_id IS NOT NULL AND s.described_at < ?", (cut,))]
    return [int(i) for i in ids]


def forget_image(engine, image_id: int, *, delete_files: bool = False) -> bool:
    """Set an image aside: she no longer recalls it, lists it or shows it, but nothing is destroyed.  What she
    wrote about it and its parts is kept with the image so that `restore_image` can put it back exactly, and
    the files stay where they are.  Only `delete_image`, on an image already set aside, removes it for good:
    getting rid of an image takes two steps so that it cannot happen by one mistake.

    `delete_files` is accepted for callers written before this and is ignored."""
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        return False
    whole = engine.get(int(row["memory_id"])) if row["memory_id"] else None
    with engine._lock:
        parts = _db(engine).execute("SELECT idx, memory_id FROM image_sections WHERE image_id = ? ORDER BY idx", (row["id"],)).fetchall()
    kept = {"at": time.time(), "text": (whole or {}).get("text", ""), "created_at": (whole or {}).get("created_at"), "parts": {}}
    for part in parts:
        mem = engine.get(int(part["memory_id"])) if part["memory_id"] else None
        if mem and mem.get("text"):
            kept["parts"][str(part["idx"])] = {"text": mem["text"], "created_at": mem.get("created_at")}
    for mid in [row["memory_id"]] + [part["memory_id"] for part in parts]:
        if mid:
            engine.forget(int(mid))
    with engine._lock:        # labels and the grid of parts stay: every search for them leaves out images set aside
        _db(engine).execute("UPDATE images SET forgotten = 1, memory_id = NULL, claimed_at = NULL, meta = ? WHERE id = ?",
                            (json.dumps(dict(json.loads(row["meta"] or "{}"), removed=kept)), row["id"]))
    return True


def removed_images(engine) -> List[dict]:
    """Images set aside and not yet deleted, most recently removed first."""
    with engine._lock:
        rows = _db(engine).execute("SELECT * FROM images WHERE forgotten = 1 ORDER BY id DESC").fetchall()
    out = []
    for r in rows:
        kept = json.loads(r["meta"] or "{}").get("removed") or {}
        out.append({"id": int(r["id"]), "origin": r["origin"], "caption": r["caption"], "realm": r["realm"], "removed_at": kept.get("at"),
                    "description": kept.get("text", ""), "parts": len(kept.get("parts") or {}), "original": str(engine.path / r["file"]),
                    "file": str(engine.path / (r["view"] or r["file"])), "files_present": (engine.path / r["file"]).exists()})
    return sorted(out, key=lambda i: -(i["removed_at"] or 0))


def restore_image(engine, cfg: Dict[str, Any], image_id: int) -> Optional[dict]:
    """Put back an image that was set aside, with what she had written about it and its parts, what was said
    when it was shown, and every correction.  None if there is no such image set aside or its file is gone."""
    row = _row(engine, image_id)
    if row is None or row["forgotten"] != 1 or not (engine.path / row["file"]).exists():
        return None
    meta = json.loads(row["meta"] or "{}")
    kept = meta.pop("removed", None)
    if kept is None:
        # Set aside by a version that kept nothing but the file: she is shown it again and starts over.
        return add_image(engine, (engine.path / row["file"]).read_bytes(), cfg, origin=row["origin"], realm=row["realm"],
                         source=row["source"], session=row["session"], count=False)
    with engine._lock:
        db = _db(engine)
        labels = db.execute("SELECT label, section FROM image_labels WHERE image_id = ?", (row["id"],)).fetchall()
        parts = db.execute("SELECT idx, place FROM image_sections WHERE image_id = ? ORDER BY idx", (row["id"],)).fetchall()
    named = lambda section, n: [l["label"] for l in labels if l["section"] == section][:n]
    whole_id = None
    if kept.get("text"):
        whole_id = engine.remember(kept["text"], kind=IMAGE, realm=row["realm"], session=row["session"], chain=False, whole=True,
                                   keys=named(-1, 8), trust=0.6, meta={"image_id": int(row["id"])}, created_at=kept.get("created_at"))[0]
    part_ids = {}
    for part in parts:
        was = (kept.get("parts") or {}).get(str(part["idx"]))
        if was and whole_id:
            part_ids[part["idx"]] = engine.remember(
                was["text"], kind=IMAGE_PART, realm=row["realm"], session=row["session"], chain=False, whole=True,
                keys=named(part["idx"], 4), links=[whole_id], salience=0.7, trust=0.6, created_at=was.get("created_at"),
                meta={"image_id": int(row["id"]), "section": int(part["idx"]), "place": part["place"]})[0]
    with engine._lock:
        db = _db(engine)
        db.execute("UPDATE images SET forgotten = 0, memory_id = ?, meta = ? WHERE id = ?", (whole_id, json.dumps(meta), row["id"]))
        for idx, mid in part_ids.items():
            db.execute("UPDATE image_sections SET memory_id = ? WHERE image_id = ? AND idx = ?", (mid, row["id"], idx))
    return get_image(engine, row["id"])


def delete_image(engine, image_id: int) -> bool:
    """Remove for good an image that was already set aside: its files, what was written about it, what was said
    when it was shown.  Refused for an image that has not been set aside first.  This cannot be undone."""
    row = _row(engine, image_id)
    if row is None or row["forgotten"] != 1:
        return False
    with engine._lock:
        db = _db(engine)
        db.execute("DELETE FROM image_labels WHERE image_id = ?", (row["id"],))
        db.execute("DELETE FROM image_sections WHERE image_id = ?", (row["id"],))
        db.execute("UPDATE images SET forgotten = 2, memory_id = NULL, caption = '', origin = '', meta = '{}' WHERE id = ?", (row["id"],))
    for rel in {row["file"], row["view"]}:
        try:
            (engine.path / rel).unlink()
        except OSError:
            pass
    return True


def group_hits(hits: list) -> List[Tuple[Any, Optional[int], list]]:
    """Recall may return an image's description and several of its parts.  Fold them into one entry per
    image, in the place of the first: (hit, image id or None, the part hits for that image)."""
    out: List[Tuple[Any, Optional[int], list]] = []
    seen: Dict[int, list] = {}
    for h in hits:
        image_id = (h.meta or {}).get("image_id") if h.kind in IMAGE_KINDS else None
        if image_id is None:
            out.append((h, None, []))
        elif image_id in seen:
            if h.kind == IMAGE_PART:
                seen[image_id].append(h)
        else:
            seen[image_id] = [h] if h.kind == IMAGE_PART else []
            out.append((h, int(image_id), seen[image_id]))
    return out
