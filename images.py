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
    "image_view_width": 1024,        # the copy she looks at fits inside this (turned on its side for a portrait image)
    "image_view_height": 768,
    "image_sections": True,          # also look at the image part by part
    "image_grid": 3,                 # 3 = nine parts, each half the width and height, overlapping by half
    "image_section_min_side": 640,   # smaller images are not cut up: there is nothing more to see
    "image_section_max_side": 1024,  # a part is shrunk to this before the model sees it
    "image_from_original": True,     # cut the parts from the original, which may hold more detail than the copy
    "image_sections_when": "idle",   # 'idle': when the conversation has gone quiet; 'now': straight away
    "image_idle_seconds": 120,
    "image_timeout": 180,
    "image_max_tokens": 700,
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
- text: any writing visible in the image, copied exactly, or an empty string if there is none."""

_NOTE = ("\nThe person who showed it said: {caption}\nIf those words name someone or something that is visible, use that name. "
         "Take nothing else from them.")

_PART = """\
This is one part of a larger image: the {place}.
The whole image was described as: {whole}

Write:
- notable: false if this part shows nothing worth recording on its own (plain background, sky, wall, floor, blur, or only the edge of something); otherwise true.
- description: one to three sentences on what is visible in this part, giving detail that the description of the whole lacks. An empty string if not notable.
- labels: up to 6 short lowercase names for the distinct things visible in this part, each a singular noun or short noun phrase. An empty list if not notable."""

_LOOK = """\
Look at this image and answer the question from what is visible. If the image does not show the answer, say so.

Question: {question}"""

_LABELS = {"type": "array", "items": {"type": "string"}}
_WHOLE_SCHEMA = {"type": "object", "properties": {"description": {"type": "string"}, "labels": _LABELS, "text": {"type": "string"}},
                 "required": ["description", "labels", "text"]}
_PART_SCHEMA = {"type": "object", "properties": {"notable": {"type": "boolean"}, "description": {"type": "string"}, "labels": _LABELS},
                "required": ["notable", "description", "labels"]}
_LOOK_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

# Hermes marks a message that carries an image in one of two ways.  For a model that can see, the picture
# goes in the message and the text gets `[Image attached at: <path>]`.  For one it believes cannot, it has
# another model describe the picture and puts that description, and the path, in the text instead.
_HINT_RE = re.compile(r"\[Image attached at: ([^\]\r\n]+)\]|\bimage_url: ([^\]\r\n]+)\]")
_MARKER_RE = re.compile(r"\[Image attached(?: at)?: [^\]\r\n]+\]|\[\d+ images?\]"
                        r"|\[The user attached an image.*?\bimage_url: [^\]\r\n]+\]", re.DOTALL)
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
    text = _MARKER_RE.sub("", text or "")
    return re.sub(r"\n{3,}", "\n\n", text.replace(_DEFAULT_CAPTION, "")).strip()


def attached_paths(text: str) -> List[str]:
    return list(dict.fromkeys((a or b).strip() for a, b in _HINT_RE.findall(text or "")))


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
              source: str = "user", realm: str = "waking", links: Tuple[int, ...] | List[int] = ()) -> dict:
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
        prompt = _WHOLE.format(note=note, n=int(ic["image_max_labels"]))
        data = _parse(see("image", _SYSTEM, prompt, _jpeg(img, view_box(row["width"], row["height"], ic)), _WHOLE_SCHEMA,
                          int(ic["image_max_tokens"])))
        description = _sentence(data.get("description"))
        if len(description) < 20:
            raise ImageError("The description came back empty or cut short; it will be tried again.")
        labels = clean_labels(data.get("labels"), int(ic["image_max_labels"]))
        written = " ".join(str(data.get("text") or "").split())[:300]
        text = description + (f" Writing in the image: {written}" if written else "")
        named = key_fn(row["caption"]) if key_fn and row["caption"] else []
        links = [i for i in json.loads(row["meta"] or "{}").get("links", []) if engine.get(int(i))]
        ids = engine.remember(text, kind=IMAGE, realm=row["realm"], session=row["session"], chain=False, whole=True,
                              keys=list(dict.fromkeys(labels[:8] + named)), links=links, trust=0.6,
                              meta={"image_id": int(image_id)})
        with engine._lock:
            db = _db(engine)
            db.execute("UPDATE images SET memory_id = ?, described_at = ?, claimed_at = NULL WHERE id = ?",
                       (ids[0], time.time(), int(image_id)))
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
    whole = (engine.get(int(row["memory_id"])) or {}).get("text") or ""
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
            data = _parse(see("image_part", _SYSTEM, _PART.format(place=s["place"], whole=whole), jpeg, _PART_SCHEMA,
                              int(ic["image_max_tokens"])))
            description = _sentence(data.get("description")) if data.get("notable") else ""
            labels = clean_labels(data.get("labels"), 6) if description else []
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
        if not db.execute("SELECT 1 FROM image_sections WHERE image_id = ? AND notable IS NULL", (int(image_id),)).fetchone():
            db.execute("UPDATE images SET sections_at = ? WHERE id = ?", (time.time(), int(image_id)))
    return done


def pending(engine) -> Dict[str, int]:
    with engine._lock:
        db = _db(engine)
        return {"images": db.execute("SELECT COUNT(*) FROM images WHERE forgotten = 0 AND memory_id IS NULL").fetchone()[0],
                "sections": db.execute("SELECT COUNT(*) FROM image_sections s JOIN images i ON i.id = s.image_id "
                                       "WHERE i.forgotten = 0 AND s.notable IS NULL").fetchone()[0]}


def process(engine, cfg: Dict[str, Any], *, see: Optional[Callable[..., str]] = None, sections: bool = True,
            should_stop: Optional[Callable[[], bool]] = None, key_fn: Optional[Callable[[str], List[str]]] = None) -> Dict[str, Any]:
    """Describe everything that is waiting: whole images first, then their parts.  A failure (the vision
    server is off, say) ends the pass; what was not done stays waiting."""
    report: Dict[str, Any] = {"described": [], "sections": 0, "errors": [], "calls": []}
    with engine._lock:
        db = _db(engine)
        whole = [r["id"] for r in db.execute("SELECT id FROM images WHERE forgotten = 0 AND memory_id IS NULL ORDER BY id")]
        parts = [r["id"] for r in db.execute(
            "SELECT DISTINCT i.id FROM images i JOIN image_sections s ON s.image_id = i.id "
            "WHERE i.forgotten = 0 AND s.notable IS NULL ORDER BY i.id")]
    try:
        for image_id in whole:
            if should_stop and should_stop():
                return report
            if describe(engine, cfg, image_id, see=see, key_fn=key_fn, report=report):
                report["described"].append(image_id)
        for image_id in (parts if sections and image_config(cfg)["image_sections"] else []):
            if should_stop and should_stop():
                return report
            report["sections"] += describe_sections(engine, cfg, image_id, see=see, should_stop=should_stop, report=report)
    except (ImageError, ReflectionError, OSError) as exc:
        report["errors"].append(str(exc))
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
    data = _parse(see("look", _SYSTEM, _LOOK.format(question=" ".join(question.split())[:600]), jpeg, _LOOK_SCHEMA,
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


def count_images(engine) -> int:
    with engine._lock:
        return _db(engine).execute("SELECT COUNT(*) FROM images WHERE forgotten = 0").fetchone()[0]


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
    """Forget an image: its description, its parts and its labels.  The files are removed only when asked,
    because that cannot be undone."""
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        return False
    with engine._lock:
        db = _db(engine)
        mids = [row["memory_id"]] + [r["memory_id"] for r in db.execute("SELECT memory_id FROM image_sections WHERE image_id = ?", (row["id"],))]
    for mid in mids:
        if mid:
            engine.forget(int(mid))
    with engine._lock:
        db = _db(engine)
        db.execute("DELETE FROM image_labels WHERE image_id = ?", (row["id"],))
        db.execute("DELETE FROM image_sections WHERE image_id = ?", (row["id"],))
        db.execute("UPDATE images SET forgotten = 1, memory_id = NULL, caption = '', origin = '', meta = '{}' WHERE id = ?", (row["id"],))
    if delete_files:
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
