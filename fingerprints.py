"""Picture fingerprints: recognising what an image shows from the image itself, without words.

A description says what the vision model thought to mention.  A fingerprint is a list of numbers worked out
from the pixels by a model made for the purpose; two pictures of the same thing come out close together
whether or not anyone would describe them alike.  Each image she keeps gets one for the whole picture and one
for each of its parts, so that a thing in the corner of one image can be matched with the same thing filling
another.

The numbers come from a small helper server (tools/fingerprint_server.py) because Ollama cannot make them.
They are kept in the `vec` columns that images.py reserved for this, with the name of the model that made
them: fingerprints from different models cannot be compared, so only those of one model ever are.

Hermes runs every top-level .py of a plugin when it loads it, so nothing outside the standard library is
imported at module level here.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FINGERPRINT_DEFAULTS: Dict[str, Any] = {
    "image_fingerprints": False,                           # make and use fingerprints
    "image_fingerprint_host": "http://127.0.0.1:8189",     # where tools/fingerprint_server.py listens
    "image_fingerprint_timeout": 120,
    "image_fingerprint_side": 448,                         # pictures are shrunk to this before they are sent
    # How alike two pictures must be (1 = the same, 0 = nothing in common) to be called alike.  Set by trying
    # it on your own images: `hermes holonomic images similar ID --all` shows the figures for every image.
    "image_fingerprint_min": 0.5,
    # Named things: an animal, object or place the user has named ("this is Sushi") is recognised in later images
    # by its look, and the vision model is told what it may be looking at.  Needs fingerprints.
    "image_names": True,
    "image_name_min": 0.6,                                 # how alike a part must be to a named thing to be offered as it
}


class FingerprintError(Exception):
    pass


def fingerprint_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(FINGERPRINT_DEFAULTS)
    out.update({k: cfg[k] for k in FINGERPRINT_DEFAULTS if cfg.get(k) is not None})
    out["image_fingerprint_host"] = str(out["image_fingerprint_host"] or "").rstrip("/")
    return out


def _call(url: str, body: Optional[dict], timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise FingerprintError(f"The fingerprint server answered {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise FingerprintError(f"Could not reach the fingerprint server at {url.rsplit('/', 1)[0]}: {exc}") from exc


def health(cfg: Dict[str, Any], timeout: float = 5.0) -> dict:
    """What the helper server says about itself: its model, the length of a fingerprint, the device."""
    fc = fingerprint_config(cfg)
    if not fc["image_fingerprint_host"]:
        raise FingerprintError("No fingerprint server is set.")
    return _call(fc["image_fingerprint_host"] + "/health", None, timeout)


def make_embedder(cfg: Dict[str, Any]) -> Callable[[List[bytes]], Tuple[str, List[List[float]]]]:
    """pictures -> (name of the model, one fingerprint per picture)."""
    fc = fingerprint_config(cfg)
    if not fc["image_fingerprint_host"]:
        raise FingerprintError("No fingerprint server is set.")

    def embed(pictures: List[bytes]) -> Tuple[str, List[List[float]]]:
        data = _call(fc["image_fingerprint_host"] + "/embed", {"images": [base64.b64encode(p).decode() for p in pictures]},
                     float(fc["image_fingerprint_timeout"]))
        vectors = data.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(pictures) or not data.get("model"):
            raise FingerprintError("The fingerprint server replied without a fingerprint for every picture.")
        return str(data["model"]), vectors
    return embed


def _pack(vector: List[float]) -> bytes:
    import numpy as np
    v = np.asarray(vector, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v).tobytes()


def _unpack(blob: bytes):
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32)


def status(engine, model: Optional[str] = None) -> Dict[str, Any]:
    """How many kept images have a fingerprint.  With `model`, one made by any other model counts as waiting."""
    from .images import _db
    with engine._lock:
        rows = _db(engine).execute("SELECT vec IS NOT NULL AS has, vec_model FROM images WHERE forgotten = 0 AND source != 'dream'").fetchall()
    done = [r for r in rows if r["has"] and (model is None or r["vec_model"] == model)]
    return {"images": len(rows), "fingerprinted": len(done), "waiting": len(rows) - len(done),
            "models": sorted({r["vec_model"] for r in rows if r["has"] and r["vec_model"]})}


def fingerprint(engine, cfg: Dict[str, Any], *, embed: Optional[Callable[..., Tuple[str, List[List[float]]]]] = None,
                model: Optional[str] = None, should_stop: Optional[Callable[[], bool]] = None, redo: bool = False,
                only: Optional[int] = None) -> Dict[str, Any]:
    """Give every kept image that lacks one a fingerprint, and each of its parts.  An image whose fingerprint
    was made by a different model than the server now runs is done again, since the two cannot be compared.
    A server that cannot be reached ends the pass; an image that cannot be read is reported and passed over."""
    from . import images as _images
    report: Dict[str, Any] = {"done": [], "errors": [], "model": model or ""}
    fc = fingerprint_config(cfg)
    ic = _images.image_config(cfg)
    try:
        embed = embed or make_embedder(cfg)
        if model is None and not redo and only is None:
            try:
                model = str(health(cfg).get("model") or "") or None
            except FingerprintError:
                model = None                         # an embedder was handed in, or the server will say so below
    except FingerprintError as exc:
        report["errors"].append(str(exc))
        return report
    with engine._lock:
        rows = _images._db(engine).execute("SELECT id, file, view, vec IS NOT NULL AS has, vec_model FROM images "
                                           "WHERE forgotten = 0 AND source != 'dream' ORDER BY id").fetchall()
    side = int(fc["image_fingerprint_side"])
    for row in rows:
        if only is not None and int(row["id"]) != int(only):
            continue
        if row["has"] and not redo and (model is None or row["vec_model"] == model):
            continue
        if should_stop and should_stop():
            break
        try:
            source = row["file"] if ic["image_from_original"] else (row["view"] or row["file"])
            img, _ = _images.open_image((engine.path / source).read_bytes())
            with engine._lock:
                parts = _images._db(engine).execute("SELECT idx, x, y, w, h FROM image_sections WHERE image_id = ? ORDER BY idx",
                                                    (row["id"],)).fetchall()
            pictures = [_images._jpeg(img, (side, side))] + [_images._crop(img, p["x"], p["y"], p["w"], p["h"], side) for p in parts]
            made_by, vectors = embed(pictures)
        except FingerprintError as exc:
            report["errors"].append(str(exc))
            break
        except (OSError, _images.ImageError) as exc:
            report["errors"].append(f"image #{row['id']}: {exc}")
            continue
        with engine._lock:
            db = _images._db(engine)
            db.execute("UPDATE images SET vec = ?, vec_model = ? WHERE id = ?", (_pack(vectors[0]), made_by, row["id"]))
            for p, v in zip(parts, vectors[1:]):
                db.execute("UPDATE image_sections SET vec = ?, vec_model = ? WHERE image_id = ? AND idx = ?", (_pack(v), made_by, row["id"], p["idx"]))
        report["done"].append(int(row["id"]))
        report["model"] = model = made_by
    return report


def _loaded(engine, model: str) -> Dict[int, List[tuple]]:
    """image id -> [(place, vector)], the whole image first ('whole').  A part known to show nothing worth
    recording is left out: two plain walls are alike, and it says nothing about the images."""
    from .images import _db
    out: Dict[int, List[tuple]] = {}
    with engine._lock:
        db = _db(engine)
        for r in db.execute("SELECT id, vec FROM images WHERE forgotten = 0 AND source != 'dream' AND vec IS NOT NULL AND vec_model = ?", (model,)):
            out[int(r["id"])] = [("whole", _unpack(r["vec"]))]
        for r in db.execute("SELECT s.image_id, s.place, s.vec FROM image_sections s JOIN images i ON i.id = s.image_id "
                            "WHERE i.forgotten = 0 AND s.vec IS NOT NULL AND s.vec_model = ? AND (s.notable IS NULL OR s.notable = 1) "
                            "ORDER BY s.image_id, s.idx", (model,)):
            if int(r["image_id"]) in out:
                out[int(r["image_id"])].append((r["place"], _unpack(r["vec"])))
    return out


def similar(engine, cfg: Dict[str, Any], image_id: int, *, n: int = 6, minimum: Optional[float] = None) -> List[dict]:
    """Images that look like this one, most alike first.  Each says how alike the whole pictures are, and
    which part of this image is most like which part of that one: the same cat can be the whole of one
    picture and a corner of another.  `minimum` -1 lists every image with its figures."""
    import numpy as np
    from .images import _row
    row = _row(engine, image_id)
    if row is None or row["forgotten"] or row["vec"] is None or not row["vec_model"]:
        return []
    floor = float(fingerprint_config(cfg)["image_fingerprint_min"] if minimum is None else minimum)
    every = _loaded(engine, row["vec_model"])
    mine = every.pop(int(image_id), None)
    if not mine:
        return []
    A = np.stack([v for _, v in mine])
    found = []
    for other, theirs in every.items():
        S = A @ np.stack([v for _, v in theirs]).T
        whole = float(S[0, 0])
        a, b = np.unravel_index(int(np.argmax(S)), S.shape)
        best = float(S[a, b])
        if best >= floor:
            found.append({"id": other, "alike": round(best, 3), "whole": round(whole, 3), "this": mine[a][0], "that": theirs[b][0]})
    return sorted(found, key=lambda f: -f["alike"])[:max(1, int(n))]


def describe_match(match: dict) -> str:
    """'the whole of both', or 'the top left of this and the centre of that'."""
    if match["this"] == "whole" and match["that"] == "whole":
        return "the two pictures as a whole"
    side = lambda place, which: f"the whole of {which}" if place == "whole" else f"the {place} of {which}"
    return f"{side(match['this'], 'this one')} and {side(match['that'], 'that one')}"


# ------------------------------------------------------------------------------------------- named things
#
# "This is Sushi."  From then on a part of a new image that looks like Sushi is offered to the vision model as
# possibly Sushi before it describes the image, so that she can recognise a particular animal, object or place
# and not only a kind of thing.  Two things have to agree before a name is used: the fingerprint says the part
# looks like it, and the vision model, told what to look for, says it can see it.
#
# What a named thing looks like is taken from the images the user named it in: the parts of those images whose
# descriptions mention the name, or the whole image if none does.  An image she recognised the thing in by
# herself is never used as an example, so that one mistake cannot grow into many.
#
# People are left out.  A fingerprint says "this looks like that"; it cannot tell one person from another who
# looks similar, and naming a person from it would be a guess presented as recognition.

import re
import time

_NAMES_SQL = """
CREATE TABLE IF NOT EXISTS image_names (
    name       TEXT PRIMARY KEY,           -- lower case, the key
    shown      TEXT NOT NULL,              -- as the user wrote it
    what       TEXT NOT NULL DEFAULT '',   -- 'a long-haired black and white cat'
    kind       TEXT NOT NULL DEFAULT '',   -- animal, object, place
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS image_name_marks (
    name       TEXT NOT NULL,
    image_id   INTEGER NOT NULL,
    state      TEXT NOT NULL,              -- 'example': the user said so; 'recognised': she saw it herself; 'not': the user said it is not
    alike      REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (name, image_id)
) WITHOUT ROWID;
"""

KINDS = ("animal", "object", "place")


class NameRefused(Exception):
    pass


def _ndb(engine):
    from .images import _db
    db = _db(engine)
    if not getattr(engine, "_image_names_ready", False):
        with engine._lock:
            db.executescript(_NAMES_SQL)
        engine._image_names_ready = True
    return db


def _key(name: str) -> str:
    return " ".join(str(name or "").lower().split())[:60]


def _mentions(shown: str, text: str) -> bool:
    return bool(shown) and re.search(r"(?<!\w)" + re.escape(shown) + r"(?!\w)", text or "", re.I) is not None


def names_on(cfg: Dict[str, Any]) -> bool:
    fc = fingerprint_config(cfg)
    return bool(fc["image_fingerprints"] and fc["image_names"])


def _texts(engine, image_id: int) -> Tuple[str, Dict[str, str]]:
    """What is written about an image: (the whole, {place: that part})."""
    from .images import _db, _row
    row = _row(engine, image_id)
    whole = ((engine.get(int(row["memory_id"])) or {}).get("text", "") if row and row["memory_id"] else "")
    with engine._lock:
        parts = _db(engine).execute("SELECT place, memory_id FROM image_sections WHERE image_id = ? AND memory_id IS NOT NULL", (int(image_id),)).fetchall()
    return whole, {p["place"]: (engine.get(int(p["memory_id"])) or {}).get("text", "") for p in parts}


def _vectors(engine, image_id: int, model: str) -> List[tuple]:
    """[(place, vector)] for one image, the whole first, every part included."""
    from .images import _db
    with engine._lock:
        db = _db(engine)
        row = db.execute("SELECT vec FROM images WHERE id = ? AND forgotten = 0 AND vec IS NOT NULL AND vec_model = ?", (int(image_id), model)).fetchone()
        parts = db.execute("SELECT place, vec FROM image_sections WHERE image_id = ? AND vec IS NOT NULL AND vec_model = ? ORDER BY idx",
                           (int(image_id), model)).fetchall()
    return ([("whole", _unpack(row["vec"]))] + [(p["place"], _unpack(p["vec"])) for p in parts]) if row else []


def _examples(engine, key: str, shown: str, model: str, exclude: int) -> list:
    """The fingerprints that say what a named thing looks like."""
    with engine._lock:
        ids = [r["image_id"] for r in _ndb(engine).execute("SELECT image_id FROM image_name_marks WHERE name = ? AND state = 'example'", (key,))]
    out = []
    for image_id in ids:
        if int(image_id) == int(exclude):
            continue
        vectors = _vectors(engine, image_id, model)
        if not vectors:
            continue
        _, parts = _texts(engine, image_id)
        told = [v for place, v in vectors[1:] if _mentions(shown, parts.get(place, ""))]
        out += told or [vectors[0][1]]
    return out


def recognise(engine, cfg: Dict[str, Any], image_id: int, *, embed: Optional[Callable[..., Any]] = None) -> Dict[str, dict]:
    """Named things this image may show, going by its look alone:
    {key: {"shown", "what", "alike", "places": {place: alike}}}.  Makes the image's fingerprint first if it
    has none.  Empty when names are off, the helper cannot be reached, or nothing is alike enough."""
    import numpy as np
    from .images import _row
    if not names_on(cfg):
        return {}
    with engine._lock:
        db = _ndb(engine)
        known = db.execute("SELECT name, shown, what FROM image_names").fetchall()
        told_not = {r["name"] for r in db.execute("SELECT name FROM image_name_marks WHERE image_id = ? AND state = 'not'", (int(image_id),))}
    if not known:
        return {}
    row = _row(engine, image_id)
    if row is None or row["forgotten"]:
        return {}
    if row["vec"] is None:
        if fingerprint(engine, cfg, embed=embed, only=int(image_id))["errors"]:
            return {}
        row = _row(engine, image_id)
        if row["vec"] is None:
            return {}
    mine = _vectors(engine, image_id, row["vec_model"])
    if not mine:
        return {}
    A, floor, out = np.stack([v for _, v in mine]), float(fingerprint_config(cfg)["image_name_min"]), {}
    for k in known:
        if k["name"] in told_not:
            continue
        examples = _examples(engine, k["name"], k["shown"], row["vec_model"], int(image_id))
        if not examples:
            continue
        best = (A @ np.stack(examples).T).max(axis=1)
        places = {mine[i][0]: round(float(b), 3) for i, b in enumerate(best) if b >= floor}
        if places:
            out[k["name"]] = {"shown": k["shown"], "what": k["what"], "alike": max(places.values()), "places": places}
    return out


def known_note(matches: Dict[str, dict], place: Optional[str] = None) -> str:
    """What the vision model is told before it looks.  With `place`, only what was matched in that part."""
    found = [m for m in matches.values() if place is None or place in m["places"]]
    if not found:
        return ""
    listed = "; ".join(m["shown"] + (f" ({m['what']})" if m["what"] else "") for m in sorted(found, key=lambda m: -m["alike"])[:4])
    where = "this image" if place is None else "this part"
    return (f"\nGoing by its look alone, {where} may show something you have been shown before: {listed}. Look for it. "
            f"If you can see it, call it by that name. If you cannot, do not mention it.")


def note_recognised(engine, image_id: int, matches: Dict[str, dict], text: str) -> List[str]:
    """After she has described an image or a part: the names she used, of those offered, are recorded as
    recognised in it, and the image can be found by them.  Returns those names."""
    used = []
    for key, m in matches.items():
        if not _mentions(m["shown"], text):
            continue
        with engine._lock:
            db = _ndb(engine)
            db.execute("INSERT OR IGNORE INTO image_name_marks (name, image_id, state, alike, created_at) VALUES (?, ?, 'recognised', ?, ?)",
                       (key, int(image_id), float(m["alike"]), time.time()))
            db.execute("INSERT OR IGNORE INTO image_labels (image_id, label, section) VALUES (?, ?, -1)", (int(image_id), key))
        used.append(m["shown"])
    return used


def name_thing(engine, cfg: Dict[str, Any], image_id: int, name: str, *, what: str = "", kind: str = "", redo: bool = True,
               see: Optional[Callable[..., str]] = None, key_fn: Optional[Callable[[str], List[str]]] = None) -> dict:
    """The user says this image shows a particular thing with this name.  The image becomes an example of what
    the thing looks like.  If what is written about the image does not use the name yet, it is looked at again
    with that taken as true, so that the parts showing the thing say so."""
    from . import images as _images
    shown, key = " ".join(str(name or "").split())[:60], _key(name)
    kind = str(kind or "").lower()
    if len(key) < 2:
        raise NameRefused("A name is needed.")
    if kind == "person":
        raise NameRefused("People are not named from fingerprints: a fingerprint cannot tell one person from another who looks similar.")
    if _images.get_image(engine, image_id) is None:
        raise NameRefused(f"No image with id {image_id}.")
    what = " ".join(str(what or "").split()).strip(" .")[:120]
    with engine._lock:
        db = _ndb(engine)
        had = db.execute("SELECT what, kind FROM image_names WHERE name = ?", (key,)).fetchone()
        db.execute("INSERT INTO image_names (name, shown, what, kind, created_at) VALUES (?, ?, ?, ?, ?) "
                   "ON CONFLICT(name) DO UPDATE SET what = excluded.what, kind = excluded.kind",
                   (key, shown, what or (had["what"] if had else ""), kind if kind in KINDS else (had["kind"] if had else ""), time.time()))
        db.execute("INSERT INTO image_name_marks (name, image_id, state, alike, created_at) VALUES (?, ?, 'example', NULL, ?) "
                   "ON CONFLICT(name, image_id) DO UPDATE SET state = 'example'", (key, int(image_id), time.time()))
        db.execute("INSERT OR IGNORE INTO image_labels (image_id, label, section) VALUES (?, ?, -1)", (int(image_id), key))
        shown, what = db.execute("SELECT shown, what FROM image_names WHERE name = ?", (key,)).fetchone()
    whole, parts = _texts(engine, image_id)
    again = False
    if redo and not _mentions(shown, " ".join([whole] + list(parts.values()))):
        try:
            again = bool(_images.redescribe(engine, cfg, image_id, correction=f"This shows {shown}" + (f", {what}" if what else "") + ".",
                                            see=see, key_fn=key_fn))
            with engine._lock:                      # describing it again cleared its labels
                _ndb(engine).execute("INSERT OR IGNORE INTO image_labels (image_id, label, section) VALUES (?, ?, -1)", (int(image_id), key))
        except Exception as exc:                    # no vision model to hand: the whole image stands as the example
            logger.debug("holonomic: image %s was not looked at again after being named: %s", image_id, exc)
    return {"name": shown, "what": what, "image_id": int(image_id), "looked_again": again}


def not_named(engine, cfg: Dict[str, Any], image_id: int, name: str, *, redo: bool = True,
              see: Optional[Callable[..., str]] = None, key_fn: Optional[Callable[[str], List[str]]] = None) -> bool:
    """The user says this image does not show that thing.  It is never offered for this image again, and if
    what is written about the image uses the name, the image is looked at again with that taken as true."""
    from . import images as _images
    key = _key(name)
    with engine._lock:
        db = _ndb(engine)
        known = db.execute("SELECT shown FROM image_names WHERE name = ?", (key,)).fetchone()
        if known is None or _images._row(engine, image_id) is None:
            return False
        db.execute("INSERT INTO image_name_marks (name, image_id, state, alike, created_at) VALUES (?, ?, 'not', NULL, ?) "
                   "ON CONFLICT(name, image_id) DO UPDATE SET state = 'not'", (key, int(image_id), time.time()))
        db.execute("DELETE FROM image_labels WHERE image_id = ? AND label = ?", (int(image_id), key))
    whole, parts = _texts(engine, image_id)
    if redo and _mentions(known["shown"], " ".join([whole] + list(parts.values()))):
        try:
            _images.redescribe(engine, cfg, image_id, correction=f"This does not show {known['shown']}.", see=see, key_fn=key_fn)
        except Exception as exc:
            logger.debug("holonomic: image %s was not looked at again after a name was taken back: %s", image_id, exc)
    return True


_NAMED_IN = """\
The person who showed this image said: {caption}

Do those words give a name to one particular animal, object or place that is visible in the image, such as their cat, their \
car or their house? A kind of thing ('a cat') is not a name. Something that is not in the picture does not count.

Write:
- names: a list, empty if they named nothing in the picture. For each: name (as they gave it), what (what it is, in a few \
words that would let someone pick it out, e.g. 'a long-haired black and white cat'), kind (animal, object, place or person)."""

_NAMED_SCHEMA = {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "object", "properties": {
    "name": {"type": "string"}, "what": {"type": "string"}, "kind": {"type": "string", "enum": ["animal", "object", "place", "person"]}},
    "required": ["name", "what", "kind"]}}}, "required": ["names"]}


def learn_from_caption(engine, cfg: Dict[str, Any], image_id: int, see: Callable[..., str]) -> List[dict]:
    """What the user said when showing an image may name something in it ("this is Sushi").  Those names are
    kept, with this image as their example.  People are passed over."""
    from . import images as _images
    from .reflect import _parse
    row = _images._row(engine, image_id)
    if not names_on(cfg) or row is None or not (row["caption"] or "").strip():
        return []
    img, _ = _images.open_image((engine.path / (row["view"] or row["file"])).read_bytes())
    data = _parse(see("names", _images._SYSTEM, _NAMED_IN.format(caption=row["caption"]), _images._jpeg(img, (1024, 1024)), _NAMED_SCHEMA, 200))
    learned = []
    for item in (data.get("names") if isinstance(data.get("names"), list) else [])[:3]:
        if not isinstance(item, dict) or str(item.get("kind") or "").lower() not in KINDS:
            continue
        name = " ".join(str(item.get("name") or "").split())
        if len(name) < 2 or not _mentions(name, row["caption"]):          # only a name that was really said
            continue
        try:
            learned.append(name_thing(engine, cfg, image_id, name, what=str(item.get("what") or ""), kind=item["kind"], redo=False))
        except NameRefused:
            pass
    return learned


def relabel(engine, image_id: int) -> None:
    """Describing an image again clears its labels; the names it is known to show are put back."""
    with engine._lock:
        db = _ndb(engine)
        keys = [r["name"] for r in db.execute("SELECT name FROM image_name_marks WHERE image_id = ? AND state != 'not'", (int(image_id),))]
        db.executemany("INSERT OR IGNORE INTO image_labels (image_id, label, section) VALUES (?, ?, -1)", [(int(image_id), k) for k in keys])


def unmark(engine, image_id: int) -> None:
    """An image deleted for good takes its part in any name with it."""
    with engine._lock:
        _ndb(engine).execute("DELETE FROM image_name_marks WHERE image_id = ?", (int(image_id),))


def names(engine) -> List[dict]:
    with engine._lock:
        db = _ndb(engine)
        rows = db.execute("SELECT * FROM image_names ORDER BY shown COLLATE NOCASE").fetchall()
        marks = db.execute("SELECT m.name, m.state, m.image_id FROM image_name_marks m JOIN images i ON i.id = m.image_id "
                           "WHERE i.forgotten = 0 ORDER BY m.image_id").fetchall()
    return [{"name": r["shown"], "what": r["what"], "kind": r["kind"],
             "examples": [m["image_id"] for m in marks if m["name"] == r["name"] and m["state"] == "example"],
             "recognised": [m["image_id"] for m in marks if m["name"] == r["name"] and m["state"] == "recognised"],
             "not": [m["image_id"] for m in marks if m["name"] == r["name"] and m["state"] == "not"]} for r in rows]


def names_in(engine, image_id: int) -> List[dict]:
    """The named things an image shows: [{"name", "said": True if the user said so, "alike"}]."""
    with engine._lock:
        rows = _ndb(engine).execute("SELECT n.shown, m.state, m.alike FROM image_name_marks m JOIN image_names n ON n.name = m.name "
                                    "WHERE m.image_id = ? AND m.state != 'not' ORDER BY n.shown", (int(image_id),)).fetchall()
    return [{"name": r["shown"], "said": r["state"] == "example", "alike": r["alike"]} for r in rows]


def forget_name(engine, name: str) -> bool:
    """Drop a name: nothing is recognised as it any more.  What was written about images stays as it is."""
    key = _key(name)
    with engine._lock:
        db = _ndb(engine)
        if db.execute("SELECT 1 FROM image_names WHERE name = ?", (key,)).fetchone() is None:
            return False
        db.execute("DELETE FROM image_name_marks WHERE name = ?", (key,))
        db.execute("DELETE FROM image_names WHERE name = ?", (key,))
    return True
