"""Faces: knowing a particular person in an image.

A picture fingerprint says two things look alike.  It cannot tell one person from another who looks similar,
so people are not named from it (see fingerprints.py).  A face model can: it finds each face in an image and
turns it into a fingerprint of its own, and two photographs of one person come out close together.

Whose faces are learned, and whether a person is ever named without being asked, are for the user to decide,
and everything here is off until they do:

    face_learn         none    no face is looked for at all
                       me      only the user's own face is learned and recognised; no other face is kept
                       named   people the user has named; no other face is kept
                       often   every face is kept, so that someone who keeps appearing can be noticed
    face_ask_names     with 'often': she may ask, once, who a person is who keeps appearing
    face_name_unasked  whether she is told who is in an image when the user has not asked
    dream_image_people who may be in an image a dream picture is drawn from: none, me, named, anyone
    dream_name_people  whose names a dream is given for the people in the images it draws on: none, me, named

What a person looks like is taken only from faces the user said were that person.  A face she recognised
herself is never an example, and the user can say that a face is not who it was taken for.

The face fingerprints come from the same helper server as the picture fingerprints (tools/fingerprint_server.py,
which needs OpenCV for this).  Hermes runs every top-level .py of a plugin when it loads it, so nothing
outside the standard library is imported at module level here.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LEARN = ("none", "me", "named", "often")
DREAM_PEOPLE = ("none", "me", "named", "anyone")

FACE_DEFAULTS: Dict[str, Any] = {
    "face_learn": "none",
    "face_ask_names": False,
    "face_name_unasked": False,
    "face_min": 0.40,              # how alike two faces must be to be the same person (the model's makers suggest 0.36)
    "face_often_images": 3,        # 'often': in this many different images
    "face_timeout": 60,
    "face_side": 1280,             # an image is shrunk to this before faces are looked for
    "dream_image_people": "",      # '' = as dream_image_use_people says (yes: anyone, no: none)
    # Whether a dream is told who the people in the images it draws on are, so that it can have them in it by
    # name: none, me (only the user), named (anyone the user has named).
    "dream_name_people": "none",
}

_SQL = """
CREATE TABLE IF NOT EXISTS people (
    name       TEXT PRIMARY KEY,           -- lower case, the key
    shown      TEXT NOT NULL,              -- as the user wrote it
    is_user    INTEGER NOT NULL DEFAULT 0,
    dream      INTEGER,                    -- 0: never in an image a dream picture is drawn from; NULL: by the general rule
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS image_faces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id   INTEGER NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,     -- as fractions of the image
    score      REAL,
    vec        BLOB,
    vec_model  TEXT,
    person     TEXT,                       -- a key in people, or NULL: not known
    state      TEXT,                       -- 'said': the user said so; 'seen': she recognised it; 'not': the user said it is not `person`
    alike      REAL,
    asked_at   REAL,                       -- she has asked who this is
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_image_faces ON image_faces(image_id);
"""


class FaceError(Exception):
    pass


class NeedsChoice(FaceError):
    """Several faces, and it was not said which is meant."""


def face_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(FACE_DEFAULTS)
    out.update({k: cfg[k] for k in FACE_DEFAULTS if cfg.get(k) is not None})
    out["face_learn"] = str(out["face_learn"] or "none").lower()
    if out["face_learn"] not in LEARN:
        out["face_learn"] = "none"
    out["host"] = str(cfg.get("image_fingerprint_host") or "http://127.0.0.1:8189").rstrip("/")
    return out


def faces_on(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("image_enabled")) and face_config(cfg)["face_learn"] != "none"


def _db(engine):
    from .images import _db as images_db
    db = images_db(engine)
    if not getattr(engine, "_faces_ready", False):
        with engine._lock:
            db.executescript(_SQL)
        engine._faces_ready = True
    return db


def _key(name: str) -> str:
    return " ".join(str(name or "").lower().split())[:60]


def _unpack(blob: bytes):
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32)


def _pack(vector) -> bytes:
    import numpy as np
    v = np.asarray(vector, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v).tobytes()


def _where(x: float, w: float, others: int) -> str:
    """Where a face is, in words a person would use."""
    if others == 0:
        return "the only face"
    centre = x + w / 2
    return "on the left" if centre < 0.38 else "on the right" if centre > 0.62 else "in the middle"


def make_finder(cfg: Dict[str, Any], timeout: Optional[float] = None) -> Callable[[List[bytes]], Tuple[str, List[List[dict]]]]:
    """pictures -> (name of the face model, the faces in each picture)."""
    from .fingerprints import FingerprintError, _call
    fc = face_config(cfg)

    def find(pictures: List[bytes]) -> Tuple[str, List[List[dict]]]:
        try:
            data = _call(fc["host"] + "/faces", {"images": [base64.b64encode(p).decode() for p in pictures]},
                         float(timeout or fc["face_timeout"]))
        except FingerprintError as exc:
            raise FaceError(str(exc)) from exc
        if not isinstance(data.get("faces"), list) or len(data["faces"]) != len(pictures):
            raise FaceError("The helper server replied without faces for every picture.")
        return str(data.get("model") or "faces"), data["faces"]
    return find


def available(cfg: Dict[str, Any]) -> Optional[str]:
    """The helper's face model, or None if it has none (or cannot be reached)."""
    from .fingerprints import FingerprintError, health
    try:
        return health(cfg).get("faces") or None
    except FingerprintError:
        return None


# ------------------------------------------------------------------------------------------------ people

def people(engine) -> List[dict]:
    with engine._lock:
        db = _db(engine)
        rows = db.execute("SELECT * FROM people ORDER BY is_user DESC, shown COLLATE NOCASE").fetchall()
        faces = db.execute("SELECT f.person, f.state, f.image_id FROM image_faces f JOIN images i ON i.id = f.image_id "
                           "WHERE i.forgotten = 0 AND f.person IS NOT NULL ORDER BY f.image_id").fetchall()
    return [{"name": r["shown"], "is_user": bool(r["is_user"]), "dream": None if r["dream"] is None else bool(r["dream"]),
             "said": sorted({f["image_id"] for f in faces if f["person"] == r["name"] and f["state"] == "said"}),
             "seen": sorted({f["image_id"] for f in faces if f["person"] == r["name"] and f["state"] == "seen"})} for r in rows]


def _person(engine, name: str):
    with engine._lock:
        return _db(engine).execute("SELECT * FROM people WHERE name = ?", (_key(name),)).fetchone()


def _examples(engine, model: str) -> Dict[str, list]:
    """What each known person looks like: the faces the user said were theirs."""
    out: Dict[str, list] = {}
    with engine._lock:
        rows = _db(engine).execute("SELECT f.person, f.vec FROM image_faces f JOIN images i ON i.id = f.image_id "
                                   "WHERE i.forgotten = 0 AND f.state = 'said' AND f.vec IS NOT NULL AND f.vec_model = ?", (model,)).fetchall()
    for r in rows:
        out.setdefault(r["person"], []).append(_unpack(r["vec"]))
    return out


def _who(vector, examples: Dict[str, list], floor: float, never: Optional[set] = None) -> Tuple[Optional[str], float]:
    """The person a face belongs to, and how alike; (None, best) if nobody is alike enough."""
    import numpy as np
    best, who = -1.0, None
    for person, vectors in examples.items():
        if never and person in never:
            continue
        score = float((np.stack(vectors) @ vector).max())
        if score > best:
            best, who = score, person
    return (who, best) if who is not None and best >= floor else (None, best)


def _overlap(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)), max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - ix * iy
    return ix * iy / union if union > 0 else 0.0


def _picture(engine, row, side: int) -> bytes:
    from . import images as _images
    img, _ = _images.open_image((engine.path / (row["view"] or row["file"])).read_bytes())
    return _images._jpeg(img, (side, side))


def detect(engine, cfg: Dict[str, Any], image_id: int, *, find: Optional[Callable[..., Any]] = None) -> List[dict]:
    """Look for faces in a kept image and record what the user's choice of `face_learn` allows: nothing but
    the user's own face, nothing but faces of people they have named, or every face.  A face the user spoke
    about (said it was someone, or was not) stays as it is.  Returns what is on record for the image."""
    from . import images as _images
    fc = face_config(cfg)
    row = _images._row(engine, image_id)
    if fc["face_learn"] == "none" or row is None or row["forgotten"] or row["source"] == "dream":
        return []
    model, found = (find or make_finder(cfg))([_picture(engine, row, int(fc["face_side"]))])
    found = found[0]
    examples = _examples(engine, model)
    with engine._lock:
        db = _db(engine)
        kept = db.execute("SELECT * FROM image_faces WHERE image_id = ? AND state IN ('said', 'not')", (int(image_id),)).fetchall()
        users = {r["name"] for r in db.execute("SELECT name FROM people WHERE is_user = 1")}
        db.execute("DELETE FROM image_faces WHERE image_id = ? AND (state IS NULL OR state = 'seen')", (int(image_id),))
        for f in found:
            box = tuple(float(v) for v in f["box"])
            spoken = [k for k in kept if _overlap(box, (k["x"], k["y"], k["w"], k["h"])) > 0.4]
            if any(k["state"] == "said" for k in spoken):
                continue                                     # the user said who this is
            never = {k["person"] for k in spoken if k["state"] == "not"}
            vector = _unpack(_pack(f["vector"]))
            who, alike = _who(vector, examples, float(fc["face_min"]), never)
            if fc["face_learn"] == "me" and who not in users:
                continue
            if fc["face_learn"] == "named" and who is None:
                continue
            db.execute("INSERT INTO image_faces (image_id, x, y, w, h, score, vec, vec_model, person, state, alike, created_at) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (int(image_id), *box, float(f.get("score") or 0), _pack(f["vector"]), model, who, "seen" if who else None,
                        alike if who else None, time.time()))
        meta = dict(json.loads(_images._row(engine, image_id)["meta"] or "{}"), faces={"n": len(found), "at": time.time(), "model": model})
        db.execute("UPDATE images SET meta = ? WHERE id = ?", (json.dumps(meta), int(image_id)))
    return faces_in(engine, image_id)


def faces_in(engine, image_id: int) -> List[dict]:
    """The faces on record for an image, left to right."""
    with engine._lock:
        db = _db(engine)
        rows = db.execute("SELECT f.*, p.shown, p.is_user FROM image_faces f LEFT JOIN people p ON p.name = f.person "
                          "WHERE f.image_id = ? AND (f.state IS NULL OR f.state != 'not') ORDER BY f.x", (int(image_id),)).fetchall()
    return [{"face_id": int(r["id"]), "box": (r["x"], r["y"], r["w"], r["h"]), "where": _where(r["x"], r["w"], len(rows) - 1),
             "name": r["shown"] if r["person"] else None, "is_user": bool(r["is_user"]) if r["person"] else False,
             "said": r["state"] == "said", "alike": r["alike"]} for r in rows]


def people_in(engine, image_id: int) -> List[dict]:
    """The people known to be in an image: [{"name", "is_user", "where", "said"}]."""
    return [f for f in faces_in(engine, image_id) if f["name"]]


def face_count(engine, image_id: int) -> Optional[int]:
    """How many faces were found in an image when it was last looked at for faces; None if it never was."""
    from .images import _row
    row = _row(engine, image_id)
    seen = json.loads(row["meta"] or "{}").get("faces") if row else None
    return int(seen["n"]) if isinstance(seen, dict) and "n" in seen else None


def waiting(engine) -> int:
    from .images import _db as images_db
    with engine._lock:
        rows = images_db(engine).execute("SELECT meta FROM images WHERE forgotten = 0 AND source != 'dream'").fetchall()
    return sum(1 for r in rows if "faces" not in json.loads(r["meta"] or "{}"))


def scan(engine, cfg: Dict[str, Any], *, find: Optional[Callable[..., Any]] = None, should_stop: Optional[Callable[[], bool]] = None,
         again: bool = False) -> Dict[str, Any]:
    """Look for faces in every kept image that has not been looked at for them (`again`: in every image).
    After someone is named for the first time everything is looked at again, since a face that was not kept
    before may be theirs."""
    from .images import _db as images_db
    report: Dict[str, Any] = {"done": [], "errors": []}
    if not faces_on(cfg):
        return report
    again = again or engine.kv_get("faces:look_again") == "1"
    with engine._lock:
        rows = images_db(engine).execute("SELECT id, meta FROM images WHERE forgotten = 0 AND source != 'dream' ORDER BY id").fetchall()
    try:
        find = find or make_finder(cfg)
    except FaceError as exc:
        report["errors"].append(str(exc))
        return report
    for r in rows:
        if not again and "faces" in json.loads(r["meta"] or "{}"):
            continue
        if should_stop and should_stop():
            return report
        try:
            detect(engine, cfg, r["id"], find=find)
            report["done"].append(int(r["id"]))
        except FaceError as exc:
            report["errors"].append(str(exc))
            return report
        except Exception as exc:
            report["errors"].append(f"image #{r['id']}: {exc}")
    engine.kv_set("faces:look_again", "0")
    return report


def look(engine, cfg: Dict[str, Any], image_id: int, *, find: Optional[Callable[..., Any]] = None) -> List[dict]:
    """Every face in an image as it is right now, left to right, with who each is if known.  Nothing is stored:
    this is for the user to see which face is which before naming one."""
    from . import images as _images
    fc = face_config(cfg)
    row = _images._row(engine, image_id)
    if row is None or row["forgotten"]:
        raise FaceError(f"No image with id {image_id}.")
    model, found = (find or make_finder(cfg))([_picture(engine, row, int(fc["face_side"]))])
    on_record = faces_in(engine, image_id)
    out = []
    for n, f in enumerate(found[0], 1):
        box = tuple(float(v) for v in f["box"])
        known = next((r for r in on_record if _overlap(box, r["box"]) > 0.4), None)
        out.append({"n": n, "box": box, "where": _where(box[0], box[2], len(found[0]) - 1), "vector": f["vector"], "model": model,
                    "score": f.get("score"), "name": known["name"] if known else None, "said": bool(known and known["said"])})
    return out


def name_person(engine, cfg: Dict[str, Any], image_id: int, name: str, *, face: Optional[int] = None, where: str = "",
                me: bool = False, find: Optional[Callable[..., Any]] = None) -> dict:
    """The user says a face in this image is a particular person (`me`: themselves).  That face becomes an
    example of the person.  With several faces, `face` (1 = leftmost) or `where` (left, middle, right) says
    which; NeedsChoice is raised if it cannot be told."""
    fc = face_config(cfg)
    shown, key = " ".join(str(name or "").split())[:60], _key(name)
    if fc["face_learn"] == "none":
        raise FaceError("Learning faces is off (face_learn is 'none').")
    if len(key) < 2:
        raise FaceError("A name is needed.")
    if fc["face_learn"] == "me" and not me:
        raise FaceError("Only your own face may be learned (face_learn is 'me').")
    found = look(engine, cfg, image_id, find=find)
    if not found:
        raise FaceError(f"No face was found in image #{image_id}.")
    chosen = None
    if face is not None:
        chosen = next((f for f in found if f["n"] == int(face)), None)
        if chosen is None:
            raise FaceError(f"Image #{image_id} has {len(found)} face(s); there is no face {face}.")
    elif len(found) == 1:
        chosen = found[0]
    else:
        side = {"left": "on the left", "right": "on the right", "middle": "in the middle", "centre": "in the middle", "center": "in the middle"}.get(
            str(where or "").lower().strip())
        match = [f for f in found if side and f["where"] == side]
        if len(match) == 1:
            chosen = match[0]
    if chosen is None:
        raise NeedsChoice(f"Image #{image_id} has {len(found)} faces: " + "; ".join(
            f"face {f['n']} {f['where']}" + (f" ({f['name']})" if f["name"] else "") for f in found) + ". Say which one is " + shown + ".")
    with engine._lock:
        db = _db(engine)
        first = db.execute("SELECT 1 FROM people WHERE name = ?", (key,)).fetchone() is None
        if me:
            db.execute("UPDATE people SET is_user = 0 WHERE name != ?", (key,))
        db.execute("INSERT INTO people (name, shown, is_user, created_at) VALUES (?, ?, ?, ?) "
                   "ON CONFLICT(name) DO UPDATE SET is_user = MAX(is_user, excluded.is_user)", (key, shown, 1 if me else 0, time.time()))
        for old in db.execute("SELECT id, x, y, w, h FROM image_faces WHERE image_id = ?", (int(image_id),)).fetchall():
            if _overlap(chosen["box"], (old["x"], old["y"], old["w"], old["h"])) > 0.4:
                db.execute("DELETE FROM image_faces WHERE id = ?", (old["id"],))
        db.execute("INSERT INTO image_faces (image_id, x, y, w, h, score, vec, vec_model, person, state, alike, created_at) "
                   "VALUES (?,?,?,?,?,?,?,?,?,'said',NULL,?)",
                   (int(image_id), *chosen["box"], float(chosen.get("score") or 0), _pack(chosen["vector"]), chosen["model"], key, time.time()))
        shown = db.execute("SELECT shown FROM people WHERE name = ?", (key,)).fetchone()["shown"]
    engine.kv_set("faces:look_again", "1")          # a face passed over before may be theirs
    return {"name": shown, "image_id": int(image_id), "face": chosen["n"], "where": chosen["where"], "is_user": bool(me), "new": first}


def not_person(engine, cfg: Dict[str, Any], image_id: int, name: str, *, face: Optional[int] = None) -> int:
    """The user says a face in this image is not that person.  It is never taken for them again.  Returns how
    many faces that settled (0: nothing in the image was taken for them)."""
    key = _key(name)
    with engine._lock:
        db = _db(engine)
        rows = db.execute("SELECT id FROM image_faces WHERE image_id = ? AND person = ? AND state IN ('said', 'seen') ORDER BY x",
                          (int(image_id), key)).fetchall()
        if face is not None:
            every = [r["id"] for r in db.execute("SELECT id FROM image_faces WHERE image_id = ? AND (state IS NULL OR state != 'not') ORDER BY x",
                                                 (int(image_id),))]
            rows = [r for r in rows if 1 <= int(face) <= len(every) and r["id"] == every[int(face) - 1]]
        for r in rows:                              # the place is kept so that it is not matched again; the fingerprint is not
            db.execute("UPDATE image_faces SET state = 'not', vec = NULL, alike = NULL WHERE id = ?", (r["id"],))
    return len(rows)


def set_dream(engine, name: str, allowed: Optional[bool]) -> bool:
    """Say whether a person may be in an image a dream picture is drawn from (None: by the general rule)."""
    with engine._lock:
        db = _db(engine)
        if db.execute("SELECT 1 FROM people WHERE name = ?", (_key(name),)).fetchone() is None:
            return False
        db.execute("UPDATE people SET dream = ? WHERE name = ?", (None if allowed is None else int(bool(allowed)), _key(name)))
    return True


def forget_person(engine, name: str) -> bool:
    """Forget who someone is.  Their faces are no longer anyone's, and are dropped unless every face is kept."""
    key = _key(name)
    with engine._lock:
        db = _db(engine)
        if db.execute("SELECT 1 FROM people WHERE name = ?", (key,)).fetchone() is None:
            return False
        db.execute("DELETE FROM image_faces WHERE person = ?", (key,))
        db.execute("DELETE FROM people WHERE name = ?", (key,))
    engine.kv_set("faces:look_again", "1")
    return True


def forget_all(engine) -> int:
    """Drop every face and every person.  Returns how many faces were dropped."""
    from .images import _db as images_db
    with engine._lock:
        db = _db(engine)
        n = db.execute("SELECT COUNT(*) FROM image_faces").fetchone()[0]
        db.execute("DELETE FROM image_faces")
        db.execute("DELETE FROM people")
        for r in images_db(engine).execute("SELECT id, meta FROM images").fetchall():
            meta = json.loads(r["meta"] or "{}")
            if meta.pop("faces", None) is not None:
                db.execute("UPDATE images SET meta = ? WHERE id = ?", (json.dumps(meta), r["id"]))
    return int(n)


def drop_image(engine, image_id: int) -> None:
    """An image deleted for good takes its faces with it."""
    with engine._lock:
        _db(engine).execute("DELETE FROM image_faces WHERE image_id = ?", (int(image_id),))


# ------------------------------------------------------------------------------- someone who keeps appearing

def strangers(engine, cfg: Dict[str, Any]) -> List[dict]:
    """With every face kept: the people nobody has named, grouped by face, who are in at least
    `face_often_images` different images.  [{"images": [...], "faces": [...], "asked": bool}], most seen first."""
    import numpy as np
    fc = face_config(cfg)
    with engine._lock:
        rows = _db(engine).execute("SELECT f.id, f.image_id, f.vec, f.asked_at FROM image_faces f JOIN images i ON i.id = f.image_id "
                                   "WHERE i.forgotten = 0 AND f.person IS NULL AND f.state IS NULL AND f.vec IS NOT NULL ORDER BY f.id").fetchall()
    groups: List[dict] = []
    for r in rows:
        v = _unpack(r["vec"])
        home = next((g for g in groups if float((np.stack(g["vectors"]) @ v).max()) >= float(fc["face_min"])), None)
        if home is None:
            home = {"vectors": [], "faces": [], "images": set(), "asked": False}
            groups.append(home)
        home["vectors"].append(v); home["faces"].append(int(r["id"])); home["images"].add(int(r["image_id"]))
        home["asked"] = home["asked"] or r["asked_at"] is not None
    often = [{"images": sorted(g["images"]), "faces": g["faces"], "asked": g["asked"]} for g in groups
             if len(g["images"]) >= int(fc["face_often_images"])]
    return sorted(often, key=lambda g: -len(g["images"]))


def asked(engine, face_ids: List[int]) -> None:
    with engine._lock:
        _db(engine).executemany("UPDATE image_faces SET asked_at = ? WHERE id = ?", [(time.time(), int(i)) for i in face_ids])


def stranger_in(engine, cfg: Dict[str, Any], image_id: int) -> Optional[dict]:
    """A person in this image whom she keeps seeing, does not know, and has not yet asked about."""
    mine = {f["face_id"] for f in faces_in(engine, image_id) if not f["name"]}
    if not mine:
        return None
    return next((g for g in strangers(engine, cfg) if not g["asked"] and mine & set(g["faces"])), None)


# ------------------------------------------------------------------------- a picture that has just arrived

def recognise_picture(engine, cfg: Dict[str, Any], data: bytes, *, find: Optional[Callable[..., Any]] = None,
                      timeout: float = 10.0, unknown: Optional[list] = None) -> List[dict]:
    """The known people in a picture that has only just arrived: [{"name", "is_user", "where", "alike"}].
    Nothing is stored.  Any failure gives nothing: a reply is waiting on this.  `unknown`, if given, is filled
    with the fingerprints of the faces that are nobody she knows."""
    from . import images as _images
    fc = face_config(cfg)
    if not faces_on(cfg):
        return []
    try:
        img, _ = _images.open_image(data)
        model, found = (find or make_finder(cfg, timeout))([_images._jpeg(img, (int(fc["face_side"]), int(fc["face_side"])))])
    except Exception as exc:
        logger.debug("holonomic: an arriving picture was not checked for faces: %s", exc)
        return []
    examples = _examples(engine, model)
    with engine._lock:
        known = {r["name"]: r for r in _db(engine).execute("SELECT * FROM people")}
    out = []
    for f in found[0]:
        who, alike = _who(_unpack(_pack(f["vector"])), examples, float(fc["face_min"]))
        if who and who in known and (fc["face_learn"] != "me" or known[who]["is_user"]):
            out.append({"name": known[who]["shown"], "is_user": bool(known[who]["is_user"]),
                        "where": _where(float(f["box"][0]), float(f["box"][2]), len(found[0]) - 1), "alike": round(alike, 3)})
        elif unknown is not None and who is None:
            unknown.append(_unpack(_pack(f["vector"])))
    return out


def keeps_appearing(engine, cfg: Dict[str, Any], *, image_id: Optional[int] = None, vectors: Optional[list] = None) -> Optional[dict]:
    """With every face kept and asking allowed: a person in this image (a kept one, or one that has only just
    arrived and is given as face fingerprints) who is in enough images, is not known, and has not been asked
    about.  {"images": n, "faces": [...]} or None."""
    import numpy as np
    fc = face_config(cfg)
    if not faces_on(cfg) or fc["face_learn"] != "often" or not fc["face_ask_names"]:
        return None
    if image_id is not None:
        found = stranger_in(engine, cfg, image_id)
        return {"images": len(found["images"]), "faces": found["faces"]} if found else None
    if not vectors:
        return None
    with engine._lock:
        rows = {int(r["id"]): _unpack(r["vec"]) for r in _db(engine).execute("SELECT id, vec FROM image_faces WHERE vec IS NOT NULL AND person IS NULL")}
    for group in strangers(engine, dict(cfg, face_often_images=max(1, int(fc["face_often_images"]) - 1))):
        if group["asked"]:
            continue                                     # this arriving image is one more than those already kept
        members = np.stack([rows[i] for i in group["faces"] if i in rows])
        if any(float((members @ v).max()) >= float(fc["face_min"]) for v in vectors):
            return {"images": len(group["images"]) + 1, "faces": group["faces"]}
    return None


_ASKS_WHO_RE = re.compile(r"\b(?:who(?:'s|se)?|recogni[sz]e[sd]?|know\s+(?:who|him|her|them|this|these|that|those)|"
                          r"(?:his|her|their|what(?:'s| is| are)(?: the| his| her| their)?)\s+names?)\b", re.I)


def asks_who(text: str) -> bool:
    """Whether the user is asking who someone is."""
    return bool(_ASKS_WHO_RE.search(text or ""))


def may_tell(cfg: Dict[str, Any], words: str) -> bool:
    """Whether she is to be told who is in an image now: always, if the user allows it; otherwise only when asked."""
    return faces_on(cfg) and (bool(face_config(cfg)["face_name_unasked"]) or asks_who(words))


def people_note(found: List[dict]) -> str:
    """What the vision model is told before it describes an image in which known people were found."""
    if not found:
        return ""
    listed = "; ".join((f["name"] + (" (the person who shows you images and talks with you)" if f["is_user"] else ""))
                       + (f", {f['where']}" if f["where"] != "the only face" else "") for f in found)
    return (f"\nBy their face, the people in this image include: {listed}. This comes from comparing faces and is reliable. "
            "Call them by name in the description.")


# ------------------------------------------------------------------------------- what the user said about it

_WHO_IN = """\
The person who showed this image said: {caption}

Do those words say who a person visible in the image is? 'This is me', 'that's my daughter Emma', 'me and my brother Sam'. \
A person who is only mentioned and is not in the picture does not count. Animals and things do not count.

Write:
- people: a list, empty if the words do not say who anyone in the picture is. For each: name (their name as given, or an empty \
string if they said only 'me'), me (true if it is the person speaking), where (left, middle or right if the words or the picture \
make clear which person is meant, otherwise an empty string)."""

_WHO_SCHEMA = {"type": "object", "properties": {"people": {"type": "array", "items": {"type": "object", "properties": {
    "name": {"type": "string"}, "me": {"type": "boolean"}, "where": {"type": "string", "enum": ["left", "middle", "right", ""]}},
    "required": ["name", "me", "where"]}}}, "required": ["people"]}


def learn_from_caption(engine, cfg: Dict[str, Any], image_id: int, see: Callable[..., str], *,
                       find: Optional[Callable[..., Any]] = None) -> List[dict]:
    """What the user said when showing an image may say who is in it.  Within what `face_learn` allows, those
    people are kept, with their face in this image as the example.  A person whose face cannot be told apart
    from the others in the picture is left for the user to point out."""
    from . import images as _images
    from .reflect import _parse
    fc = face_config(cfg)
    row = _images._row(engine, image_id)
    if not faces_on(cfg) or row is None or not (row["caption"] or "").strip():
        return []
    if not (face_count(engine, image_id) or 0):          # nobody's face is in it
        return []
    img, _ = _images.open_image((engine.path / (row["view"] or row["file"])).read_bytes())
    data = _parse(see("people", _images._SYSTEM, _WHO_IN.format(caption=row["caption"]), _images._jpeg(img, (1024, 1024)), _WHO_SCHEMA, 200))
    learned = []
    for item in (data.get("people") if isinstance(data.get("people"), list) else [])[:4]:
        if not isinstance(item, dict):
            continue
        me = bool(item.get("me"))
        name = " ".join(str(item.get("name") or "").split())
        if me and not re.search(r"\b(me|i|myself|i'm|im)\b", row["caption"], re.I):
            continue                                       # only if they really said so
        if not me and (len(name) < 2 or name.lower() not in row["caption"].lower()):
            continue
        if fc["face_learn"] == "me" and not me:
            continue
        if me:
            mine = next((p["name"] for p in people(engine) if p["is_user"]), None)
            name = mine or (name if len(name) >= 2 else "you")
        try:
            learned.append(name_person(engine, cfg, image_id, name, where=str(item.get("where") or ""), me=me, find=find))
        except FaceError as exc:
            logger.debug("holonomic: %s was not learned from image %s: %s", name, image_id, exc)
    return learned


# ------------------------------------------------------------------------------------------------- dreams

def dream_policy(cfg: Dict[str, Any], use_people: bool) -> str:
    said = str(face_config(cfg)["dream_image_people"] or "").lower()
    return said if said in DREAM_PEOPLE else ("anyone" if use_people else "none")


def people_allowed_in_dream(engine, cfg: Dict[str, Any], image_id: int, use_people: bool) -> bool:
    """Whether the people in an image leave it free to be drawn from in a dream, by `dream_image_people`:
    none: no image with people; me: only if every face in it is the user's; named: only if every face is
    someone the user has named; anyone: yes.  A person the user has kept out of dreams keeps any image out.
    An image with no people in it is not this function's concern and is always allowed."""
    from .images import has_people
    known = people_in(engine, image_id)
    with engine._lock:
        barred = {r["shown"] for r in _db(engine).execute("SELECT shown FROM people WHERE dream = 0")}
    if any(p["name"] in barred for p in known):
        return False
    count = face_count(engine, image_id)
    if not has_people(engine, image_id) and not count:
        return True
    policy = dream_policy(cfg, use_people)
    if policy == "anyone":
        return True
    if policy == "none" or not count:                     # people, but no face to go by: it cannot be shown who they are
        return False
    if len(known) < count:
        return False
    return all(p["is_user"] for p in known) if policy == "me" else True


DREAM_NAMES = ("none", "me", "named")


def dream_names(engine, cfg: Dict[str, Any], image_id: Optional[int]) -> str:
    """What a dream is told about who is in an image it draws on, as far as `dream_name_people` allows: nothing,
    the user only, or anyone the user has named.  A person the user has kept out of dreams is never given.
    Names that are already in what was said or written about an image are not touched by this: it only decides
    whether what she knows from faces is handed to the dream."""
    policy = str(face_config(cfg)["dream_name_people"] or "none").lower()
    if not image_id or not faces_on(cfg) or policy not in ("me", "named"):
        return ""
    with engine._lock:
        barred = {r["shown"] for r in _db(engine).execute("SELECT shown FROM people WHERE dream = 0")}
    who = [p for p in people_in(engine, image_id) if p["name"] not in barred and (policy == "named" or p["is_user"])]
    if not who:
        return ""
    return " In it, known to you by face: " + "; ".join(
        p["name"] + (" (the person you talk with)" if p["is_user"] else "") + ("" if p["where"] == "the only face" else f", {p['where']}")
        for p in who) + "."
