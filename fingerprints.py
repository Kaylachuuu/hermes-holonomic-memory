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
                model: Optional[str] = None, should_stop: Optional[Callable[[], bool]] = None, redo: bool = False) -> Dict[str, Any]:
    """Give every kept image that lacks one a fingerprint, and each of its parts.  An image whose fingerprint
    was made by a different model than the server now runs is done again, since the two cannot be compared.
    A server that cannot be reached ends the pass; an image that cannot be read is reported and passed over."""
    from . import images as _images
    report: Dict[str, Any] = {"done": [], "errors": [], "model": model or ""}
    fc = fingerprint_config(cfg)
    ic = _images.image_config(cfg)
    try:
        embed = embed or make_embedder(cfg)
        if model is None and not redo:
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
