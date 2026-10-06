"""A small server that turns pictures into fingerprints, for the holonomic memory plugin.

A fingerprint is a list of numbers worked out from the picture itself.  Two pictures of the same thing come out
close together, whatever words anyone would use for them.  Ollama cannot make these, so this runs beside it.

It is not part of the plugin that Hermes loads: run it with a Python that has `torch` and `transformers`,
for example the one ComfyUI uses:

    C:\\Users\\you\\ComfyUI\\.venv\\Scripts\\python.exe fingerprint_server.py

The first run downloads the model (about 350 MB for the default) from Hugging Face.

    --model   a Hugging Face vision model (default facebook/dinov2-base)
    --device  cpu (default), xpu, cuda or mps.  The processor is fast enough, and leaves the graphics card alone.
    --host    address to listen on (default 127.0.0.1; use 0.0.0.0 to let another machine reach it)
    --port    default 8189

    GET  /health  -> {"model": ..., "dim": ..., "device": ..., "faces": name of the face model or null}
    POST /embed   {"images": [base64, ...]} -> {"model": ..., "dim": ..., "vectors": [[...], ...]}
    POST /faces   {"images": [base64, ...]} -> {"model": ..., "faces": [[{"box": [x, y, w, h], "score": s, "vector": [...]}, ...], ...]}

Faces are optional and need one more package, OpenCV, in the same Python:

    C:\\Users\\you\\ComfyUI\\.venv\\Scripts\\python.exe -m pip install opencv-python-headless

With it present, two small models (39 MB together) are downloaded on first start: one finds faces, one turns
each face into a fingerprint of its own.  A face box is given as fractions of the picture's width and height.
Without OpenCV everything else works and /faces answers that faces are not available.  --no-faces turns
them off.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List

MAX_IMAGES = 32
MAX_BODY = 80_000_000


def load_embedder(model_name: str, device: str) -> tuple:
    """(a function from pictures as bytes to unit-length vectors, the vector length)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    try:                                    # phone photos
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    lock = threading.Lock()                 # one batch at a time

    def embed(pictures: List[bytes]) -> List[List[float]]:
        images = [Image.open(io.BytesIO(p)).convert("RGB") for p in pictures]
        with lock, torch.no_grad():
            out = model(**{k: v.to(device) for k, v in processor(images=images, return_tensors="pt").items()})
            # The summary token the model itself produces.  Models without one: the mean over the picture.
            vec = getattr(out, "pooler_output", None)
            if vec is None:
                vec = out.last_hidden_state.mean(dim=1)
            vec = torch.nn.functional.normalize(vec.float(), dim=-1)
        return vec.cpu().tolist()

    dim = len(embed([_blank()])[0])
    return embed, dim


FACE_MODELS = {      # from the OpenCV model zoo (github.com/opencv/opencv_zoo), Apache-2.0 / MIT
    "face_detection_yunet_2023mar.onnx":
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
FACE_MODEL_NAME = "yunet-2023mar+sface-2021dec"


def _fetch(name: str, url: str, folder: str) -> str:
    import os
    import urllib.request
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        print(f"Downloading {name} ...", flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) < 100_000:
            raise RuntimeError(f"{name} did not download properly from {url}")
        os.replace(tmp, path)
    return path


def load_face_finder(folder: str, *, min_score: float = 0.8, max_side: int = 1280) -> Callable[[List[bytes]], List[List[dict]]]:
    """A function from pictures to the faces in each: where it is, how sure, and a unit-length fingerprint."""
    import cv2
    import numpy as np
    detector_path, recogniser_path = [_fetch(name, url, folder) for name, url in FACE_MODELS.items()]
    recogniser = cv2.FaceRecognizerSF.create(recogniser_path, "")
    lock = threading.Lock()

    def find(pictures: List[bytes]) -> List[List[dict]]:
        out = []
        for data in pictures:
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("not a picture OpenCV can read (send JPEG or PNG)")
            h, w = img.shape[:2]
            scale = min(1.0, max_side / float(max(h, w)))
            if scale < 1.0:
                img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
            found = []
            with lock:
                detector = cv2.FaceDetectorYN.create(detector_path, "", (w, h), min_score, 0.3, 5000)
                _, faces = detector.detect(img)
                for face in (faces if faces is not None else []):
                    x, y, fw, fh = [float(v) for v in face[:4]]
                    if fw < 20 or fh < 20:                  # too small to tell anyone by
                        continue
                    vec = recogniser.feature(recogniser.alignCrop(img, face)).astype("float32").reshape(-1)
                    norm = float(np.linalg.norm(vec))
                    if norm <= 0:
                        continue
                    found.append({"box": [max(0.0, x / w), max(0.0, y / h), min(1.0, fw / w), min(1.0, fh / h)],
                                  "score": round(float(face[14]), 3), "vector": (vec / norm).tolist()})
            out.append(sorted(found, key=lambda f: f["box"][0]))       # left to right
        return out
    return find


def _blank() -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(out, "PNG")
    return out.getvalue()


def make_server(embed: Callable[[List[bytes]], List[List[float]]], *, model: str, dim: int, device: str,
                host: str = "127.0.0.1", port: int = 8189,
                faces: Callable[[List[bytes]], List[List[dict]]] | None = None, face_model: str = FACE_MODEL_NAME) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(200, {"model": model, "dim": dim, "device": device, "faces": face_model if faces else None})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            what = self.path.rstrip("/")
            if what not in ("/embed", "/faces"):
                return self._send(404, {"error": "not found"})
            if what == "/faces" and faces is None:
                return self._send(501, {"error": "faces are not available: install opencv-python-headless in this Python and restart"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY:
                    return self._send(413, {"error": "the request is empty or too large"})
                images = json.loads(self.rfile.read(length)).get("images")
                if not isinstance(images, list) or not 0 < len(images) <= MAX_IMAGES:
                    return self._send(400, {"error": f"'images' must be a list of 1 to {MAX_IMAGES} base64 pictures"})
                pictures = [base64.b64decode(i) for i in images]
            except (ValueError, TypeError) as exc:
                return self._send(400, {"error": f"could not read the request: {exc}"})
            try:
                if what == "/faces":
                    return self._send(200, {"model": face_model, "faces": faces(pictures)})
                vectors = embed(pictures)
            except Exception as exc:                # a file that is not a picture, or the model failing
                return self._send(422, {"error": f"could not fingerprint: {exc}"})
            self._send(200, {"model": model, "dim": dim, "vectors": vectors})

        def log_message(self, *args) -> None:       # one line per picture batch would bury anything useful
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Picture fingerprints for the holonomic memory plugin")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--no-faces", action="store_true", help="do not load the face models")
    ap.add_argument("--models", default="", help="folder for the face models (default: a .cache folder in your home)")
    args = ap.parse_args(argv)
    print(f"Loading {args.model} on {args.device} (the first run downloads it) ...", flush=True)
    try:
        embed, dim = load_embedder(args.model, args.device)
    except ImportError as exc:
        print(f"This needs torch, transformers and Pillow: {exc}\nRun it with a Python that has them, such as ComfyUI's.", file=sys.stderr)
        return 1
    faces = None
    if not args.no_faces:
        try:
            import os
            faces = load_face_finder(args.models or os.path.join(os.path.expanduser("~"), ".cache", "holonomic-faces"))
        except ImportError:
            print("Faces are off: OpenCV is not installed in this Python (pip install opencv-python-headless). Everything else works.", flush=True)
        except Exception as exc:
            print(f"Faces are off: {exc}", flush=True)
    server = make_server(embed, model=args.model, dim=dim, device=args.device, host=args.host, port=args.port, faces=faces)
    print(f"Fingerprints ready: {args.model}, {dim} numbers per picture, at http://{args.host}:{args.port}"
          + (f"; faces: {FACE_MODEL_NAME}" if faces else "; faces: off"), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
