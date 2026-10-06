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

    GET  /health  -> {"model": ..., "dim": ..., "device": ...}
    POST /embed   {"images": [base64, ...]} -> {"model": ..., "dim": ..., "vectors": [[...], ...]}
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


def _blank() -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(out, "PNG")
    return out.getvalue()


def make_server(embed: Callable[[List[bytes]], List[List[float]]], *, model: str, dim: int, device: str,
                host: str = "127.0.0.1", port: int = 8189) -> ThreadingHTTPServer:
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
                self._send(200, {"model": model, "dim": dim, "device": device})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/embed":
                return self._send(404, {"error": "not found"})
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
    args = ap.parse_args(argv)
    print(f"Loading {args.model} on {args.device} (the first run downloads it) ...", flush=True)
    try:
        embed, dim = load_embedder(args.model, args.device)
    except ImportError as exc:
        print(f"This needs torch, transformers and Pillow: {exc}\nRun it with a Python that has them, such as ComfyUI's.", file=sys.stderr)
        return 1
    server = make_server(embed, model=args.model, dim=dim, device=args.device, host=args.host, port=args.port)
    print(f"Fingerprints ready: {args.model}, {dim} numbers per picture, at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
