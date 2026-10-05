"""Making a picture from words, for dreams.

The plugin does not draw.  It asks an image-generation server that the user runs, through one of two
widely supported interfaces:

  a1111    the Stable Diffusion WebUI API (AUTOMATIC1111, Forge, SD.Next):  /sdapi/v1/txt2img, /sdapi/v1/img2img
  openai   the OpenAI images API, which several local servers also speak:   /v1/images/generations, /v1/images/edits

A painter is a function (prompt, starting picture or None) -> image bytes.  With a starting picture the
server is asked to rework it towards the prompt instead of starting from noise.

Only the standard library is imported: Hermes executes every top-level module in this folder at load.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, Optional

APIS = ("a1111", "openai")


class PaintError(RuntimeError):
    pass


def _post(url: str, body: bytes, content_type: str, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise PaintError(f"The image server returned {exc.code} for {url}: {exc.read().decode(errors='replace')[:300]}") from exc
    except TimeoutError as exc:
        raise PaintError(f"The image server did not finish within {timeout:.0f} s") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        if "timed out" in str(exc).lower():
            raise PaintError(f"The image server did not finish within {timeout:.0f} s") from exc
        raise PaintError(f"Could not reach the image server at {url}: {exc}") from exc


def _decode(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise PaintError("The image server replied without a picture.")
    try:
        return base64.b64decode(value.split(",", 1)[1] if value.startswith("data:") else value)
    except Exception as exc:
        raise PaintError("The image server's picture could not be decoded.") from exc


def _multipart(fields: Dict[str, str], name: str, filename: str, data: bytes, mime: str) -> tuple:
    boundary = "----holonomic" + uuid.uuid4().hex
    out = b""
    for key, value in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
    out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return out, f"multipart/form-data; boundary={boundary}"


def make_painter(sc: Dict[str, Any]) -> Callable[[str, Optional[bytes]], bytes]:
    api = str(sc.get("dream_image_api") or "").lower()
    host = str(sc.get("dream_image_host") or "").rstrip("/")
    if api not in APIS or not host:
        raise PaintError("No image generator is set. Run: hermes holonomic dreams images pictures --api a1111|openai --host URL")
    model = str(sc.get("dream_image_model") or "")
    width, height = int(sc.get("dream_image_width") or 768), int(sc.get("dream_image_height") or 512)
    steps, timeout = int(sc.get("dream_image_steps") or 0), float(sc.get("dream_image_timeout") or 600)
    strength = min(max(float(sc.get("dream_image_strength") or 0.6), 0.05), 1.0)
    negative = str(sc.get("dream_image_negative") or "")

    def a1111(prompt: str, start: Optional[bytes]) -> bytes:
        body: Dict[str, Any] = {"prompt": prompt, "width": width, "height": height, "batch_size": 1, "n_iter": 1}
        if negative:
            body["negative_prompt"] = negative
        if steps:
            body["steps"] = steps
        if model:
            body["override_settings"] = {"sd_model_checkpoint": model}
        if start:
            body.update(init_images=[base64.b64encode(start).decode()], denoising_strength=strength)
        data = _post(f"{host}/sdapi/v1/{'img2img' if start else 'txt2img'}", json.dumps(body).encode(), "application/json", timeout)
        images = data.get("images") or []
        return _decode(images[0] if images else None)

    def openai(prompt: str, start: Optional[bytes]) -> bytes:
        fields = {"prompt": prompt, "size": f"{width}x{height}", "n": "1", "response_format": "b64_json"}
        if model:
            fields["model"] = model
        if start:
            body, content_type = _multipart(fields, "image", "start.png", start, "image/png")
            data = _post(f"{host}/v1/images/edits", body, content_type, timeout)
        else:
            data = _post(f"{host}/v1/images/generations", json.dumps(dict(fields, n=1)).encode(), "application/json", timeout)
        first = (data.get("data") or [{}])[0]
        if first.get("b64_json"):
            return _decode(first["b64_json"])
        if str(first.get("url") or "").startswith(("http://", "https://")):
            try:
                with urllib.request.urlopen(first["url"], timeout=timeout) as resp:
                    return resp.read()
            except (urllib.error.URLError, OSError) as exc:
                raise PaintError(f"Could not fetch the picture the image server made: {exc}") from exc
        raise PaintError("The image server replied without a picture.")

    return a1111 if api == "a1111" else openai
