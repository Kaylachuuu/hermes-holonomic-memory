"""Making a picture from words, for dreams.

The plugin does not draw.  It asks an image-generation server that the user runs, through one of two
of three interfaces:

  comfyui  ComfyUI: a job is queued at /prompt, watched at /history and its picture fetched from /view
  a1111    the Stable Diffusion WebUI API (AUTOMATIC1111, Forge, SD.Next):  /sdapi/v1/txt2img, /sdapi/v1/img2img
  openai   the OpenAI images API, which several local servers also speak:   /v1/images/generations, /v1/images/edits

A painter is a function (prompt, starting picture or None, size or None) -> image bytes.  With a starting picture the
server is asked to rework it towards the prompt instead of starting from noise.

Only the standard library is imported: Hermes executes every top-level module in this folder at load.
"""

from __future__ import annotations

import base64
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Dict, Optional

APIS = ("comfyui", "a1111", "openai")


class PaintError(RuntimeError):
    pass


def _post(url: str, body: Optional[bytes], content_type: str, timeout: float, raw: bool = False) -> Any:
    """POST `body`, or GET when there is none.  Returns the parsed JSON reply, or the bytes when `raw`."""
    req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return data if raw else json.loads(data)
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
        raise PaintError("No image generator is set. Run: hermes holonomic dreams images pictures --api comfyui|a1111|openai --host URL")
    model = str(sc.get("dream_image_model") or "")
    default_width, default_height = int(sc.get("dream_image_width") or 768), int(sc.get("dream_image_height") or 512)
    steps, timeout = int(sc.get("dream_image_steps") or 0), float(sc.get("dream_image_timeout") or 600)
    strength = min(max(float(sc.get("dream_image_strength") or 0.6), 0.05), 1.0)
    negative = str(sc.get("dream_image_negative") or "")
    poll = float(sc.get("dream_image_poll_seconds") or 1.0)
    family = str(sc.get("dream_image_family") or "").lower()          # comfyui: '' (work it out from the model's name), 'checkpoint', 'zimage'
    text_encoder, vae = str(sc.get("dream_image_text_encoder") or ""), str(sc.get("dream_image_vae") or "")

    def a1111(prompt: str, start: Optional[bytes], size: Optional[tuple] = None) -> bytes:
        width, height = size or (default_width, default_height)
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

    def openai(prompt: str, start: Optional[bytes], size: Optional[tuple] = None) -> bytes:
        width, height = size or (default_width, default_height)
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

    def comfyui(prompt: str, start: Optional[bytes], size: Optional[tuple] = None) -> bytes:
        width, height = size or (default_width, default_height)
        checkpoint = model
        zimage = family == "zimage" or (not family and "z_image" in model.lower().replace("-", "_"))
        if not checkpoint and zimage:
            checkpoint = "z_image_turbo_bf16.safetensors"
        if not checkpoint:                       # none named: use the first one the server has
            info = _post(f"{host}/object_info/CheckpointLoaderSimple", None, "", min(timeout, 60))
            try:
                names = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            except (KeyError, IndexError, TypeError):
                names = []
            if not names or not isinstance(names, list):
                raise PaintError("ComfyUI has no checkpoint to draw with. Put a model file in ComfyUI/models/checkpoints.")
            checkpoint = names[0]
        if zimage:
            # Z-Image-Turbo comes as three files (the model, a language model that reads the prompt, and a decoder)
            # and is drawn in a few steps with no negative prompt.
            model_out, clip_out, vae_out, empty = ["13", 0], ["11", 0], ["12", 0], "EmptySD3LatentImage"
            graph: Dict[str, Any] = {
                "10": {"class_type": "UNETLoader", "inputs": {"unet_name": checkpoint, "weight_dtype": "default"}},
                "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder or "qwen_3_4b.safetensors", "type": "lumina2"}},
                "12": {"class_type": "VAELoader", "inputs": {"vae_name": vae or "ae.safetensors"}},
                "13": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["10", 0], "shift": 3.0}},
                "3": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}},
            }
            sampler = {"steps": steps or 9, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}
        else:
            model_out, clip_out, vae_out, empty = ["1", 0], ["1", 1], ["1", 2], "EmptyLatentImage"
            graph = {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip_out}},
            }
            sampler = {"steps": steps or 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal"}
        graph.update({
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip_out}},
            "5": {"class_type": "KSampler", "inputs": dict(
                sampler, seed=random.getrandbits(48), denoise=strength if start else 1.0, model=model_out, positive=["2", 0],
                negative=["3", 0], latent_image=["4", 0])},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": vae_out}},
            "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "holonomic_dream", "images": ["6", 0]}},
        })
        if start:
            body, content_type = _multipart({"overwrite": "true"}, "image", f"holonomic_{uuid.uuid4().hex[:12]}.png", start, "image/png")
            uploaded = _post(f"{host}/upload/image", body, content_type, timeout)
            name = "/".join(p for p in (uploaded.get("subfolder"), uploaded.get("name")) if p)
            graph["8"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            graph["9"] = {"class_type": "ImageScale", "inputs": {"image": ["8", 0], "upscale_method": "lanczos", "width": width,
                                                                 "height": height, "crop": "center"}}
            graph["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": vae_out}}
        else:
            graph["4"] = {"class_type": empty, "inputs": {"width": width, "height": height, "batch_size": 1}}
        queued = _post(f"{host}/prompt", json.dumps({"prompt": graph, "client_id": "holonomic"}).encode(), "application/json", min(timeout, 60))
        job = queued.get("prompt_id")
        if not job:
            raise PaintError(f"ComfyUI did not accept the job: {json.dumps(queued)[:300]}")
        deadline = time.time() + timeout
        while time.time() < deadline:
            entry = (_post(f"{host}/history/{job}", None, "", min(timeout, 60)) or {}).get(job)
            if entry:
                for output in (entry.get("outputs") or {}).values():
                    for image in output.get("images") or []:
                        query = urllib.parse.urlencode({"filename": image.get("filename", ""), "subfolder": image.get("subfolder", ""),
                                                        "type": image.get("type", "output")})
                        return _post(f"{host}/view?{query}", None, "", min(timeout, 60), raw=True)
                status = entry.get("status") or {}
                if status.get("completed") or status.get("status_str") == "error":
                    raise PaintError(f"ComfyUI finished without a picture: {json.dumps(status)[:300]}")
            time.sleep(poll)
        raise PaintError(f"The image server did not finish within {timeout:.0f} s")

    return {"a1111": a1111, "openai": openai, "comfyui": comfyui}[api]
