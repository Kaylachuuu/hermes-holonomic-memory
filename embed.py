"""Embedders: turn text into vectors.

OllamaEmbedder talks to a local Ollama server (default model: nomic-embed-text).
HashEmbedder is a dependency-free lexical fallback used by the tests.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request

import numpy as np

# Models that expect a task prefix on every input, and the prefixes they expect.
_PREFIXES = {
    "nomic-embed-text": {"document": "search_document: ", "query": "search_query: "},
    "mxbai-embed-large": {"document": "", "query": "Represent this sentence for searching relevant passages: "},
}


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434",
                 prefixes: dict | None = None, timeout: float = 30.0, batch_size: int = 32):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        if prefixes is None:
            base = model.split(":")[0].split("/")[-1].lower()
            prefixes = next((p for name, p in _PREFIXES.items() if base.startswith(name)), {})
        self.prefixes = {"document": prefixes.get("document", ""), "query": prefixes.get("query", "")}
        self._dim: int | None = None

    @property
    def signature(self) -> str:
        return f"ollama:{self.model.split(':')[0]}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"], "document").shape[1])
        return self._dim

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        prefix = self.prefixes.get(kind, "")
        inputs = [prefix + t for t in texts]
        out: list[list[float]] = []
        try:
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i:i + self.batch_size]
                try:
                    out.extend(self._post("/api/embed", {"model": self.model, "input": batch, "truncate": True})["embeddings"])
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                    # Older Ollama: one prompt per call on the legacy endpoint.
                    out.extend(self._post("/api/embeddings", {"model": self.model, "prompt": t})["embedding"] for t in batch)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            raise EmbeddingError(f"Ollama embedding failed ({self.model} at {self.host}): {exc}") from exc
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise EmbeddingError(f"Ollama returned {arr.shape} for {len(texts)} inputs")
        self._dim = int(arr.shape[1])
        return arr


class HashEmbedder:
    """Bag-of-words embedding from hashed tokens.  Captures word overlap only;
    it exists so the engine can be tested (and can limp along) without a model."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    @property
    def signature(self) -> str:
        return f"hash:{self.dim}"

    def _token(self, tok: str) -> np.ndarray:
        vec = self._cache.get(tok)
        if vec is None:
            seed = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "little")
            vec = self._cache[tok] = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return vec

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = [t for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 2] or ["__empty__"]
            for tok in set(toks):
                out[i] += self._token(tok)
            out[i] /= max(float(np.linalg.norm(out[i])), 1e-12)
        return out
