"""Phasor (Fourier-HRR) algebra for the holonomic memory engine.

Every concept is a vector of unit-modulus complex numbers ("phasors").
  bind    = element-wise multiply      (phases add)
  unbind  = multiply by the conjugate  (phases subtract)
  bundle  = plain addition             (superposition; this is what a plate is)

Embeddings from a language model are lifted into phasor space with a fixed
complex random projection, keeping only the phase.  Similar embeddings give
similar phasors, so a cue that merely *resembles* a stored key still resonates
with it.  The projection can be run backwards (`Projector.back`) to turn a
noisy unbound vector into an embedding-space estimate for cleanup.

References: Plate (1995) Holographic Reduced Representations; Plate (2003)
on frequency-domain HRRs; Gayler (2004) Vector Symbolic Architectures.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np

# E|z| for z ~ CN(0, 1): the gain of a phase-only projection run backwards.
KAPPA = math.sqrt(math.pi) / 2.0
_TWO_PI = 2.0 * math.pi


def _hash_uint16(name: str, n: int) -> np.ndarray:
    """n deterministic uint16 values from SHA-256 counter blocks (stable across
    platforms, Python versions and numpy versions, unlike numpy's RNG streams)."""
    blocks = math.ceil(n / 16)
    raw = b"".join(hashlib.sha256(f"{name}:{i}".encode()).digest() for i in range(blocks))
    return np.frombuffer(raw, dtype="<u2")[:n]


@lru_cache(maxsize=4096)
def atom(name: str, dim: int) -> np.ndarray:
    """Deterministic random phasor vector for a discrete symbol (a key or role)."""
    phases = _hash_uint16(name, dim).astype(np.float64) * (_TWO_PI / 65536.0)
    out = np.exp(1j * phases).astype(np.complex64)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=8)
def permutation(name: str, dim: int) -> np.ndarray:
    """Deterministic permutation of range(dim), used to mark the cue side of a binding."""
    keys = _hash_uint16("perm:" + name, dim * 2).astype(np.uint32)
    keys = (keys[0::2] << 16) | keys[1::2]
    perm = np.argsort(keys, kind="stable")
    perm.setflags(write=False)
    return perm


def make_projection(embed_dim: int, dim: int, seed: int = 20261002) -> np.ndarray:
    """Complex Gaussian projection matrix (dim x embed_dim), entries ~ CN(0, 1).
    Generated once per memory store and then persisted, so it never changes."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((dim, embed_dim), dtype=np.float32)
    imag = rng.standard_normal((dim, embed_dim), dtype=np.float32)
    return ((real + 1j * imag) / math.sqrt(2.0)).astype(np.complex64)


class Projector:
    """Moves vectors between embedding space and phasor space."""

    def __init__(self, matrix: np.ndarray, center: np.ndarray):
        self.P = np.ascontiguousarray(matrix, dtype=np.complex64)
        self._P_conj = np.conj(self.P)
        self.center = np.asarray(center, dtype=np.float32)
        self.dim, self.embed_dim = self.P.shape

    def prep(self, embeddings: np.ndarray) -> np.ndarray:
        """Centre and unit-normalise raw embeddings.  Centring removes the large
        component that all texts share, which would otherwise make every memory
        resonate weakly with every cue."""
        x = np.atleast_2d(np.asarray(embeddings, dtype=np.float32)) - self.center
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)

    def phasor(self, prepped: np.ndarray) -> np.ndarray:
        """Prepared embeddings (n, embed_dim) -> phasors (n, dim)."""
        z = np.atleast_2d(prepped).astype(np.float32) @ self.P.T
        return (z / np.maximum(np.abs(z), 1e-12)).astype(np.complex64)

    def back(self, vectors: np.ndarray, aperture_dim: int | None = None) -> np.ndarray:
        """Phasor-space vectors (n, d) -> embedding-space estimates (n, embed_dim).
        back(phasor(x)) ~= x.  With `aperture_dim`, only the first d components of
        the plate are used: a fragment of the hologram still reconstructs the
        whole, at lower fidelity."""
        v = np.atleast_2d(vectors)
        d = v.shape[1] if aperture_dim is None else aperture_dim
        return (v[:, :d] @ self._P_conj[:d]).real.astype(np.float32) / (KAPPA * d)


def noise_sigma(load: float, cue_weight2: float, d: int) -> float:
    """Std-dev of the crosstalk a plate adds to one cleanup score, given the
    plate's load (sum of squared binding weights), the cue's squared weight, and
    the number of phasor components used."""
    return math.sqrt(max(load, 0.0) * cue_weight2 / (2.0 * d)) / KAPPA
