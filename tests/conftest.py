import hashlib, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: E402,F401


class TopicEmbedder:
    """Synthetic stand-in for a real embedding model: every vector shares a large
    common component (as real models do), texts starting with the same 'topicN'
    word are similar, and each text has its own unique part."""
    def __init__(self, dim=768, common=0.6, topic=0.55):
        self.dim, self.common, self.topic = dim, common, topic
        self._c = self._vec("__common__")
    signature = "topic:768"
    def _vec(self, s):
        seed = int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)
    def embed(self, texts, kind="document"):
        out = []
        for t in texts:
            uniq = max(0.0, 1 - self.common**2 - self.topic**2) ** 0.5
            v = self.common * self._c + self.topic * self._vec(t.split()[0]) + uniq * self._vec(t)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


class StructEmbedder(TopicEmbedder):
    """First word is a path like 'd3/c17/t2'.  Texts sharing a domain are a bit
    similar, sharing a conversation more so, and the same turn (a paraphrase)
    most of all, which is roughly how real conversation embeddings cluster."""
    signature = "struct:768"
    def __init__(self, dim=768, common=0.6, levels=(0.3, 0.35, 0.5)):
        super().__init__(dim, common, 0.0)
        self.levels = levels
    def embed(self, texts, kind="document"):
        out = []
        for t in texts:
            parts = t.split()[0].split("/")
            v = self.common * self._c
            used = self.common ** 2
            for i, w in enumerate(self.levels[:len(parts)]):
                v = v + w * self._vec("/".join(parts[:i + 1]))
                used += w * w
            v = v + max(0.0, 1 - used) ** 0.5 * self._vec(t)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)
