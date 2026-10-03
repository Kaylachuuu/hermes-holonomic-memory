"""Holonomic memory provider for Hermes Agent.

Associations between memories are stored as superposed interference patterns
("plates"); see engine.py.  Hermes loads this package from
$HERMES_HOME/plugins/holonomic/ and calls register().
"""

from .embed import EmbeddingError, HashEmbedder, OllamaEmbedder
from .engine import HolonomicMemory, Recollection

__version__ = "0.2.0"
__all__ = ["HolonomicMemory", "Recollection", "OllamaEmbedder", "HashEmbedder", "EmbeddingError", "register"]


def register(ctx) -> None:
    """Entry point used by Hermes: ctx.register_memory_provider(...)."""
    from .provider import HolonomicMemoryProvider      # imports Hermes modules, so only inside Hermes
    ctx.register_memory_provider(HolonomicMemoryProvider())
