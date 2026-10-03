"""Holonomic memory provider for Hermes Agent.

Associations between memories are stored as superposed interference patterns
("plates"); see engine.py.  Hermes loads this package from
$HERMES_HOME/plugins/holonomic/ and calls register().

Nothing here imports numpy at load time.  Hermes lists a provider in
`hermes memory setup` only if its package imports cleanly, and installs the
provider's dependencies after it has been chosen, so the package has to load
before numpy is there.
"""

__version__ = "0.3.1"
__all__ = ["HolonomicMemory", "Recollection", "OllamaEmbedder", "HashEmbedder", "EmbeddingError", "register"]

_LAZY = {"HolonomicMemory": "engine", "Recollection": "engine",
         "OllamaEmbedder": "embed", "HashEmbedder": "embed", "EmbeddingError": "embed"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register(ctx) -> None:
    """Entry point used by Hermes: ctx.register_memory_provider(...)."""
    from .provider import HolonomicMemoryProvider      # imports Hermes modules, so only inside Hermes
    ctx.register_memory_provider(HolonomicMemoryProvider())
