"""Frozen d1-2B plus experimental one-pass list decoding."""
from .decoder import decode
from .snapshot import Snapshot

__version__ = "0.1.0"
__all__ = ["Snapshot", "decode", "ListExtractor"]


def __getattr__(name):
    if name == "ListExtractor":
        from .backend import ListExtractor
        return ListExtractor
    raise AttributeError(name)

