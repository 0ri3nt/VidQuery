from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import AudioExtractor
    from .preprocessor import VideoPreprocessor
    from .visual import VisualExtractor

__all__ = ["VideoPreprocessor", "VisualExtractor", "AudioExtractor"]


def __getattr__(name: str):
    modules = {
        "VideoPreprocessor": ".preprocessor",
        "VisualExtractor": ".visual",
        "AudioExtractor": ".audio",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name, __name__), name)
