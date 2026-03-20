"""Arborration library package."""

__all__ = [
    "AADForestGenerativeClassifier",
    "IsoForestGenerativeClassifier",
    "PineForestGenerativeClassifier",
]


def __getattr__(name):
    if name in __all__:
        from .classifiers import (
            AADForestGenerativeClassifier,
            IsoForestGenerativeClassifier,
            PineForestGenerativeClassifier,
        )

        exports = {
            "AADForestGenerativeClassifier": AADForestGenerativeClassifier,
            "IsoForestGenerativeClassifier": IsoForestGenerativeClassifier,
            "PineForestGenerativeClassifier": PineForestGenerativeClassifier,
        }
        return exports[name]
    raise AttributeError(f"module 'arborration' has no attribute {name!r}")
