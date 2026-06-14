"""Arborration library package."""

__all__ = [
    "AADForestGenerativeClassifier",
    "IsoForestGenerativeClassifier",
    "PineForestGenerativeClassifier",
    "ShapIsolationForestFeatureSelector",
]


def __getattr__(name):
    if name == "ShapIsolationForestFeatureSelector":
        from .feature_selection import ShapIsolationForestFeatureSelector

        return ShapIsolationForestFeatureSelector

    if name in {
        "AADForestGenerativeClassifier",
        "IsoForestGenerativeClassifier",
        "PineForestGenerativeClassifier",
    }:
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
