"""Arborration library package."""

__all__ = [
    "AADForestGenerativeClassifier",
    "IsoForestGenerativeClassifier",
    "PineForestGenerativeClassifier",
    "select_features_by_class_contrast_isotree_json_usage",
    "select_features_by_target_weighted_isotree_json_usage",
    "select_features_for_task",
]


def __getattr__(name):
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
    if name in {
        "select_features_by_class_contrast_isotree_json_usage",
        "select_features_by_target_weighted_isotree_json_usage",
        "select_features_for_task",
    }:
        from .feature_selection import (
            select_features_by_class_contrast_isotree_json_usage,
            select_features_by_target_weighted_isotree_json_usage,
            select_features_for_task,
        )

        exports = {
            "select_features_by_class_contrast_isotree_json_usage": select_features_by_class_contrast_isotree_json_usage,
            "select_features_by_target_weighted_isotree_json_usage": select_features_by_target_weighted_isotree_json_usage,
            "select_features_for_task": select_features_for_task,
        }
        return exports[name]
    raise AttributeError(f"module 'arborration' has no attribute {name!r}")
