import numpy as np
import pandas as pd

from arborration.feature_selection import (
    _add_competitive_permutation_calibration,
    _add_permutation_calibration,
    _anchored_flip_usage_from_isotree_json,
    _isotree_structural_distance_matrix,
    _leaf_backtrack_usage_from_isotree_json,
    _make_competitive_augmented_column_weights,
    _select_cluster_winners,
)


class _FakeIsoTreeModel:
    def __init__(self, tree):
        self.tree = tree

    def to_json(self, as_str=False):
        return self.tree


def test_permutation_calibration_aligns_null_usage_by_feature_name():
    usage = pd.DataFrame(
        {
            "feature": ["b", "a"],
            "target_anchored_usage_for_selection": [5, 2],
        }
    )
    null_usage_matrix = np.array(
        [
            [1, 4],
            [3, 6],
        ]
    )

    calibrated, result = _add_permutation_calibration(
        usage,
        observed_column="target_anchored_usage_for_selection",
        null_usage_matrix=null_usage_matrix,
        feature_names=["a", "b"],
    )

    by_feature = calibrated.set_index("feature")
    assert by_feature.loc["a", "permutation_null_usage_mean"] == 2
    assert by_feature.loc["b", "permutation_null_usage_mean"] == 5
    assert by_feature.loc["a", "permutation_adjusted_usage_for_selection"] == 0
    assert by_feature.loc["b", "permutation_adjusted_usage_for_selection"] == 0
    assert result["n_null_fits"] == 2


def test_competitive_permutation_calibration_aligns_decoy_usage_by_feature_name():
    usage = pd.DataFrame(
        {
            "feature": ["b", "a"],
            "target_anchored_usage_for_selection": [7, 3],
        }
    )
    decoy_usage_matrix = np.array(
        [
            [1, 5],
            [3, 7],
        ]
    )

    calibrated, result = _add_competitive_permutation_calibration(
        usage,
        observed_column="target_anchored_usage_for_selection",
        decoy_selection_usage=decoy_usage_matrix.max(axis=0),
        decoy_usage_matrix=decoy_usage_matrix,
        feature_names=["a", "b"],
    )

    by_feature = calibrated.set_index("feature")
    assert by_feature.loc["a", "permutation_decoy_usage_for_selection"] == 3
    assert by_feature.loc["b", "permutation_decoy_usage_for_selection"] == 7
    assert by_feature.loc["a", "permutation_adjusted_usage_for_selection"] == 0
    assert by_feature.loc["b", "permutation_adjusted_usage_for_selection"] == 0
    assert result["mode"] == "competitive"
    assert result["n_competitive_refits"] == 2


def test_competitive_column_weights_keep_total_target_probability_with_real_lean():
    weights = _make_competitive_augmented_column_weights(
        ["x1", "x2", "x3"],
        ["target_a", "target_b"],
        ["decoy_a", "decoy_b"],
        target_draw_probability=0.25,
        real_target_weight=0.60,
    )

    x_weight = weights[:3].sum()
    real_weight = weights[3:5].sum()
    decoy_weight = weights[5:].sum()
    total = weights.sum()
    assert np.isclose((real_weight + decoy_weight) / total, 0.25)
    assert np.isclose(real_weight / (real_weight + decoy_weight), 0.60)
    assert real_weight > decoy_weight
    assert x_weight > real_weight + decoy_weight


def test_isotree_structural_substitution_distance_groups_alternate_split_features():
    X = pd.DataFrame(
        {
            "x1": [-2.0, -1.0, 1.0, 2.0],
            "x2": [-3.0, -2.0, 2.0, 3.0],
            "x3": [0.0, 1.0, 0.0, 1.0],
        }
    )
    tree_x1 = {
        "0": {
            "condition": [{"column": "x1", "coef": 1.0}],
            "threshold": 0.0,
            "left": "1",
            "right": "2",
        },
        "1": {},
        "2": {},
    }
    tree_x2 = {
        "0": {
            "condition": [{"column": "x2", "coef": 1.0}],
            "threshold": 0.0,
            "left": "1",
            "right": "2",
        },
        "1": {},
        "2": {},
    }

    distance, stats = _isotree_structural_distance_matrix(
        _FakeIsoTreeModel([tree_x1, tree_x2]),
        X=X,
        feature_names=["x1", "x2", "x3"],
        redundancy_mode="substitution",
        depth_decay=0.0,
        min_leaf_samples=1,
    )

    assert stats["n_extractable_splits"] == 2
    assert distance[0, 1] == 0.0
    assert distance[0, 2] == 1.0
    assert distance[1, 2] == 1.0


def test_select_cluster_winners_keeps_highest_scoring_feature_per_cluster():
    feature_scores = pd.DataFrame(
        {
            "feature": ["x2", "x1", "x3"],
            "score": [5.0, 2.0, 1.0],
        }
    )

    selected, removed, clusters, winners = _select_cluster_winners(
        ["x1", "x2", "x3"],
        np.array([1, 1, 2]),
        feature_scores,
        min_features_to_keep=1,
    )

    assert selected == ["x2", "x3"]
    assert removed == ["x1"]
    assert clusters == {1: ["x1", "x2"], 2: ["x3"]}
    assert winners == {1: "x2", 2: "x3"}


def test_target_anchored_flip_usage_counts_counterfactual_routing_changes():
    X_aug = pd.DataFrame(
        {
            "x1": [2.0, -2.0, 0.2],
            "x2": [3.0, 3.0, 3.0],
            "__target__z": [0.0, 0.0, 0.0],
        }
    )
    tree = {
        "features": ["x1", "__target__z"],
        "coefs": [1.0, 1.0],
        "threshold": 0.0,
    }

    counts = _anchored_flip_usage_from_isotree_json(
        _FakeIsoTreeModel(tree),
        X_aug=X_aug,
        x_feature_names=["x1", "x2"],
        target_feature_names=["__target__z"],
    )

    assert counts.tolist() == [2, 0]


def test_target_anchored_flip_usage_routes_flat_isotree_json():
    X_aug = pd.DataFrame(
        {
            "x1": [2.0, -2.0, 0.2],
            "x2": [3.0, 3.0, 3.0],
            "__target__z": [0.0, 0.0, 0.0],
        }
    )
    tree = {
        "0": {
            "condition": [
                {"column": "x1", "coef": 1.0, "centering": 0.0},
                {"column": "__target__z", "coef": 1.0, "centering": 0.0},
            ],
            "threshold": 0.0,
            "left": "1",
            "right": "2",
        },
        "1": {},
        "2": {},
    }

    counts = _anchored_flip_usage_from_isotree_json(
        _FakeIsoTreeModel(tree),
        X_aug=X_aug,
        x_feature_names=["x1", "x2"],
        target_feature_names=["__target__z"],
    )

    assert counts.tolist() == [2, 0]


def test_target_anchored_flip_usage_routes_isotree_export_json_keys():
    X_aug = pd.DataFrame(
        {
            "x1": [2.0, -2.0, 0.2],
            "x2": [3.0, 3.0, 3.0],
            "__target__z": [0.0, 0.0, 0.0],
        }
    )
    tree = {
        "0": {
            "terminal": "no",
            "node_when_condition_is_met": 1,
            "node_when_condition_is_not_met": 2,
            "combination": [
                {
                    "column": "x1",
                    "column_type": "numeric",
                    "coefficient": 1.0,
                    "centering": 0.0,
                },
                {
                    "column": "__target__z",
                    "column_type": "numeric",
                    "coefficient": 1.0,
                    "centering": 0.0,
                },
            ],
            "condition": "<=",
            "value": 0.0,
        },
        "1": {"terminal": "yes", "score": 1.0, "leaf": 0},
        "2": {"terminal": "yes", "score": 1.0, "leaf": 1},
    }

    counts = _anchored_flip_usage_from_isotree_json(
        _FakeIsoTreeModel(tree),
        X_aug=X_aug,
        x_feature_names=["x1", "x2"],
        target_feature_names=["__target__z"],
    )

    assert counts.tolist() == [2, 0]


def test_leaf_backtrack_usage_credits_paths_to_low_variance_target_leaves():
    X_aug = pd.DataFrame(
        {
            "x1": [-2.0, -1.0, 1.0, 2.0],
            "x2": [0.0, 0.0, 0.0, 0.0],
            "__target__z": [0.0, 0.0, 0.0, 10.0],
        }
    )
    tree = {
        "features": ["x1", "__target__z"],
        "coefs": [1.0, 0.0],
        "threshold": 0.0,
        "left": {},
        "right": {},
    }

    scores = _leaf_backtrack_usage_from_isotree_json(
        _FakeIsoTreeModel(tree),
        X_aug=X_aug,
        x_feature_names=["x1", "x2"],
        target_feature_names=["__target__z"],
        signal_target_feature_names=["__target__z"],
        leaf_signal_quantile=0.5,
        leaf_min_samples=1,
    )

    assert scores.tolist() == [2.0, 0.0]


def test_leaf_backtrack_usage_routes_flat_isotree_json():
    X_aug = pd.DataFrame(
        {
            "x1": [-2.0, -1.0, 1.0, 2.0],
            "x2": [0.0, 0.0, 0.0, 0.0],
            "__target__z": [0.0, 0.0, 0.0, 10.0],
        }
    )
    tree = {
        "0": {
            "condition": [
                {"column": "x1", "coef": 1.0, "centering": 0.0},
                {"column": "__target__z", "coef": 0.0, "centering": 0.0},
            ],
            "threshold": 0.0,
            "left": "1",
            "right": "2",
        },
        "1": {},
        "2": {},
    }

    scores = _leaf_backtrack_usage_from_isotree_json(
        _FakeIsoTreeModel(tree),
        X_aug=X_aug,
        x_feature_names=["x1", "x2"],
        target_feature_names=["__target__z"],
        signal_target_feature_names=["__target__z"],
        leaf_signal_quantile=0.5,
        leaf_min_samples=1,
    )

    assert scores.tolist() == [2.0, 0.0]


def test_leaf_backtrack_usage_routes_isotree_export_json_keys():
    X_aug = pd.DataFrame(
        {
            "x1": [-2.0, -1.0, 1.0, 2.0],
            "x2": [0.0, 0.0, 0.0, 0.0],
            "__target__z": [0.0, 0.0, 0.0, 10.0],
        }
    )
    tree = {
        "0": {
            "terminal": "no",
            "node_when_condition_is_met": 1,
            "node_when_condition_is_not_met": 2,
            "combination": [
                {
                    "column": "x1",
                    "column_type": "numeric",
                    "coefficient": 1.0,
                    "centering": 0.0,
                },
                {
                    "column": "__target__z",
                    "column_type": "numeric",
                    "coefficient": 0.0,
                    "centering": 0.0,
                },
            ],
            "condition": "<=",
            "value": 0.0,
        },
        "1": {"terminal": "yes", "score": 1.0, "leaf": 0},
        "2": {"terminal": "yes", "score": 1.0, "leaf": 1},
    }

    scores = _leaf_backtrack_usage_from_isotree_json(
        _FakeIsoTreeModel(tree),
        X_aug=X_aug,
        x_feature_names=["x1", "x2"],
        target_feature_names=["__target__z"],
        signal_target_feature_names=["__target__z"],
        leaf_signal_quantile=0.5,
        leaf_min_samples=1,
    )

    assert scores.tolist() == [2.0, 0.0]
