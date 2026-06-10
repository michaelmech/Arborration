import numpy as np
import pandas as pd

from arborration.feature_selection import (
    _add_competitive_permutation_calibration,
    _add_permutation_calibration,
    _make_competitive_augmented_column_weights,
)


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
