import numpy as np
import pandas as pd

from arborration.feature_selection import _add_permutation_calibration


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
