import sys
import types

import numpy as np
import pandas as pd

from arborration.feature_selection import ShapIsolationForestFeatureSelector


class _FakeTreeExplainer:
    def __init__(self, estimator):
        self.estimator = estimator

    def shap_values(self, X):
        values = np.tile(np.arange(X.shape[1], dtype=float), (len(X), 1))
        return values


def test_shap_isolation_forest_selector_records_state_and_transforms(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(TreeExplainer=_FakeTreeExplainer))
    X = pd.DataFrame(
        {
            "least_important": np.r_[np.zeros(30), np.ones(5)],
            "middle": np.linspace(0.0, 1.0, 35),
            "most_important": np.r_[np.zeros(30), np.full(5, 10.0)],
        }
    )

    selector = ShapIsolationForestFeatureSelector(
        n_estimators=20,
        random_state=7,
        drop_fraction=0.34,
        max_drop_per_iter=1,
        min_features=2,
        max_iter=1,
        min_improvement=0.0,
    )
    metrics = iter([0.0, 1.0])
    monkeypatch.setattr(selector, "_dispersion", lambda scores: next(metrics))
    selector.fit(X)

    assert selector.n_features_in_ == 3
    assert selector.selected_feature_names_ == ["middle", "most_important"]
    assert selector.selected_features_ == ["middle", "most_important"]
    assert selector.get_support().tolist() == [False, True, True]
    assert selector.get_support(indices=True).tolist() == [1, 2]
    assert selector.transform(X).columns.tolist() == ["middle", "most_important"]
    assert selector.final_estimator_ is selector.estimator_
    assert isinstance(selector.final_metric_, float)
    assert selector.shap_importances_["feature"].tolist() == ["most_important", "middle"]
    assert len(selector.history_) == 1
    assert selector.history_[0]["candidate_removed_features"] == ["least_important"]


def test_shap_isolation_forest_selector_stops_on_rejected_removal(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(TreeExplainer=_FakeTreeExplainer))
    X = pd.DataFrame(
        {
            "a": np.arange(20, dtype=float),
            "b": np.arange(20, dtype=float) * 2.0,
            "c": np.arange(20, dtype=float) * -1.0,
        }
    )

    selector = ShapIsolationForestFeatureSelector(
        n_estimators=10,
        random_state=11,
        max_iter=3,
        min_improvement=1e9,
    )
    metrics = iter([1.0, 1.0])
    monkeypatch.setattr(selector, "_dispersion", lambda scores: next(metrics))
    selector.fit(X)

    assert selector.selected_feature_names_ == ["a", "b", "c"]
    assert len(selector.history_) == 1
    assert selector.history_[0]["accepted"] is False
