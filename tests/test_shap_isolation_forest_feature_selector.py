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


class _FakeEstimator:
    def __init__(self, n_features):
        self.n_features = n_features

    def score_samples(self, X):
        return -np.full(len(X), 1.0 / self.n_features)


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


def test_shap_isolation_forest_selector_can_validate_removal_with_cv(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(TreeExplainer=_FakeTreeExplainer))
    X = pd.DataFrame(
        {
            "least_important": np.arange(30, dtype=float),
            "middle": np.arange(30, dtype=float) * 2.0,
            "most_important": np.arange(30, dtype=float) * -1.0,
        }
    )

    selector = ShapIsolationForestFeatureSelector(
        random_state=5,
        max_iter=1,
        min_features=2,
        validation_cv=3,
    )
    monkeypatch.setattr(
        selector,
        "_fit_isolation_forest",
        lambda X_fold, *, iteration: _FakeEstimator(X_fold.shape[1]),
    )
    monkeypatch.setattr(selector, "_dispersion", lambda scores: float(np.mean(scores)))
    selector.fit(X)

    assert selector.selected_feature_names_ == ["middle", "most_important"]
    assert selector.history_[0]["validation_mode"] == "cross_validated"
    assert selector.history_[0]["validation_cv"] == 3
    assert len(selector.history_[0]["fold_improvements"]) == 3
    assert selector.history_[0]["improvement"] > 0


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


def test_tail_mean_median_gap_averages_the_top_score_tail():
    scores = np.arange(100, dtype=float)
    robust_selector = ShapIsolationForestFeatureSelector(
        dispersion_metric="tail_mean_median_gap",
        tail_quantile=0.95,
    )

    assert robust_selector._dispersion(scores) == np.mean([95, 96, 97, 98, 99]) - np.median(scores)
