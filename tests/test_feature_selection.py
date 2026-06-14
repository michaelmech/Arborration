import numpy as np
import pandas as pd

from arborration.feature_selection import ShapIsolationForestFeatureSelector


class _DummyModel:
    pass


def test_selector_accepts_low_shap_feature_drop_when_metric_improves(monkeypatch):
    X = pd.DataFrame(
        {
            "signal": [0.0, 0.1, 0.2, 6.0, 7.0],
            "noise": [4.0, 4.1, 4.2, 4.3, 4.4],
            "rare": [1.0, 1.0, 1.0, 8.0, 9.0],
        }
    )
    selector = ShapIsolationForestFeatureSelector(
        random_state=0,
        max_iter=1,
        drop_fraction=1 / 3,
        max_drop_per_iter=1,
    )

    monkeypatch.setattr(selector, "_fit_isolation_forest", lambda X, iteration: _DummyModel())
    monkeypatch.setattr(selector, "_anomaly_scores", lambda model, X: np.array([3 - X.shape[1]], dtype=float))
    monkeypatch.setattr(selector, "_dispersion_metric", lambda scores: float(scores[0]))
    monkeypatch.setattr(selector, "_shap_importances", lambda model, X, iteration: np.arange(X.shape[1]))

    selector.fit(X)

    assert selector.selected_features_ == ("noise", "rare")
    assert selector.history_[0].dropped_features == ("signal",)
    assert selector.history_[0].accepted is True
    assert selector.get_support().tolist() == [False, True, True]
    pd.testing.assert_frame_equal(selector.transform(X), X[["noise", "rare"]])


def test_selector_rejects_drop_when_metric_does_not_improve(monkeypatch):
    X = np.arange(20, dtype=float).reshape(5, 4)
    selector = ShapIsolationForestFeatureSelector(random_state=0, max_drop_per_iter=2)

    monkeypatch.setattr(selector, "_fit_isolation_forest", lambda X, iteration: _DummyModel())
    monkeypatch.setattr(selector, "_anomaly_scores", lambda model, X: np.array([1.0]))
    monkeypatch.setattr(selector, "_dispersion_metric", lambda scores: float(scores[0]))
    monkeypatch.setattr(selector, "_shap_importances", lambda model, X, iteration: np.arange(X.shape[1]))

    selector.fit(X)

    assert selector.selected_features_ == ("x0", "x1", "x2", "x3")
    assert selector.history_[0].accepted is False
    assert selector.get_support(indices=True).tolist() == [0, 1, 2, 3]
