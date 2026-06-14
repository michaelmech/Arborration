from dataclasses import dataclass
from math import ceil

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_array, check_is_fitted


@dataclass(frozen=True)
class FeatureSelectionStep:
    iteration: int
    feature_names: tuple[str, ...]
    dropped_features: tuple[str, ...]
    metric: float
    trial_metric: float | None
    improvement: float | None
    accepted: bool


class ShapIsolationForestFeatureSelector(TransformerMixin, BaseEstimator):
    """Iteratively remove low-SHAP features when anomaly-score dispersion improves."""

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        drop_fraction=0.1,
        max_drop_per_iter=1,
        min_features=1,
        max_iter=None,
        min_improvement=0.0,
        metric="tail_gap",
        metric_quantiles=(99.0, 50.0),
        shap_sample_size=512,
        background_size=256,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.drop_fraction = drop_fraction
        self.max_drop_per_iter = max_drop_per_iter
        self.min_features = min_features
        self.max_iter = max_iter
        self.min_improvement = min_improvement
        self.metric = metric
        self.metric_quantiles = metric_quantiles
        self.shap_sample_size = shap_sample_size
        self.background_size = background_size

    def fit(self, X, y=None):
        del y
        X_array, feature_names = self._validate_X_with_names(X)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = X_array.shape[1]

        if not 1 <= self.min_features <= self.n_features_in_:
            raise ValueError("min_features must be between 1 and the number of input features.")
        if not 0 < self.drop_fraction <= 1:
            raise ValueError("drop_fraction must be in the interval (0, 1].")
        if self.max_drop_per_iter < 1:
            raise ValueError("max_drop_per_iter must be at least 1.")

        current_features = list(range(self.n_features_in_))
        history = []
        iteration = 0

        while True:
            model = self._fit_isolation_forest(X_array[:, current_features], iteration)
            scores = self._anomaly_scores(model, X_array[:, current_features])
            metric = self._dispersion_metric(scores)
            importances = self._shap_importances(model, X_array[:, current_features], iteration)

            self.estimator_ = model
            self.score_metric_ = metric
            self.feature_importances_ = self._expand_importances(current_features, importances)
            self.selected_features_ = tuple(self.feature_names_in_[current_features])

            if self._should_stop(iteration, current_features):
                history.append(
                    FeatureSelectionStep(
                        iteration=iteration,
                        feature_names=tuple(self.feature_names_in_[current_features]),
                        dropped_features=(),
                        metric=metric,
                        trial_metric=None,
                        improvement=None,
                        accepted=False,
                    )
                )
                break

            candidate_positions = self._candidate_positions(importances, len(current_features))
            drop_features = [current_features[pos] for pos in candidate_positions]
            trial_features = [idx for idx in current_features if idx not in drop_features]

            trial_model = self._fit_isolation_forest(X_array[:, trial_features], iteration + 1)
            trial_scores = self._anomaly_scores(trial_model, X_array[:, trial_features])
            trial_metric = self._dispersion_metric(trial_scores)
            improvement = trial_metric - metric
            accepted = improvement > self.min_improvement

            history.append(
                FeatureSelectionStep(
                    iteration=iteration,
                    feature_names=tuple(self.feature_names_in_[current_features]),
                    dropped_features=tuple(self.feature_names_in_[drop_features]),
                    metric=metric,
                    trial_metric=trial_metric,
                    improvement=improvement,
                    accepted=accepted,
                )
            )

            if not accepted:
                break

            current_features = trial_features
            iteration += 1

        self.history_ = history
        self.support_ = np.isin(np.arange(self.n_features_in_), current_features)
        self.selected_indices_ = np.asarray(current_features, dtype=int)
        return self

    def transform(self, X):
        check_is_fitted(self, ["support_", "selected_indices_"])
        if hasattr(X, "iloc"):
            return X.iloc[:, self.selected_indices_]
        X_array = check_array(X, accept_sparse=False, dtype=np.float64)
        return X_array[:, self.selected_indices_]

    def fit_transform(self, X, y=None, **fit_params):
        del fit_params
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        if indices:
            return self.selected_indices_.copy()
        return self.support_.copy()

    def _validate_X_with_names(self, X):
        if hasattr(X, "columns"):
            feature_names = [str(col) for col in X.columns]
        else:
            feature_names = [f"x{i}" for i in range(np.asarray(X).shape[1])]
        X_array = check_array(X, accept_sparse=False, dtype=np.float64)
        return X_array, feature_names

    def _fit_isolation_forest(self, X, iteration):
        return IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self._iteration_random_state(iteration),
            verbose=self.verbose,
            warm_start=self.warm_start,
        ).fit(X)

    def _iteration_random_state(self, iteration):
        if self.random_state is None:
            return None
        return int(np.random.RandomState(self.random_state).randint(0, 2**31 - 1, size=iteration + 1)[-1])

    def _anomaly_scores(self, model, X):
        return -model.score_samples(X)

    def _dispersion_metric(self, scores):
        if self.metric == "std":
            return float(np.std(scores))
        if self.metric == "var":
            return float(np.var(scores))
        if self.metric == "tail_gap":
            high, low = self.metric_quantiles
            return float(np.percentile(scores, high) - np.percentile(scores, low))
        raise ValueError("metric must be one of {'tail_gap', 'std', 'var'}.")

    def _shap_importances(self, model, X, iteration):
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "ShapIsolationForestFeatureSelector requires the optional dependency 'shap'. "
                "Install it with `python -m pip install shap`."
            ) from exc

        X_explain = self._sample_rows(X, self.shap_sample_size, iteration)
        background = self._sample_rows(X, self.background_size, iteration + 17)
        explainer = shap.TreeExplainer(model, data=background)
        shap_values = explainer.shap_values(X_explain)
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[..., 0]
        return np.abs(values).mean(axis=0)

    def _sample_rows(self, X, max_rows, iteration):
        if max_rows is None or X.shape[0] <= max_rows:
            return X
        rng = np.random.RandomState(self._iteration_random_state(iteration))
        rows = rng.choice(X.shape[0], size=max_rows, replace=False)
        return X[rows]

    def _candidate_positions(self, importances, n_current_features):
        n_drop = min(
            self.max_drop_per_iter,
            max(1, ceil(n_current_features * self.drop_fraction)),
            n_current_features - self.min_features,
        )
        return np.argsort(importances)[:n_drop].tolist()

    def _should_stop(self, iteration, current_features):
        if len(current_features) <= self.min_features:
            return True
        if self.max_iter is not None and iteration >= self.max_iter:
            return True
        return False

    def _expand_importances(self, current_features, importances):
        expanded = np.zeros(self.n_features_in_, dtype=float)
        expanded[current_features] = importances
        return expanded
