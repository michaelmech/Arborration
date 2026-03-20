import importlib

import numpy as np
from coniferest.aadforest import AADForest
from coniferest.label import Label
from coniferest.pineforest import PineForest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def _get_isotree_isolationforest():
    return importlib.import_module("isotree").IsolationForest


class IsoForestGenerativeClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    _eps = 1e-12

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        ndim=1,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_depth=None,
        calibration="discriminator",
        global_scale=1.0,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.ndim = ndim
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.max_depth = max_depth
        self.calibration = calibration
        self.global_scale = global_scale

    def _make_iforest(self, random_state):
        return _get_isotree_isolationforest()(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=random_state,
            bootstrap=self.bootstrap,
            ndim=self.ndim,
            missing_action="fail",
            scoring_metric="depth",
            penalize_range=True,
        )

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need at least 2 classes.")

        self.n_classes_ = n_classes
        self.class_priors_ = np.bincount(y_enc) / float(len(y))
        self.log_class_priors_ = np.log(self.class_priors_ + self._eps)

        self.iforests_ = []
        self.calibrators_ = []

        rng = np.random.RandomState(self.random_state)

        for k in range(n_classes):
            Xk = X[y_enc == k]
            rs_k = None if self.random_state is None else rng.randint(0, 2**31 - 1)
            iforest_k = self._make_iforest(random_state=rs_k)
            iforest_k.fit(Xk)
            self.iforests_.append(iforest_k)

            scores_k = -iforest_k.decision_function(Xk)

            if self.calibration == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(len(Xk), 1000),
                )
                qt.fit(scores_k.reshape(-1, 1))
                self.calibrators_.append(qt)
            elif self.calibration in {"zscore", "discriminator"}:
                mu_k = scores_k.mean()
                std_k = scores_k.std()
                if std_k <= 0:
                    std_k = 1.0
                self.calibrators_.append((mu_k, std_k))
            else:
                raise ValueError("calibration must be one of {'quantile', 'zscore', 'discriminator'}")

        if self.calibration == "discriminator":
            S_train = self._get_raw_scores(X, apply_zscore=True)
            self.discriminator_ = LogisticRegression(class_weight="balanced")
            self.discriminator_.fit(S_train, y_enc)

        self.n_features_in_ = X.shape[1]
        return self

    def _get_raw_scores(self, X, apply_zscore=False):
        n_samples = X.shape[0]
        S = np.zeros((n_samples, self.n_classes_), dtype=float)

        for k, iforest_k in enumerate(self.iforests_):
            s_k = -iforest_k.decision_function(X)

            if self.calibration == "quantile":
                s_k = self.calibrators_[k].transform(s_k.reshape(-1, 1)).flatten()
            elif apply_zscore or self.calibration == "zscore":
                mu_k, std_k = self.calibrators_[k]
                s_k = (s_k - mu_k) / std_k

            S[:, k] = s_k
        return S

    def predict_proba(self, X):
        check_is_fitted(self, ["iforests_", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if self.calibration == "discriminator":
            S = self._get_raw_scores(X, apply_zscore=True)
            return self.discriminator_.predict_proba(S)

        S = self._get_raw_scores(X)
        L = self.global_scale * S + self.log_class_priors_

        max_L = np.max(L, axis=1, keepdims=True)
        exp_L = np.exp(L - max_L)
        Z = exp_L.sum(axis=1, keepdims=True)
        return exp_L / Z

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return self.classes_[idx]


class PineForestGenerativeClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    _eps = 1e-12

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_depth=None,
        calibration="discriminator",
        global_scale=1.0,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.max_depth = max_depth
        self.calibration = calibration
        self.global_scale = global_scale

    def _make_pineforest(self, random_state):
        pineforest = PineForest(
            n_trees=self.n_estimators,
            n_subsamples=self.max_samples,
            max_depth=self.max_depth,
            n_jobs=-1 if self.n_jobs is None else self.n_jobs,
            random_seed=random_state,
        )
        if self.bootstrap:
            pineforest.bootstrap_samples = True
        return pineforest

    def _score_pine_normality(self, model, X):
        return model.score_samples(X)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need at least 2 classes.")

        self.n_classes_ = n_classes
        self.class_priors_ = np.bincount(y_enc) / float(len(y))
        self.log_class_priors_ = np.log(self.class_priors_ + self._eps)

        self.pineforests_ = []
        self.calibrators_ = []

        rng = np.random.RandomState(self.random_state)

        for k in range(n_classes):
            Xk = X[y_enc == k]
            rs_k = None if self.random_state is None else rng.randint(0, 2**31 - 1)
            pineforest_k = self._make_pineforest(random_state=rs_k)
            pineforest_k.fit(Xk)
            self.pineforests_.append(pineforest_k)

            scores_k = self._score_pine_normality(pineforest_k, Xk)

            if self.calibration == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(len(Xk), 1000),
                )
                qt.fit(scores_k.reshape(-1, 1))
                self.calibrators_.append(qt)
            elif self.calibration in {"zscore", "discriminator"}:
                mu_k = scores_k.mean()
                std_k = scores_k.std()
                if std_k <= 0:
                    std_k = 1.0
                self.calibrators_.append((mu_k, std_k))
            else:
                raise ValueError("calibration must be one of {'quantile', 'zscore', 'discriminator'}")

        if self.calibration == "discriminator":
            S_train = self._get_raw_scores(X, apply_zscore=True)
            self.discriminator_ = LogisticRegression(class_weight="balanced")
            self.discriminator_.fit(S_train, y_enc)

        self.n_features_in_ = X.shape[1]
        return self

    def _get_raw_scores(self, X, apply_zscore=False):
        n_samples = X.shape[0]
        S = np.zeros((n_samples, self.n_classes_), dtype=float)

        for k, pineforest_k in enumerate(self.pineforests_):
            s_k = self._score_pine_normality(pineforest_k, X)

            if self.calibration == "quantile":
                s_k = self.calibrators_[k].transform(s_k.reshape(-1, 1)).flatten()
            elif apply_zscore or self.calibration == "zscore":
                mu_k, std_k = self.calibrators_[k]
                s_k = (s_k - mu_k) / std_k

            S[:, k] = s_k
        return S

    def predict_proba(self, X):
        check_is_fitted(self, ["pineforests_", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if self.calibration == "discriminator":
            S = self._get_raw_scores(X, apply_zscore=True)
            return self.discriminator_.predict_proba(S)

        S = self._get_raw_scores(X)
        L = self.global_scale * S + self.log_class_priors_

        max_L = np.max(L, axis=1, keepdims=True)
        exp_L = np.exp(L - max_L)
        Z = exp_L.sum(axis=1, keepdims=True)
        return exp_L / Z

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return self.classes_[idx]


class AADForestGenerativeClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"
    _eps = 1e-12

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        max_depth=None,
        calibration="discriminator",
        global_scale=1.0,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.max_depth = max_depth
        self.calibration = calibration
        self.global_scale = global_scale

    def _make_aadforest(self, random_state):
        return AADForest(
            n_trees=self.n_estimators,
            n_subsamples=self.max_samples,
            max_depth=self.max_depth,
            n_jobs=-1 if self.n_jobs is None else self.n_jobs,
            random_seed=random_state,
        )

    def _score_aad_class_normality(self, model, X):
        return model.score_samples(X)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need at least 2 classes.")

        self.n_classes_ = n_classes
        self.class_priors_ = np.bincount(y_enc) / float(len(y))
        self.log_class_priors_ = np.log(self.class_priors_ + self._eps)

        self.aadforests_ = []
        self.calibrators_ = []

        rng = np.random.RandomState(self.random_state)

        for k in range(n_classes):
            rs_k = None if self.random_state is None else rng.randint(0, 2**31 - 1)
            aadforest_k = self._make_aadforest(random_state=rs_k)

            labels_k = np.where(y_enc == k, Label.REGULAR, Label.ANOMALY)
            aadforest_k.fit(X, labels=labels_k)
            self.aadforests_.append(aadforest_k)

            scores_k = self._score_aad_class_normality(aadforest_k, X)

            if self.calibration == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(len(X), 1000),
                )
                qt.fit(scores_k.reshape(-1, 1))
                self.calibrators_.append(qt)
            elif self.calibration in {"zscore", "discriminator"}:
                mu_k = scores_k.mean()
                std_k = scores_k.std()
                if std_k <= 0:
                    std_k = 1.0
                self.calibrators_.append((mu_k, std_k))
            else:
                raise ValueError("calibration must be one of {'quantile', 'zscore', 'discriminator'}")

        if self.calibration == "discriminator":
            S_train = self._get_raw_scores(X, apply_zscore=True)
            self.discriminator_ = LogisticRegression(class_weight="balanced")
            self.discriminator_.fit(S_train, y_enc)

        self.n_features_in_ = X.shape[1]
        return self

    def _get_raw_scores(self, X, apply_zscore=False):
        n_samples = X.shape[0]
        S = np.zeros((n_samples, self.n_classes_), dtype=float)

        for k, aadforest_k in enumerate(self.aadforests_):
            s_k = self._score_aad_class_normality(aadforest_k, X)

            if self.calibration == "quantile":
                s_k = self.calibrators_[k].transform(s_k.reshape(-1, 1)).flatten()
            elif apply_zscore or self.calibration == "zscore":
                mu_k, std_k = self.calibrators_[k]
                s_k = (s_k - mu_k) / std_k

            S[:, k] = s_k
        return S

    def predict_proba(self, X):
        check_is_fitted(self, ["aadforests_", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if self.calibration == "discriminator":
            S = self._get_raw_scores(X, apply_zscore=True)
            return self.discriminator_.predict_proba(S)

        S = self._get_raw_scores(X)
        L = self.global_scale * S + self.log_class_priors_

        max_L = np.max(L, axis=1, keepdims=True)
        exp_L = np.exp(L - max_L)
        Z = exp_L.sum(axis=1, keepdims=True)
        return exp_L / Z

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return self.classes_[idx]
