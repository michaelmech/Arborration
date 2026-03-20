from sklearn.datasets import make_classification

from . import PineForestGenerativeClassifier


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    clf = PineForestGenerativeClassifier(
        n_estimators=25,
        max_samples=32,
        calibration="zscore",
        random_state=0,
    )
    clf.fit(X, y)
    print(clf.predict_proba(X[:5]))
    print(clf.predict(X[:5]))
