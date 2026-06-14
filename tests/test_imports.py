import arborration


def test_package_exposes_classifier_names_in_public_api():
    assert arborration.__all__ == [
        "AADForestGenerativeClassifier",
        "IsoForestGenerativeClassifier",
        "PineForestGenerativeClassifier",
        "ShapIsolationForestFeatureSelector",
        "select_features_by_class_contrast_isotree_json_usage",
        "select_features_by_isotree_structural_redundancy",
        "select_features_by_target_weighted_isotree_json_usage",
        "select_features_for_task",
    ]


def test_package_lazily_exposes_feature_selector_names():
    assert callable(arborration.select_features_for_task)
    assert callable(arborration.ShapIsolationForestFeatureSelector)
    assert callable(arborration.select_features_by_isotree_structural_redundancy)
