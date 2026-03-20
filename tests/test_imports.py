import arborration


def test_package_exposes_classifier_names_in_public_api():
    assert arborration.__all__ == [
        "AADForestGenerativeClassifier",
        "IsoForestGenerativeClassifier",
        "PineForestGenerativeClassifier",
    ]
