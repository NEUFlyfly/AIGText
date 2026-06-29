from collections.abc import Mapping

from src.vision.coarse_classifier import CoarseClassifier, load_coarse_labels


class FakeClassifierBackend:
    def __init__(self, result: Mapping[str, object]) -> None:
        self._result: Mapping[str, object] = result

    def classify(self, image_bytes: bytes) -> Mapping[str, object]:
        assert image_bytes == b"fixture-image"
        return self._result


def test_classifier_output_schema() -> None:
    classifier = CoarseClassifier()

    result = classifier.classify(b"fixture-image")

    assert set(result) == {"coarse_category", "confidence", "status"}
    assert isinstance(result["coarse_category"], str)
    assert isinstance(result["confidence"], float)
    assert result["status"] in {"OK", "LOW_CONFIDENCE", "MODEL_NOT_READY", "UNKNOWN"}
    assert result["coarse_category"] in {*classifier.labels, "UNKNOWN"}


def test_labels_loaded_from_taxonomy_in_file_order() -> None:
    labels = load_coarse_labels("data/iot_knowledge/taxonomy.json")

    assert labels == ["智能传感器", "智能摄像头"]


def test_runtime_without_injected_model_is_model_not_ready() -> None:
    classifier = CoarseClassifier(labels=["智能传感器"])

    result = classifier.classify(b"fixture-image")

    assert result == {
        "coarse_category": "UNKNOWN",
        "confidence": 0.0,
        "status": "MODEL_NOT_READY",
    }


def test_injected_model_returns_ok_for_known_label_above_threshold() -> None:
    classifier = CoarseClassifier(
        labels=["智能传感器", "智能摄像头"],
        model=FakeClassifierBackend({"coarse_category": "智能传感器", "confidence": 0.92}),
        confidence_threshold=0.5,
    )

    result = classifier.classify(b"fixture-image")

    assert result == {
        "coarse_category": "智能传感器",
        "confidence": 0.92,
        "status": "OK",
    }


def test_low_confidence_returns_unknown_without_fabricating_category() -> None:
    classifier = CoarseClassifier(
        labels=["智能传感器"],
        model=FakeClassifierBackend({"coarse_category": "智能传感器", "confidence": 0.49}),
        confidence_threshold=0.5,
    )

    result = classifier.classify(b"fixture-image")

    assert result == {
        "coarse_category": "UNKNOWN",
        "confidence": 0.49,
        "status": "LOW_CONFIDENCE",
    }


def test_unknown_label_is_not_accepted_as_category() -> None:
    classifier = CoarseClassifier(
        labels=["智能传感器"],
        model=FakeClassifierBackend({"coarse_category": "不存在的类别", "confidence": 0.99}),
    )

    result = classifier.classify(b"fixture-image")

    assert result == {
        "coarse_category": "UNKNOWN",
        "confidence": 0.99,
        "status": "UNKNOWN",
    }
