from typing import cast

from src.rag.visual_retriever import VisualRetriever
from src.rag.visual_store import ImageMetadata, InMemoryVisualStore
from src.vision.coarse_classifier import CoarseClassification


Candidate = dict[str, object]


class FakeQueryEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector: list[float] = vector
        self.inputs: list[object] = []

    def embed_image(self, image_input: object) -> list[float]:
        self.inputs.append(image_input)
        return self.vector


class FakeCoarseClassifier:
    def __init__(self, result: CoarseClassification) -> None:
        self.result: CoarseClassification = result
        self.inputs: list[bytes] = []

    def classify(self, image_bytes: bytes) -> CoarseClassification:
        self.inputs.append(image_bytes)
        return self.result


def test_query_image_returns_top_k_doc_candidates() -> None:
    store = _seed_visual_store()
    embedder = FakeQueryEmbedder([1.0, 0.0])
    classifier = FakeCoarseClassifier({
        "coarse_category": "智能传感器",
        "confidence": 0.91,
        "status": "OK",
    })
    retriever = VisualRetriever(
        store=store,
        embedder=embedder,
        classifier=classifier,
        top_k=2,
        min_score=0.5,
    )

    result = retriever.retrieve(b"temp-sensor-query")

    candidates = cast(list[Candidate], result["visual_candidates"])
    assert result["status"] == "OK"
    assert result["coarse_category"] == "智能传感器"
    assert result["coarse_status"] == "OK"
    assert result["matched_image_count"] == 3
    assert embedder.inputs == [b"temp-sensor-query"]
    assert classifier.inputs == [b"temp-sensor-query"]

    assert len(candidates) == 2
    assert candidates[0] == {
        "doc_id": "fixture_temp_sensor",
        "sub_category": "温湿度传感器",
        "coarse_category": "智能传感器",
        "score": 1.0,
        "evidence_image_id": "temp-image-a",
        "evidence_image_path": "tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/images/a.png",
        "matched_image_count": 2,
        "status": "OK",
    }
    assert candidates[1]["doc_id"] == "fixture_pir_sensor"
    assert {candidate["doc_id"] for candidate in candidates} == {
        "fixture_temp_sensor",
        "fixture_pir_sensor",
    }
    assert all(candidate["coarse_category"] == "智能传感器" for candidate in candidates)


def test_ok_classifier_filters_visual_search_by_coarse_category() -> None:
    store = _seed_visual_store()
    retriever = VisualRetriever(
        store=store,
        embedder=FakeQueryEmbedder([0.0, 1.0]),
        classifier=FakeCoarseClassifier({
            "coarse_category": "智能传感器",
            "confidence": 0.9,
            "status": "OK",
        }),
        top_k=5,
        min_score=0.0,
    )

    result = retriever.retrieve(b"camera-like-query")
    candidates = cast(list[Candidate], result["visual_candidates"])
    assert result["details"] == {"coarse_filter_applied": True, "candidate_count": 2}
    assert {candidate["doc_id"] for candidate in candidates} == {
        "fixture_temp_sensor",
        "fixture_pir_sensor",
    }


def test_non_ok_classifier_searches_all_assets_without_fabricating_coarse_category() -> None:
    store = _seed_visual_store()
    retriever = VisualRetriever(
        store=store,
        embedder=FakeQueryEmbedder([0.0, 1.0]),
        classifier=FakeCoarseClassifier({
            "coarse_category": "UNKNOWN",
            "confidence": 0.2,
            "status": "LOW_CONFIDENCE",
        }),
        top_k=1,
        min_score=0.5,
    )

    result = retriever.retrieve(b"camera-like-query")
    candidates = cast(list[Candidate], result["visual_candidates"])
    assert result["status"] == "OK"
    assert result["coarse_category"] == "UNKNOWN"
    assert result["coarse_status"] == "LOW_CONFIDENCE"
    assert result["details"] == {"coarse_filter_applied": False, "candidate_count": 1}
    assert candidates[0]["doc_id"] == "fixture_ptz_camera"
    assert candidates[0]["coarse_category"] == "智能摄像头"


def test_missing_coarse_classifier_searches_all_assets_without_fabricating_category() -> None:
    store = _seed_visual_store()
    retriever = VisualRetriever(
        store=store,
        embedder=FakeQueryEmbedder([0.0, 1.0]),
        classifier=FakeCoarseClassifier({
            "coarse_category": "UNKNOWN",
            "confidence": 0.0,
            "status": "MODEL_NOT_READY",
        }),
        top_k=1,
        min_score=0.5,
    )

    result = retriever.retrieve(b"camera-like-query")
    candidates = cast(list[Candidate], result["visual_candidates"])
    assert result["status"] == "OK"
    assert result["coarse_category"] == "UNKNOWN"
    assert result["coarse_status"] == "MODEL_NOT_READY"
    assert result["details"] == {"coarse_filter_applied": False, "candidate_count": 1}
    assert candidates[0]["doc_id"] == "fixture_ptz_camera"
    assert candidates[0]["coarse_category"] == "智能摄像头"


def test_low_score_returns_no_match_status() -> None:
    store = _seed_visual_store()
    retriever = VisualRetriever(
        store=store,
        embedder=FakeQueryEmbedder([0.0, 1.0]),
        classifier=FakeCoarseClassifier({
            "coarse_category": "UNKNOWN",
            "confidence": 0.2,
            "status": "LOW_CONFIDENCE",
        }),
        top_k=3,
        min_score=1.1,
    )

    result = retriever.retrieve(b"low-score-query")

    assert result["status"] == "NO_VISUAL_MATCH"
    assert result["coarse_category"] == "UNKNOWN"
    assert result["coarse_status"] == "LOW_CONFIDENCE"
    assert result["visual_candidates"] == []
    assert result["matched_image_count"] == 4


def test_empty_visual_store_returns_index_not_ready() -> None:
    retriever = VisualRetriever(
        store=InMemoryVisualStore(),
        embedder=FakeQueryEmbedder([1.0, 0.0]),
        classifier=FakeCoarseClassifier({
            "coarse_category": "智能传感器",
            "confidence": 0.9,
            "status": "OK",
        }),
    )

    result = retriever.retrieve(b"query")

    assert result["status"] == "INDEX_NOT_READY"
    assert result["visual_candidates"] == []


def _seed_visual_store() -> InMemoryVisualStore:
    store = InMemoryVisualStore()
    store.upsert_images(
        [
            _metadata(
                "fixture_temp_sensor",
                "智能传感器",
                "温湿度传感器",
                "temp-image-a",
                "tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/images/a.png",
            ),
            _metadata(
                "fixture_temp_sensor",
                "智能传感器",
                "温湿度传感器",
                "temp-image-b",
                "tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/images/b.png",
            ),
            _metadata(
                "fixture_pir_sensor",
                "智能传感器",
                "人体红外传感器",
                "pir-image-a",
                "tests/fixtures/iot_knowledge/智能传感器/人体红外传感器/images/a.png",
            ),
            _metadata(
                "fixture_ptz_camera",
                "智能摄像头",
                "室内云台摄像头",
                "camera-image-a",
                "tests/fixtures/iot_knowledge/智能摄像头/室内云台摄像头/images/a.png",
            ),
        ],
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.7, 0.714],
            [0.0, 1.0],
        ],
    )
    return store


def _metadata(
    doc_id: str,
    coarse_category: str,
    sub_category: str,
    image_id: str,
    image_path: str,
) -> ImageMetadata:
    return {
        "doc_id": doc_id,
        "coarse_category": coarse_category,
        "sub_category": sub_category,
        "asset_type": "image",
        "image_id": image_id,
        "image_path": image_path,
    }
