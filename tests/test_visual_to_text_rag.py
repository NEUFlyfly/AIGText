from typing import cast

from src.rag.retriever import MetadataFilter, SearchResult
from src.rag.visual_pipeline import VisualRAGPipeline
from src.rag.visual_retriever import VisualCandidate, VisualRetrievalResult


class FakeVisualRetriever:
    def __init__(self, result: VisualRetrievalResult) -> None:
        self.result: VisualRetrievalResult = result
        self.inputs: list[bytes] = []

    def retrieve(self, image_bytes: bytes) -> VisualRetrievalResult:
        self.inputs.append(image_bytes)
        return self.result


class FakeTextPipeline:
    def __init__(self, chunks: list[SearchResult]) -> None:
        self._chunks: list[SearchResult] = chunks
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        coarse_category: str | None = None,
        sub_category: str | None = None,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        self.calls.append({
            "query": query,
            "doc_ids": doc_ids,
            "coarse_category": coarse_category,
            "sub_category": sub_category,
            "where": where,
        })
        if doc_ids is None:
            return list(self._chunks)
        allowed_doc_ids = set(doc_ids)
        return [
            chunk for chunk in self._chunks
            if str(chunk["doc_id"]) in allowed_doc_ids
        ]


def test_text_rag_filtered_to_visual_doc_ids() -> None:
    visual_retriever = FakeVisualRetriever(_visual_result([
        _candidate("fixture_temp_sensor", "温湿度传感器", 0.94),
    ]))
    text_pipeline = FakeTextPipeline(_text_chunks())
    pipeline = VisualRAGPipeline(
        visual_retriever=visual_retriever,
        text_pipeline=text_pipeline,
    )

    payload = pipeline.run(b"temp-sensor-image", "这个设备有什么用途？")

    assert payload["status"] == "OK"
    assert visual_retriever.inputs == [b"temp-sensor-image"]
    assert text_pipeline.calls == [{
        "query": "这个设备有什么用途？",
        "doc_ids": ["fixture_temp_sensor"],
        "coarse_category": None,
        "sub_category": None,
        "where": None,
    }]

    retrieved_chunks = cast(list[SearchResult], payload["retrieved_chunks"])
    assert retrieved_chunks
    assert {chunk["doc_id"] for chunk in retrieved_chunks} == {"fixture_temp_sensor"}


def test_augmented_prompt_contains_candidate_summary_and_numbered_text_sources() -> None:
    pipeline = VisualRAGPipeline(
        visual_retriever=FakeVisualRetriever(_visual_result([
            _candidate("fixture_temp_sensor", "温湿度传感器", 0.94),
        ])),
        text_pipeline=FakeTextPipeline(_text_chunks()),
    )

    payload = pipeline.run(b"temp-sensor-image", "请介绍这个设备")

    prompt = str(payload["augmented_prompt"])
    assert "视觉候选设备" in prompt
    assert "doc_id=fixture_temp_sensor" in prompt
    assert "子类别=温湿度传感器" in prompt
    assert "[1] 温湿度传感器用于采集温度和湿度。" in prompt
    assert "(来源: tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/document.md, 相关度: 0.88)" in prompt
    assert "人体红外传感器用于检测人员移动。" not in prompt


def test_no_visual_candidates_returns_clear_status_without_text_context() -> None:
    visual_retriever = FakeVisualRetriever(_visual_result([], status="NO_VISUAL_MATCH"))
    text_pipeline = FakeTextPipeline(_text_chunks())
    pipeline = VisualRAGPipeline(
        visual_retriever=visual_retriever,
        text_pipeline=text_pipeline,
    )

    payload = pipeline.run(b"unknown-image", "这是什么？")

    assert payload == {
        "coarse_category": "UNKNOWN",
        "coarse_confidence": 0.2,
        "coarse_status": "LOW_CONFIDENCE",
        "visual_candidates": [],
        "retrieved_chunks": [],
        "augmented_prompt": "",
        "status": "NO_VISUAL_MATCH",
    }
    assert text_pipeline.calls == []


def test_candidate_without_text_chunks_returns_clear_status() -> None:
    visual_retriever = FakeVisualRetriever(_visual_result([
        _candidate("fixture_missing_document", "不存在的设备", 0.91),
    ]))
    text_pipeline = FakeTextPipeline(_text_chunks())
    pipeline = VisualRAGPipeline(
        visual_retriever=visual_retriever,
        text_pipeline=text_pipeline,
    )

    payload = pipeline.run(b"missing-doc-image", "这个设备有什么用途？")

    assert payload["status"] == "NO_TEXT_CHUNKS"
    assert payload["retrieved_chunks"] == []
    assert text_pipeline.calls[0]["doc_ids"] == ["fixture_missing_document"]

    prompt = str(payload["augmented_prompt"])
    assert "doc_id=fixture_missing_document" in prompt
    assert "未检索到可引用的候选文档文本片段" in prompt
    assert "温湿度传感器用于采集温度和湿度。" not in prompt
    assert "人体红外传感器用于检测人员移动。" not in prompt


def test_visual_index_not_ready_status_is_preserved() -> None:
    pipeline = VisualRAGPipeline(
        visual_retriever=FakeVisualRetriever(_visual_result([], status="INDEX_NOT_READY")),
        text_pipeline=FakeTextPipeline(_text_chunks()),
    )

    payload = pipeline.query(b"query-image", "这是什么？")

    assert payload["status"] == "INDEX_NOT_READY"
    assert payload["visual_candidates"] == []
    assert payload["retrieved_chunks"] == []
    assert payload["augmented_prompt"] == ""


def _visual_result(
    candidates: list[VisualCandidate],
    *,
    status: str = "OK",
) -> VisualRetrievalResult:
    return {
        "status": status,
        "coarse_category": "UNKNOWN" if not candidates else "智能传感器",
        "coarse_confidence": 0.2 if not candidates else 0.91,
        "coarse_status": "LOW_CONFIDENCE" if not candidates else "OK",
        "visual_candidates": candidates,
        "matched_image_count": len(candidates),
        "details": {"coarse_filter_applied": bool(candidates), "candidate_count": len(candidates)},
    }


def _candidate(doc_id: str, sub_category: str, score: float) -> VisualCandidate:
    return {
        "doc_id": doc_id,
        "sub_category": sub_category,
        "coarse_category": "智能传感器",
        "score": score,
        "evidence_image_id": f"{doc_id}-image-a",
        "evidence_image_path": f"tests/fixtures/iot_knowledge/智能传感器/{sub_category}/images/a.png",
        "matched_image_count": 2,
        "status": "OK",
    }


def _text_chunks() -> list[SearchResult]:
    return [
        {
            "text": "温湿度传感器用于采集温度和湿度。",
            "source": "tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/document.md",
            "doc_id": "fixture_temp_sensor",
            "coarse_category": "智能传感器",
            "sub_category": "温湿度传感器",
            "asset_type": "text",
            "chunk_id": 0,
            "score": 0.88,
        },
        {
            "text": "人体红外传感器用于检测人员移动。",
            "source": "tests/fixtures/iot_knowledge/智能传感器/人体红外传感器/document.md",
            "doc_id": "fixture_pir_sensor",
            "coarse_category": "智能传感器",
            "sub_category": "人体红外传感器",
            "asset_type": "text",
            "chunk_id": 0,
            "score": 0.86,
        },
    ]
