import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from src.rag.iot_loader import load_iot_documents
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import MetadataFilter, Retriever, SearchResult
from src.rag.splitter import split_text
from src.rag.visual_index import build_visual_index
from src.rag.visual_pipeline import VisualRAGPipeline
from src.rag.visual_retriever import VisualCandidate, VisualRetrievalResult, VisualRetriever
from src.rag.visual_store import InMemoryVisualStore
from src.vision.coarse_classifier import CoarseClassification


REPO_ROOT = Path(__file__).resolve().parents[1]


Chunk = dict[str, str | int | float]


class FixtureVisualEmbedder:
    def embed_images(self, image_inputs: Iterable[object]) -> list[list[float]]:
        return [_visual_vector(str(image_input)) for image_input in image_inputs]


class FixtureQueryImageEmbedder:
    def embed_image(self, image_input: object) -> list[float]:
        assert image_input == b"fixture-query-image"
        return [1.0, 0.0]


class FixtureCoarseClassifier:
    def classify(self, image_bytes: bytes) -> CoarseClassification:
        assert image_bytes == b"fixture-query-image"
        return {
            "coarse_category": "智能传感器",
            "confidence": 0.93,
            "status": "OK",
        }


class FixtureTextEmbedder:
    def embed_query(self, query: str) -> list[float]:
        assert query
        return [1.0, 0.0]


class MemoryTextStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks: list[Chunk] = chunks
        self.seen_filters: list[MetadataFilter | None] = []

    @property
    def count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        assert query_vector == [1.0, 0.0]
        self.seen_filters.append(where)
        matched_chunks = [chunk for chunk in self._chunks if _matches_where(chunk, where)]
        return [_search_result(chunk) for chunk in matched_chunks[:top_k]]


class StaticVisualRetriever:
    def __init__(self, result: VisualRetrievalResult) -> None:
        self.result: VisualRetrievalResult = result

    def retrieve(self, image_bytes: bytes) -> VisualRetrievalResult:
        assert image_bytes
        return self.result


class RecordingTextPipeline:
    def __init__(self, chunks: list[SearchResult]) -> None:
        self.chunks: list[SearchResult] = chunks
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
        return list(self.chunks)


def test_image_to_topk_to_text_chunks_to_augmented_payload() -> None:
    visual_store = InMemoryVisualStore()
    visual_report = build_visual_index(
        iot_documents_dir="tests/fixtures/iot_knowledge",
        fixture_mode=True,
        store=visual_store,
        embedder=FixtureVisualEmbedder(),
    )
    assert visual_report.errors == []
    assert visual_report.indexed_count >= 3

    text_store = MemoryTextStore(_fixture_text_chunks())
    text_pipeline = _fixture_text_pipeline(text_store)
    visual_retriever = VisualRetriever(
        store=visual_store,
        embedder=FixtureQueryImageEmbedder(),
        classifier=FixtureCoarseClassifier(),
        top_k=2,
        min_score=0.1,
    )
    pipeline = VisualRAGPipeline(
        visual_retriever=visual_retriever,
        text_pipeline=text_pipeline,
    )

    payload = pipeline.run(b"fixture-query-image", "请介绍这个传感器的用途")

    assert payload["status"] == "OK"
    assert payload["coarse_category"] == "智能传感器"

    visual_candidates = cast(list[VisualCandidate], payload["visual_candidates"])
    retrieved_chunks = cast(list[SearchResult], payload["retrieved_chunks"])
    assert visual_candidates
    assert retrieved_chunks
    assert payload["augmented_prompt"]

    visual_doc_ids = {str(candidate["doc_id"]) for candidate in visual_candidates}
    retrieved_doc_ids = {str(chunk["doc_id"]) for chunk in retrieved_chunks}
    assert visual_doc_ids == {"fixture_temp_sensor", "fixture_pir_sensor"}
    assert retrieved_doc_ids == visual_doc_ids
    assert all(chunk["doc_id"] in visual_doc_ids for chunk in retrieved_chunks)
    assert "室内云台摄像头" not in str(payload["augmented_prompt"])
    assert text_store.seen_filters == [
        {"doc_id": {"$in": ["fixture_temp_sensor", "fixture_pir_sensor"]}}
    ]


def test_visual_rag_fallbacks_do_not_fabricate_context() -> None:
    text_pipeline = RecordingTextPipeline(_fixture_search_results())
    no_visual_pipeline = VisualRAGPipeline(
        visual_retriever=StaticVisualRetriever(_visual_result([])),
        text_pipeline=text_pipeline,
    )

    no_visual_payload = no_visual_pipeline.run(b"unknown-image", "这是什么设备？")

    assert no_visual_payload["status"] == "NO_VISUAL_MATCH"
    assert no_visual_payload["visual_candidates"] == []
    assert no_visual_payload["retrieved_chunks"] == []
    assert no_visual_payload["augmented_prompt"] == ""
    assert text_pipeline.calls == []

    no_text_pipeline = VisualRAGPipeline(
        visual_retriever=StaticVisualRetriever(_visual_result([
            _candidate("fixture_temp_sensor", "温湿度传感器", 0.94),
        ])),
        text_pipeline=RecordingTextPipeline([]),
    )

    no_text_payload = no_text_pipeline.run(b"temp-image", "这个设备怎么用？")

    assert no_text_payload["status"] == "NO_TEXT_CHUNKS"
    assert no_text_payload["retrieved_chunks"] == []
    assert "未检索到可引用的候选文档文本片段" in str(no_text_payload["augmented_prompt"])


def test_fixture_index_commands_exit_zero_without_live_models(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    commands = [
        [
            sys.executable,
            "-m",
            "src.rag.index",
            "--fixtures",
            "--persist-dir",
            str(tmp_path / "text_vectorstore"),
        ],
        [
            sys.executable,
            "-m",
            "src.rag.visual_index",
            "--fixtures",
            "--persist-dir",
            str(tmp_path / "visual_vectorstore"),
        ],
    ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout


def _fixture_text_pipeline(store: MemoryTextStore) -> RAGPipeline:
    pipeline = object.__new__(RAGPipeline)
    setattr(pipeline, "_store", store)
    setattr(pipeline, "_top_k", 5)
    setattr(pipeline, "_min_score", 0.0)
    setattr(pipeline, "_embedder", FixtureTextEmbedder())
    setattr(pipeline, "_retriever", Retriever(FixtureTextEmbedder(), store))
    return pipeline


def _fixture_text_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_iot_documents("tests/fixtures/iot_knowledge"):
        chunks.extend(_test_chunks(split_text(
            document["content"],
            document["source"],
            chunk_size=240,
            chunk_overlap=20,
            metadata={
                "doc_id": document["doc_id"],
                "coarse_category": document["coarse_category"],
                "sub_category": document["sub_category"],
                "asset_type": document["asset_type"],
            },
        )))
    return chunks


def _test_chunks(chunks: list[dict[str, str | int]]) -> list[Chunk]:
    return [dict(chunk) for chunk in chunks]


def _fixture_search_results() -> list[SearchResult]:
    return [_search_result(chunk) for chunk in _fixture_text_chunks()]


def _search_result(chunk: Chunk) -> SearchResult:
    return {
        "text": str(chunk["text"]),
        "source": str(chunk["source"]),
        "doc_id": str(chunk["doc_id"]),
        "coarse_category": str(chunk["coarse_category"]),
        "sub_category": str(chunk["sub_category"]),
        "asset_type": str(chunk["asset_type"]),
        "chunk_id": int(chunk["chunk_id"]),
        "score": 0.99,
    }


def _matches_where(chunk: Chunk, where: MetadataFilter | None) -> bool:
    if where is None:
        return True

    and_clauses = where.get("$and")
    if isinstance(and_clauses, list):
        for raw_clause in cast(list[object], and_clauses):
            if not isinstance(raw_clause, dict):
                return False
            clause = _metadata_filter(cast(dict[object, object], raw_clause))
            if not _matches_where(chunk, clause):
                return False
        return True

    for key, expected in where.items():
        if key == "$and":
            continue
        actual = chunk.get(key)
        if isinstance(expected, dict):
            expected_filter = _metadata_filter(cast(dict[object, object], expected))
            allowed = expected_filter.get("$in")
            if isinstance(allowed, list) and actual not in allowed:
                return False
        elif actual != expected:
            return False
    return True


def _metadata_filter(raw_filter: dict[object, object]) -> MetadataFilter:
    return {
        key: value
        for key, value in raw_filter.items()
        if isinstance(key, str)
    }


def _visual_vector(image_path: str) -> list[float]:
    normalized_path = image_path.replace("\\", "/")
    if "温湿度传感器" in normalized_path:
        return [1.0, 0.0]
    if "人体红外传感器" in normalized_path:
        return [0.8, 0.6]
    if "室内云台摄像头" in normalized_path:
        return [0.0, 1.0]
    return [0.0, 0.0]


def _visual_result(candidates: list[VisualCandidate]) -> VisualRetrievalResult:
    return {
        "status": "OK" if candidates else "NO_VISUAL_MATCH",
        "coarse_category": "智能传感器" if candidates else "UNKNOWN",
        "coarse_confidence": 0.93 if candidates else 0.2,
        "coarse_status": "OK" if candidates else "LOW_CONFIDENCE",
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
        "matched_image_count": 1,
        "status": "OK",
    }
