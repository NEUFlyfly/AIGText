"""Constrained visual Top-K retrieval for IoT Visual RAG."""

from __future__ import annotations

from typing import Protocol, TypeAlias

from config.settings import settings
from src.vision.coarse_classifier import CoarseClassification, CoarseClassifier

from .visual_embedder import VisualEmbedder
from .visual_store import MetadataFilter, VisualSearchResult, VisualVectorStore


VisualCandidate: TypeAlias = dict[str, str | float | int]
VisualRetrievalResult: TypeAlias = dict[str, object]


class QueryImageEmbedder(Protocol):
    def embed_image(self, image_input: object) -> list[float]:
        ...


class VisualSearchStore(Protocol):
    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        where: MetadataFilter | None = None,
    ) -> list[VisualSearchResult]:
        ...

    @property
    def count(self) -> int:
        ...


class QueryCoarseClassifier(Protocol):
    def classify(self, image_bytes: bytes) -> CoarseClassification:
        ...


class VisualRetriever:
    """Retrieve unique document candidates from reference image embeddings."""

    def __init__(
        self,
        *,
        embedder: QueryImageEmbedder | None = None,
        store: VisualSearchStore | None = None,
        classifier: QueryCoarseClassifier | None = None,
        top_k: int = settings.VISUAL_TOP_K,
        min_score: float = settings.VISUAL_MIN_SCORE,
    ) -> None:
        self._embedder: QueryImageEmbedder = embedder or VisualEmbedder()
        self._store: VisualSearchStore = store or VisualVectorStore()
        self._classifier: QueryCoarseClassifier = classifier or CoarseClassifier()
        self._top_k: int = top_k
        self._min_score: float = min_score

    def retrieve(
        self,
        image_bytes: bytes,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        classification: CoarseClassification | None = None,
    ) -> VisualRetrievalResult:
        result_top_k = top_k or self._top_k
        result_min_score = self._min_score if min_score is None else min_score
        coarse_result = classification or self._classifier.classify(image_bytes)

        if self._store.count == 0:
            return _result(
                status="INDEX_NOT_READY",
                coarse_result=coarse_result,
                candidates=[],
                matched_image_count=0,
                coarse_filter_applied=False,
            )

        query_vector = self._embedder.embed_image(image_bytes)
        coarse_status = str(coarse_result["status"])
        coarse_category = str(coarse_result["coarse_category"])
        coarse_filter = _coarse_filter(coarse_status, coarse_category)
        search_limit = max(result_top_k, self._store.count)
        raw_results = self._store.search(
            query_vector,
            top_k=search_limit,
            where=coarse_filter,
        )

        candidates = _deduplicate_candidates(
            raw_results,
            top_k=result_top_k,
            min_score=result_min_score,
        )
        if not candidates:
            return _result(
                status="NO_VISUAL_MATCH",
                coarse_result=coarse_result,
                candidates=[],
                matched_image_count=len(raw_results),
                coarse_filter_applied=coarse_filter is not None,
            )

        return _result(
            status="OK",
            coarse_result=coarse_result,
            candidates=candidates,
            matched_image_count=len(raw_results),
            coarse_filter_applied=coarse_filter is not None,
        )

    @property
    def store(self) -> VisualSearchStore:
        return self._store


def _coarse_filter(status: str, coarse_category: str) -> MetadataFilter | None:
    if status == "OK" and coarse_category != "UNKNOWN":
        return {"coarse_category": coarse_category}
    return None


def _deduplicate_candidates(
    raw_results: list[VisualSearchResult],
    *,
    top_k: int,
    min_score: float,
) -> list[VisualCandidate]:
    by_doc_id: dict[str, VisualCandidate] = {}
    for raw_result in raw_results:
        score = _float_value(raw_result.get("score"))
        if score < min_score:
            continue

        doc_id = str(raw_result.get("doc_id", ""))
        if not doc_id:
            continue

        current = by_doc_id.get(doc_id)
        if current is None:
            by_doc_id[doc_id] = _candidate_from_result(raw_result, score)
            continue

        current["matched_image_count"] = int(current["matched_image_count"]) + 1
        if score > float(current["score"]):
            best_candidate = _candidate_from_result(raw_result, score)
            best_candidate["matched_image_count"] = current["matched_image_count"]
            by_doc_id[doc_id] = best_candidate

    candidates = list(by_doc_id.values())
    candidates.sort(key=lambda candidate: float(candidate["score"]), reverse=True)
    return candidates[:top_k]


def _candidate_from_result(raw_result: VisualSearchResult, score: float) -> VisualCandidate:
    return {
        "doc_id": str(raw_result.get("doc_id", "")),
        "sub_category": str(raw_result.get("sub_category", "UNKNOWN")),
        "coarse_category": str(raw_result.get("coarse_category", "UNKNOWN")),
        "score": round(score, 4),
        "evidence_image_id": str(raw_result.get("image_id", "")),
        "evidence_image_path": str(raw_result.get("image_path", "")),
        "matched_image_count": 1,
        "status": "OK",
    }


def _result(
    *,
    status: str,
    coarse_result: CoarseClassification,
    candidates: list[VisualCandidate],
    matched_image_count: int,
    coarse_filter_applied: bool,
) -> VisualRetrievalResult:
    return {
        "status": status,
        "coarse_category": coarse_result["coarse_category"],
        "coarse_confidence": coarse_result["confidence"],
        "coarse_status": coarse_result["status"],
        "visual_candidates": candidates,
        "matched_image_count": matched_image_count,
        "details": {
            "coarse_filter_applied": coarse_filter_applied,
            "candidate_count": len(candidates),
        },
    }


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0
