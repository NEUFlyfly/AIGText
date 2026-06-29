"""Coarse IoT category classifier interface for Visual RAG."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

from config.settings import settings


CoarseStatus: TypeAlias = Literal["OK", "LOW_CONFIDENCE", "MODEL_NOT_READY", "UNKNOWN"]
CoarseClassification: TypeAlias = dict[str, str | float]


class CoarseClassifierBackend(Protocol):
    def classify(self, image_bytes: bytes) -> Mapping[str, object]:
        ...


class CoarseClassifier:
    """Stable coarse-category classifier facade.

    Production weights are intentionally not downloaded or loaded here. Runtime callers can
    pass a local backend, while tests can inject deterministic fake classifiers.
    """

    def __init__(
        self,
        *,
        taxonomy_path: str | Path | None = None,
        labels: list[str] | None = None,
        model: CoarseClassifierBackend | None = None,
        confidence_threshold: float = settings.COARSE_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._taxonomy_path: Path = Path(taxonomy_path or _default_taxonomy_path())
        self._labels: list[str] = labels or load_coarse_labels(self._taxonomy_path)
        self._model: CoarseClassifierBackend | None = model
        self._confidence_threshold: float = confidence_threshold

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def classify(self, image_bytes: bytes) -> CoarseClassification:
        if self._model is None:
            return classification_result("UNKNOWN", 0.0, "MODEL_NOT_READY")
        if not image_bytes:
            return classification_result("UNKNOWN", 0.0, "UNKNOWN")

        try:
            raw_result = self._model.classify(image_bytes)
        except (OSError, ValueError, RuntimeError):
            return classification_result("UNKNOWN", 0.0, "UNKNOWN")

        return self._normalize_backend_result(raw_result)

    def _normalize_backend_result(
        self,
        raw_result: Mapping[str, object],
    ) -> CoarseClassification:
        raw_category = raw_result.get("coarse_category", raw_result.get("category"))
        raw_confidence = raw_result.get("confidence", 0.0)

        coarse_category = raw_category if isinstance(raw_category, str) else "UNKNOWN"
        confidence = _coerce_confidence(raw_confidence)

        if coarse_category not in self._labels:
            return classification_result("UNKNOWN", confidence, "UNKNOWN")
        if confidence < self._confidence_threshold:
            return classification_result("UNKNOWN", confidence, "LOW_CONFIDENCE")

        return classification_result(coarse_category, confidence, "OK")


def classification_result(
    coarse_category: str,
    confidence: float,
    status: CoarseStatus,
) -> CoarseClassification:
    return {
        "coarse_category": coarse_category,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "status": status,
    }


def load_coarse_labels(taxonomy_path: str | Path | None = None) -> list[str]:
    path = Path(taxonomy_path or _default_taxonomy_path())
    try:
        taxonomy_data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(taxonomy_data, list):
        return []

    labels: list[str] = []
    seen_labels: set[str] = set()
    for raw_entry in cast(list[object], taxonomy_data):
        if not isinstance(raw_entry, dict):
            continue
        coarse_category = cast(dict[object, object], raw_entry).get("coarse_category")
        if not isinstance(coarse_category, str) or not coarse_category:
            continue
        if coarse_category in seen_labels:
            continue
        labels.append(coarse_category)
        seen_labels.add(coarse_category)
    return labels


def _coerce_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _default_taxonomy_path() -> str:
    return str(Path(settings.IOT_DOCUMENTS_DIR) / "taxonomy.json")
