"""Visual image vector store backed by a separate ChromaDB collection."""

from __future__ import annotations

import importlib
import math
import os
from typing import Protocol, TypeAlias, cast

from config.settings import settings


ImageMetadata: TypeAlias = dict[str, str]
VisualSearchResult: TypeAlias = dict[str, str | float]
QueryResults: TypeAlias = dict[str, object]
MetadataFilter: TypeAlias = dict[str, object]


class VisualCollection(Protocol):
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[ImageMetadata],
    ) -> None:
        ...

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: MetadataFilter | None = None,
    ) -> QueryResults:
        ...

    def count(self) -> int:
        ...


class VisualClient(Protocol):
    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, str],
    ) -> VisualCollection:
        ...

    def delete_collection(self, name: str) -> None:
        ...


class PersistentClientFactory(Protocol):
    def __call__(self, path: str, settings: object) -> VisualClient:
        ...


class SettingsFactory(Protocol):
    def __call__(self, anonymized_telemetry: bool) -> object:
        ...


_chromadb_module: object | None
_chroma_config_module: object | None
try:
    _chromadb_module = importlib.import_module("chromadb")
    _chroma_config_module = importlib.import_module("chromadb.config")
except ImportError:
    _chromadb_module = None
    _chroma_config_module = None


class VisualVectorStore:
    """Separate Chroma collection for reference image embeddings."""

    COLLECTION_NAME: str = "aigtext_visual_images"

    def __init__(self, persist_dir: str = settings.CHROMA_VISUAL_PATH) -> None:
        if _chromadb_module is None or _chroma_config_module is None:
            raise ImportError("请安装 chromadb: pip install chromadb")

        os.makedirs(persist_dir, exist_ok=True)

        persistent_client = cast(
            PersistentClientFactory,
            getattr(_chromadb_module, "PersistentClient"),
        )
        settings_factory = cast(
            SettingsFactory,
            getattr(_chroma_config_module, "Settings"),
        )

        self._client: VisualClient = persistent_client(
            path=persist_dir,
            settings=settings_factory(anonymized_telemetry=False),
        )
        self._collection: VisualCollection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._persist_dir: str = persist_dir

    def upsert_images(
        self,
        image_metadatas: list[ImageMetadata],
        embeddings: list[list[float]],
    ) -> None:
        if not image_metadatas:
            return
        if len(image_metadatas) != len(embeddings):
            raise ValueError("image metadata and embedding counts must match")

        self._collection.upsert(
            ids=[metadata["image_id"] for metadata in image_metadatas],
            embeddings=embeddings,
            documents=[metadata["image_path"] for metadata in image_metadatas],
            metadatas=image_metadatas,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        where: MetadataFilter | None = None,
    ) -> list[VisualSearchResult]:
        if self._collection.count() == 0:
            return []

        result_count = min(top_k, self._collection.count())
        if where is None:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=result_count,
            )
        else:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=result_count,
                where=where,
            )

        metadatas = _nested_metadata_list(results.get("metadatas"))
        distances = _nested_float_list(results.get("distances"))
        search_results: list[VisualSearchResult] = []
        for index, metadata in enumerate(metadatas):
            distance = distances[index] if index < len(distances) else 0.0
            score = max(0.0, 1.0 - distance)
            search_results.append({**metadata, "score": round(score, 4)})
        return search_results

    def clear(self) -> None:
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()


class InMemoryVisualStore:
    """Small injectable store for fixture indexing and tests without ChromaDB."""

    COLLECTION_NAME: str = VisualVectorStore.COLLECTION_NAME

    def __init__(self) -> None:
        self.image_metadatas: list[ImageMetadata] = []
        self.embeddings: list[list[float]] = []

    def upsert_images(
        self,
        image_metadatas: list[ImageMetadata],
        embeddings: list[list[float]],
    ) -> None:
        if len(image_metadatas) != len(embeddings):
            raise ValueError("image metadata and embedding counts must match")

        by_id = {metadata["image_id"]: index for index, metadata in enumerate(self.image_metadatas)}
        for metadata, embedding in zip(image_metadatas, embeddings):
            existing_index = by_id.get(metadata["image_id"])
            if existing_index is None:
                by_id[metadata["image_id"]] = len(self.image_metadatas)
                self.image_metadatas.append(metadata)
                self.embeddings.append(embedding)
            else:
                self.image_metadatas[existing_index] = metadata
                self.embeddings[existing_index] = embedding

    def clear(self) -> None:
        self.image_metadatas.clear()
        self.embeddings.clear()

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        where: MetadataFilter | None = None,
    ) -> list[VisualSearchResult]:
        scored_results: list[VisualSearchResult] = []
        for metadata, embedding in zip(self.image_metadatas, self.embeddings):
            if not _matches_metadata_filter(metadata, where):
                continue
            score = _cosine_similarity(query_vector, embedding)
            scored_results.append({**metadata, "score": round(max(0.0, score), 4)})

        scored_results.sort(key=lambda result: float(result["score"]), reverse=True)
        return scored_results[:top_k]

    @property
    def count(self) -> int:
        return len(self.image_metadatas)


def _nested_float_list(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        return []
    first = cast(list[object], value)[0]
    if not isinstance(first, list):
        return []
    items = cast(list[object], first)

    floats: list[float] = []
    for item in items:
        if isinstance(item, int | float):
            floats.append(float(item))
    return floats


def _nested_metadata_list(value: object) -> list[ImageMetadata]:
    if not isinstance(value, list) or not value:
        return []
    first = cast(list[object], value)[0]
    if not isinstance(first, list):
        return []
    items = cast(list[object], first)

    metadatas: list[ImageMetadata] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata: ImageMetadata = {}
        for key, raw_value in cast(dict[object, object], item).items():
            if isinstance(key, str) and isinstance(raw_value, str):
                metadata[key] = raw_value
        metadatas.append(metadata)
    return metadatas


def _matches_metadata_filter(metadata: ImageMetadata, where: MetadataFilter | None) -> bool:
    if where is None:
        return True

    and_clauses = where.get("$and")
    if isinstance(and_clauses, list):
        typed_clauses = cast(list[object], and_clauses)
        for raw_clause in typed_clauses:
            if not isinstance(raw_clause, dict):
                return False
            if not _matches_metadata_filter(metadata, cast(MetadataFilter, raw_clause)):
                return False
        return True

    for key, expected in where.items():
        if key == "$and":
            continue
        actual = metadata.get(key)
        if isinstance(expected, dict):
            expected_filter = cast(dict[str, object], expected)
            allowed = expected_filter.get("$in")
            if isinstance(allowed, list) and actual not in allowed:
                return False
        elif actual != expected:
            return False
    return True


def _cosine_similarity(query_vector: list[float], stored_vector: list[float]) -> float:
    if not query_vector or not stored_vector or len(query_vector) != len(stored_vector):
        return 0.0
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    stored_norm = math.sqrt(sum(value * value for value in stored_vector))
    if query_norm == 0.0 or stored_norm == 0.0:
        return 0.0
    dot_product = sum(query_value * stored_value for query_value, stored_value in zip(query_vector, stored_vector))
    return dot_product / (query_norm * stored_norm)
