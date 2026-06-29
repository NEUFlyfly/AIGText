"""
RAG 模块 — 向量库封装 (ChromaDB)

职责:
  - 持久化存储文档 chunk 的 embedding 向量及元数据
  - 按查询向量检索最相似的 top-k chunk
"""

import os
import importlib
from typing import Protocol, TypeAlias, cast


Chunk: TypeAlias = dict[str, str | int | float]
MetadataFilter: TypeAlias = dict[str, object]
SearchResult: TypeAlias = dict[str, str | int | float]
QueryResults: TypeAlias = dict[str, object]


class ChromaCollection(Protocol):
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str | int | float],
        metadatas: list[dict[str, str | int | float]],
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


class ChromaClient(Protocol):
    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, str],
    ) -> ChromaCollection:
        ...

    def delete_collection(self, name: str) -> None:
        ...


class PersistentClientFactory(Protocol):
    def __call__(self, path: str, settings: object) -> ChromaClient:
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


class VectorStore:
    """ChromaDB 向量存储封装。

    使用持久化模式，数据保存在指定目录。
    """

    _COLLECTION_NAME: str = "aigtext_docs"

    def __init__(self, persist_dir: str = "./data/vectorstore"):
        if _chromadb_module is None or _chroma_config_module is None:
            raise ImportError(
                "请安装 chromadb: pip install chromadb"
            )

        os.makedirs(persist_dir, exist_ok=True)

        persistent_client = cast(
            PersistentClientFactory,
            getattr(_chromadb_module, "PersistentClient"),
        )
        settings_factory = cast(
            SettingsFactory,
            getattr(_chroma_config_module, "Settings"),
        )

        self._client: ChromaClient = persistent_client(
            path=persist_dir,
            settings=settings_factory(anonymized_telemetry=False),
        )
        self._collection: ChromaCollection = self._client.get_or_create_collection(
            name=self._COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._persist_dir: str = persist_dir

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """插入或更新文档向量。

        Args:
            chunks: [{"chunk_id": int, "text": str, "source": str, ...metadata}, ...]
            embeddings: 对应的向量列表
        """
        if not chunks:
            return

        ids = [
            f"{c.get('doc_id', c['source'])}__{c['source']}__{c['chunk_id']}"
            for c in chunks
        ]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "doc_id": c.get("doc_id", "unknown"),
                "coarse_category": c.get("coarse_category", "unknown"),
                "sub_category": c.get("sub_category", "unknown"),
                "asset_type": c.get("asset_type", "text"),
                "source": c["source"],
                "chunk_id": c["chunk_id"],
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """检索最相似的 top_k 个 chunk。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            where: ChromaDB metadata filter

        Returns:
            [{"text": str, "source": str, "score": float, ...metadata}, ...]
        """
        if self._collection.count() == 0:
            return []

        if where is None:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self._collection.count()),
            )
        else:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self._collection.count()),
                where=where,
            )

        docs: list[SearchResult] = []
        documents = _nested_list(results.get("documents"))
        metadatas = _nested_metadata_list(results.get("metadatas"))
        distances = _nested_float_list(results.get("distances"))

        if documents:
            for i, doc_text in enumerate(documents):
                meta = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0.0
                # hnsw:space=cosine, distance 即余弦距离 (0~2)
                # 转换为相似度: cos_sim = 1 - distance, 范围 [-1, 1]
                score = max(0.0, 1.0 - distance)
                docs.append({
                    "text": doc_text,
                    "source": meta.get("source", "unknown"),
                    "doc_id": meta.get("doc_id", "unknown"),
                    "coarse_category": meta.get("coarse_category", "unknown"),
                    "sub_category": meta.get("sub_category", "unknown"),
                    "asset_type": meta.get("asset_type", "text"),
                    "chunk_id": meta.get("chunk_id", -1),
                    "score": round(score, 4),
                })

        return docs

    def clear(self) -> None:
        """清空当前 collection 中的所有数据。"""
        self._client.delete_collection(self._COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """当前 collection 中的文档数量。"""
        return self._collection.count()


def _nested_list(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if not isinstance(first, list):
        return []
    return [str(item) for item in first]


def _nested_float_list(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if not isinstance(first, list):
        return []

    floats: list[float] = []
    for item in first:
        if isinstance(item, int | float):
            floats.append(float(item))
    return floats


def _nested_metadata_list(value: object) -> list[dict[str, str | int | float]]:
    if not isinstance(value, list) or not value:
        return []
    first = value[0]
    if not isinstance(first, list):
        return []

    metadatas: list[dict[str, str | int | float]] = []
    for item in first:
        if not isinstance(item, dict):
            continue
        metadata: dict[str, str | int | float] = {}
        for key, raw_value in item.items():
            if isinstance(key, str) and isinstance(raw_value, str | int | float):
                metadata[key] = raw_value
        metadatas.append(metadata)
    return metadatas
