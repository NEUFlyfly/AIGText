"""
RAG 模块 — 检索器

职责:
  - 接收用户查询
  - 调用 embedder + vector store 完成检索
  - 返回 top-k 相关文档 chunk
"""

from typing import Protocol, TypeAlias

MetadataFilter: TypeAlias = dict[str, object]
SearchResult: TypeAlias = dict[str, str | int | float]


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> list[float]:
        ...


class SearchStore(Protocol):
    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        ...

    @property
    def count(self) -> int:
        ...


class Retriever:
    """检索器：embedding + 向量搜索的组合。"""

    def __init__(self, embedder: QueryEmbedder, store: SearchStore):
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.3,
        doc_ids: list[str] | None = None,
        coarse_category: str | None = None,
        sub_category: str | None = None,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """检索与查询最相关的 k 个文档 chunk。

        Args:
            query: 用户查询文本
            top_k: 返回结果数量
            min_score: 最低相似度阈值（0~1），低于此值的 chunk 被过滤
            doc_ids: 限制检索的文档 ID 列表
            coarse_category: 限制检索的 IoT 粗分类
            sub_category: 限制检索的 IoT 子分类
            where: 直接传入的 ChromaDB metadata filter

        Returns:
            [{"text": str, "source": str, "score": float}, ...]
        """
        if self._store.count == 0:
            return []

        query_vector = self._embedder.embed_query(query)
        metadata_filter = where or build_metadata_filter(
            doc_ids=doc_ids,
            coarse_category=coarse_category,
            sub_category=sub_category,
        )
        results = self._store.search(
            query_vector,
            top_k=top_k,
            where=metadata_filter,
        )

        if min_score > 0:
            results = [r for r in results if r["score"] >= min_score]

        return results

    @property
    def store(self) -> SearchStore:
        return self._store


def build_metadata_filter(
    doc_ids: list[str] | None = None,
    coarse_category: str | None = None,
    sub_category: str | None = None,
) -> MetadataFilter | None:
    clauses: list[MetadataFilter] = []

    if doc_ids:
        clauses.append({"doc_id": {"$in": doc_ids}})
    if coarse_category:
        clauses.append({"coarse_category": coarse_category})
    if sub_category:
        clauses.append({"sub_category": sub_category})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
