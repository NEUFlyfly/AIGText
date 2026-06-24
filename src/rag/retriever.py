"""
RAG 模块 — 检索器

职责:
  - 接收用户查询
  - 调用 embedder + vector store 完成检索
  - 返回 top-k 相关文档 chunk
"""

from typing import List, Dict

from .embedder import Embedder
from .store import VectorStore


class Retriever:
    """检索器：embedding + 向量搜索的组合。"""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self._embedder = embedder
        self._store = store

    def retrieve(
        self, query: str, top_k: int = 3, min_score: float = 0.3
    ) -> List[Dict]:
        """检索与查询最相关的 k 个文档 chunk。

        Args:
            query: 用户查询文本
            top_k: 返回结果数量
            min_score: 最低相似度阈值（0~1），低于此值的 chunk 被过滤

        Returns:
            [{"text": str, "source": str, "score": float}, ...]
        """
        if self._store.count == 0:
            return []

        query_vector = self._embedder.embed_query(query)
        results = self._store.search(query_vector, top_k=top_k)

        if min_score > 0:
            results = [r for r in results if r["score"] >= min_score]

        return results

    @property
    def store(self) -> VectorStore:
        return self._store
