"""
RAG 模块 — 向量库封装 (ChromaDB)

职责:
  - 持久化存储文档 chunk 的 embedding 向量及元数据
  - 按查询向量检索最相似的 top-k chunk
"""

import os
from typing import List, Dict, Optional

try:
    import chromadb
    from chromadb.config import Settings

    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False


class VectorStore:
    """ChromaDB 向量存储封装。

    使用持久化模式，数据保存在指定目录。
    """

    _COLLECTION_NAME = "aigtext_docs"

    def __init__(self, persist_dir: str = "./data/vectorstore"):
        if not _HAS_CHROMADB:
            raise ImportError(
                "请安装 chromadb: pip install chromadb"
            )

        os.makedirs(persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._persist_dir = persist_dir

    def upsert(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """插入或更新文档向量。

        Args:
            chunks: [{"chunk_id": int, "text": str, "source": str}, ...]
            embeddings: 对应的向量列表
        """
        if not chunks:
            return

        ids = [f"{c['source']}__{c['chunk_id']}" for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {"source": c["source"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self, query_vector: List[float], top_k: int = 3
    ) -> List[Dict]:
        """检索最相似的 top_k 个 chunk。

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量

        Returns:
            [{"text": str, "source": str, "score": float}, ...]
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self._collection.count()),
        )

        docs: List[Dict] = []
        if results.get("documents") and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                meta = (
                    results["metadatas"][0][i]
                    if results.get("metadatas") and results["metadatas"][0]
                    else {}
                )
                distance = (
                    results["distances"][0][i]
                    if results.get("distances") and results["distances"][0]
                    else 0.0
                )
                # hnsw:space=cosine, distance 即余弦距离 (0~2)
                # 转换为相似度: cos_sim = 1 - distance, 范围 [-1, 1]
                score = max(0.0, 1.0 - distance)
                docs.append({
                    "text": doc_text,
                    "source": meta.get("source", "unknown"),
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
