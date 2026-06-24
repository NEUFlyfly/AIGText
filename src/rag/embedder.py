"""
RAG 模块 — Embedding 模型封装

职责:
  - 加载本地 embedding 模型 (BAAI/bge-small-zh-v1.5)
  - 将文本批量转换为向量
  - 支持文档和查询两种模式的 embedding
"""

import os

# 必须在 import sentence_transformers 之前设置：
#   HF_ENDPOINT     → 国内镜像（首次下载用）
#   HF_HUB_OFFLINE  → 禁止版本检查网络请求（模型已缓存时避免超时）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

from typing import List

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


class Embedder:
    """文本向量化器。

    使用 BAAI/bge-small-zh-v1.5 模型，
    约 100MB 大小，首次运行自动下载。

    首次下载走国内镜像 hf-mirror.com，
    已缓存后走纯本地模式（HF_HUB_OFFLINE=1），
    不再发起任何网络请求。
    """

    _DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_name: str = ""):
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )
        self._model_name = model_name or self._DEFAULT_MODEL
        self._model: "SentenceTransformer | None" = None

    @property
    def model(self) -> "SentenceTransformer":
        """延迟加载模型。优先纯本地，失败则临时放行网络重试。"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> "SentenceTransformer":
        try:
            return SentenceTransformer(self._model_name, local_files_only=True)
        except Exception:
            # 缓存不完整，临时放开网络走镜像下载
            os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                return SentenceTransformer(self._model_name)
            finally:
                os.environ["HF_HUB_OFFLINE"] = "1"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """将文档文本批量转换为归一化向量。

        Args:
            texts: 待编码文本列表

        Returns:
            向量列表，每个向量为 float 列表
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """将查询文本转换为归一化向量。

        Args:
            query: 查询文本

        Returns:
            归一化向量
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
