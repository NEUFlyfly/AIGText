"""
RAG 模块 — Embedding 模型封装

职责:
  - 加载本地 embedding 模型 (BAAI/bge-small-zh-v1.5)
  - 将文本批量转换为向量
  - 支持文档和查询两种模式的 embedding
"""

import importlib
import os
from typing import Protocol, cast

# 必须在 import sentence_transformers 之前设置：
#   HF_HOME         → 模型缓存目录（默认 C 盘，改为项目本地）
#   HF_ENDPOINT     → 国内镜像（首次下载用）
#   HF_HUB_OFFLINE  → 禁止版本检查网络请求（模型已缓存时避免超时）

# 模型缓存到项目 models/embedding/ 下，避免污染 C 盘
_embedding_cache = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models", "embedding"
)
os.makedirs(_embedding_cache, exist_ok=True)
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = _embedding_cache
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    _ = importlib.import_module("sentence_transformers")
    _has_sentence_transformers = True
except ImportError:
    _has_sentence_transformers = False


class EmbeddingArray(Protocol):
    def tolist(self) -> list[float] | list[list[float]]:
        ...


class TextEmbeddingModel(Protocol):
    def encode(
        self,
        texts: str | list[str],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> EmbeddingArray:
        ...


class SentenceTransformerFactory(Protocol):
    def __call__(
        self,
        model_name_or_path: str,
        *,
        local_files_only: bool,
    ) -> TextEmbeddingModel:
        ...


class Embedder:
    """文本向量化器。

    使用 BAAI/bge-small-zh-v1.5 模型，
    约 100MB 大小。运行时只使用 models/embedding 下的本地模型，
    不发起任何网络请求。
    """

    _DEFAULT_MODEL: str = "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_name: str = ""):
        if not _has_sentence_transformers:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )
        self._model_name: str = model_name or self._DEFAULT_MODEL
        self._model: TextEmbeddingModel | None = None

    @property
    def model(self) -> TextEmbeddingModel:
        """延迟加载本地模型。"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> TextEmbeddingModel:
        try:
            return _sentence_transformer_factory()(self._model_name, local_files_only=True)
        except Exception as exc:
            message = "".join([
                "Text embedding model is not available locally. ",
                f"Place or download {self._model_name} under {_embedding_cache} ",
                "and retry; online model downloads are disabled at runtime.",
            ])
            raise RuntimeError(
                message
            ) from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
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
        return cast(list[list[float]], embeddings.tolist())

    def embed_query(self, query: str) -> list[float]:
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
        return cast(list[float], embedding.tolist())


def _sentence_transformer_factory() -> SentenceTransformerFactory:
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise ImportError(
            "请安装 sentence-transformers: pip install sentence-transformers"
        ) from exc
    return cast(SentenceTransformerFactory, getattr(module, "SentenceTransformer"))
