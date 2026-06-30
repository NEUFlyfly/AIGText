"""
RAG 模块 — 问答管线编排器

职责:
  - 加载检索器 → 检索知识库 → 构造 prompt → 返回上下文
  - 从 prompt/ 目录读取 prompt 模板
  - 将检索结果格式化为模型可理解的上下文
"""

from typing import TypeAlias

from ..prompt_loader import load_prompt
from .embedder import Embedder
from .store import VectorStore
from .retriever import MetadataFilter, QueryEmbedder, Retriever, SearchResult, SearchStore


Chunk: TypeAlias = dict[str, str | int | float]


# RAG 模板名 — 对应 prompt/rag_prompt.txt
_RAG_PROMPT_NAME = "rag_prompt"
# 语音模式模板名 — 对应 prompt/voice_prompt.txt（无 CoT，简洁口语）
_VOICE_PROMPT_NAME = "voice_prompt"


class RAGPipeline:
    """RAG 管线：加载检索器、管理 prompt 模板、构造上下文。

    用法:
        pipeline = RAGPipeline()
        pipeline.ensure_ready()            # 检查向量库是否就绪
        chunks = pipeline.retrieve(query)  # 检索相关文档
        prompt = pipeline.augment(query, chunks)  # 构造增强 prompt
    """

    def __init__(
        self,
        persist_dir: str = "./data/vectorstore",
        top_k: int = 3,
        min_score: float = 0.35,
    ):
        self._store: SearchStore = VectorStore(persist_dir=persist_dir)
        self._embedder: QueryEmbedder | None = None
        self._retriever: Retriever | None = None
        self._top_k = top_k
        self._min_score = min_score
        self._prompt_template = load_prompt(_RAG_PROMPT_NAME)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _ensure_embedder(self) -> QueryEmbedder:
        if self._embedder is None:
            self._embedder = Embedder()
            self._retriever = Retriever(self._embedder, self._store)
        return self._embedder

    @property
    def retriever(self) -> Retriever:
        self._ensure_embedder()
        assert self._retriever is not None
        return self._retriever

    @property
    def is_ready(self) -> bool:
        """向量库是否已建索引（包含数据）。"""
        return self._store.count > 0

    @property
    def chunk_count(self) -> int:
        return self._store.count

    # ------------------------------------------------------------------
    # 检索 + Prompt 构造
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        coarse_category: str | None = None,
        sub_category: str | None = None,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """检索与查询相关的文档 chunk（自动过滤低相似度结果）。"""
        if not self.is_ready:
            return []
        return self.retriever.retrieve(
            query,
            top_k=self._top_k,
            min_score=self._min_score,
            doc_ids=doc_ids,
            coarse_category=coarse_category,
            sub_category=sub_category,
            where=where,
        )

    def augment(self, query: str, chunks: list[Chunk], prompt_name: str | None = None) -> str:
        """将查询和检索结果拼接为增强后的 prompt。

        Args:
            query: 用户原始输入
            chunks: 检索到的文档片段
            prompt_name: 可选，指定 prompt 模板名（默认使用 _RAG_PROMPT_NAME）

        Returns:
            填充了上下文和问题的完整 prompt
        """
        if not chunks:
            return query

        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] {chunk['text']}\n"
                f"   (来源: {chunk['source']}, "
                f"相关度: {chunk['score']:.2f})"
            )

        context = "\n\n".join(parts)

        if prompt_name:
            template = load_prompt(prompt_name)
            return template.format(context=context, query=query)
        return self._prompt_template.format(context=context, query=query)
