"""Bridge visual Top-K candidates into metadata-filtered text RAG."""

from __future__ import annotations

import os
from typing import Protocol, TypeAlias, cast

from config.settings import settings


from .pipeline import RAGPipeline
from .retriever import MetadataFilter, SearchResult
from .visual_retriever import VisualCandidate, VisualRetrievalResult, VisualRetriever


VisualRAGPayload: TypeAlias = dict[str, object]

_PROMPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "prompt",
)
_VISUAL_RAG_PROMPT_FILE = os.path.join(_PROMPT_DIR, "visual_rag_prompt.txt")


class VisualTopKRetriever(Protocol):
    def retrieve(self, image_bytes: bytes) -> VisualRetrievalResult:
        ...


class FilteredTextRAG(Protocol):
    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        coarse_category: str | None = None,
        sub_category: str | None = None,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        ...


class VisualRAGPipeline:
    """Visual-to-text RAG bridge.

    The pipeline does not call an LLM. It returns the filtered retrieval payload and
    augmented prompt for the API layer to pass to a language model later.
    """

    def __init__(
        self,
        *,
        visual_retriever: VisualTopKRetriever | None = None,
        text_pipeline: FilteredTextRAG | None = None,
    ) -> None:
        self._visual_retriever: VisualTopKRetriever = visual_retriever or VisualRetriever()
        self._text_pipeline: FilteredTextRAG = text_pipeline or RAGPipeline(
            persist_dir=settings.CHROMA_TEXT_PATH,
            top_k=settings.TEXT_TOP_K,
        )
        self._prompt_template: str = self._load_prompt_template()

    def run(self, image_bytes: bytes, question: str) -> VisualRAGPayload:
        visual_result = self._visual_retriever.retrieve(image_bytes)
        visual_candidates = _candidate_list(visual_result.get("visual_candidates"))

        if not visual_candidates:
            return self._payload(
                visual_result=visual_result,
                visual_candidates=[],
                retrieved_chunks=[],
                augmented_prompt="",
                status=_empty_visual_status(visual_result),
            )

        doc_ids = _candidate_doc_ids(visual_candidates)
        retrieved_chunks = self._text_pipeline.retrieve(question, doc_ids=doc_ids)
        if not retrieved_chunks:
            return self._payload(
                visual_result=visual_result,
                visual_candidates=visual_candidates,
                retrieved_chunks=[],
                augmented_prompt=self.augment(question, visual_candidates, []),
                status="NO_TEXT_CHUNKS",
            )

        return self._payload(
            visual_result=visual_result,
            visual_candidates=visual_candidates,
            retrieved_chunks=retrieved_chunks,
            augmented_prompt=self.augment(question, visual_candidates, retrieved_chunks),
            status="OK",
        )

    def query(self, image_bytes: bytes, question: str) -> VisualRAGPayload:
        return self.run(image_bytes, question)

    def augment(
        self,
        question: str,
        visual_candidates: list[VisualCandidate],
        chunks: list[SearchResult],
    ) -> str:
        candidate_summary = _format_candidate_summary(visual_candidates)
        context = _format_text_sources(chunks)
        if not context:
            context = "未检索到可引用的候选文档文本片段。"
        return self._prompt_template.format(
            candidate_summary=candidate_summary,
            context=context,
            query=question,
        )

    def _payload(
        self,
        *,
        visual_result: VisualRetrievalResult,
        visual_candidates: list[VisualCandidate],
        retrieved_chunks: list[SearchResult],
        augmented_prompt: str,
        status: str,
    ) -> VisualRAGPayload:
        return {
            "coarse_category": _string_value(visual_result.get("coarse_category")),
            "coarse_confidence": _float_value(visual_result.get("coarse_confidence")),
            "coarse_status": _string_value(visual_result.get("coarse_status")),
            "visual_candidates": visual_candidates,
            "retrieved_chunks": retrieved_chunks,
            "augmented_prompt": augmented_prompt,
            "status": status,
        }

    def _load_prompt_template(self) -> str:
        try:
            with open(_VISUAL_RAG_PROMPT_FILE, "r", encoding="utf-8") as prompt_file:
                return prompt_file.read()
        except FileNotFoundError:
            return (
                "你是一个物联网设备识别与说明助手。\n\n"
                "视觉候选设备：\n{candidate_summary}\n\n"
                "参考资料：\n{context}\n\n"
                "用户问题：{query}\n\n"
                "请基于参考资料回答，并使用 [1]、[2] 这样的来源编号。"
            )


def _candidate_list(value: object) -> list[VisualCandidate]:
    if not isinstance(value, list):
        return []

    candidates: list[VisualCandidate] = []
    items = cast(list[object], value)
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate: VisualCandidate = {}
        for key, raw_value in cast(dict[object, object], item).items():
            if isinstance(key, str) and isinstance(raw_value, (str, int, float)):
                candidate[key] = raw_value
        if candidate.get("doc_id"):
            candidates.append(candidate)
    return candidates


def _candidate_doc_ids(candidates: list[VisualCandidate]) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        doc_id = str(candidate.get("doc_id", ""))
        if not doc_id or doc_id in seen:
            continue
        doc_ids.append(doc_id)
        seen.add(doc_id)
    return doc_ids


def _format_candidate_summary(candidates: list[VisualCandidate]) -> str:
    if not candidates:
        return "无视觉候选设备。"

    parts: list[str] = []
    for index, candidate in enumerate(candidates, 1):
        score = _float_value(candidate.get("score"))
        matched_image_count = int(_float_value(candidate.get("matched_image_count")))
        parts.append("".join([
            f"{index}. doc_id={_string_value(candidate.get('doc_id'))}; ",
            f"粗类别={_string_value(candidate.get('coarse_category'))}; ",
            f"子类别={_string_value(candidate.get('sub_category'))}; ",
            f"视觉分数={score:.2f}; ",
            f"证据图片={_string_value(candidate.get('evidence_image_id'))}; ",
            f"匹配图片数={matched_image_count}",
        ]))
    return "\n".join(parts)


def _format_text_sources(chunks: list[SearchResult]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        parts.append("".join([
            f"[{index}] {_string_value(chunk.get('text'))}\n",
            f"   (来源: {_string_value(chunk.get('source'))}, ",
            f"相关度: {_float_value(chunk.get('score')):.2f})",
        ]))
    return "\n\n".join(parts)


def _empty_visual_status(visual_result: VisualRetrievalResult) -> str:
    status = _string_value(visual_result.get("status"))
    if status and status != "OK":
        return status
    return "NO_VISUAL_MATCH"


def _string_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
