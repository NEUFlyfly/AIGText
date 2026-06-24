"""
RAG 模块 — 文本切分器

职责:
  - 将长文档按语义边界切分成 chunk
  - 保留 chunk 重叠区域，防止关键信息被截断
  - 优先在段落/句子边界处分割，避免切断语义
"""

from typing import List, Dict


# 语义切分标记（按优先级排序）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", "；"]


def split_text(
    text: str,
    source: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> List[Dict]:
    """将长文本切分为带重叠的 chunk。

    Args:
        text: 原始文本
        source: 来源文件名
        chunk_size: 每个 chunk 的目标字符数 (中文约 2 chars/token)
        chunk_overlap: 相邻 chunk 的重叠字符数

    Returns:
        [{"chunk_id": 0, "text": "...", "source": "xxx.txt"}, ...]
    """
    text = text.strip()
    if not text:
        return []

    # 首先按段落（双换行）切分
    paragraphs = _split_by_separator(text, "\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: List[Dict] = []
    chunk_id = 0
    current_chunk = ""

    for para in paragraphs:
        # 单个段落过长时进一步切分
        if len(para) > chunk_size:
            # 先把当前积累的 chunk 输出
            if current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "source": source,
                })
                chunk_id += 1
                current_chunk = ""

            # 切分长段落
            sub_chunks = _split_long_paragraph(para, chunk_size, chunk_overlap)
            for sub in sub_chunks:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": sub,
                    "source": source,
                })
                chunk_id += 1
        else:
            # 尝试合并段落
            if current_chunk and len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += "\n\n" + para
            else:
                if current_chunk:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": current_chunk.strip(),
                        "source": source,
                    })
                    chunk_id += 1
                current_chunk = para

    # 最后一个 chunk
    if current_chunk.strip():
        chunks.append({
            "chunk_id": chunk_id,
            "text": current_chunk.strip(),
            "source": source,
        })

    return chunks


def _split_by_separator(text: str, separator: str) -> List[str]:
    """按分隔符切分文本。"""
    parts = text.split(separator)
    return [p for p in parts if p.strip()]


def _split_long_paragraph(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """将单个长段落切分为多个带重叠的片段。

    优先在标点符号处断开。
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # 在 char_size 附近寻找最佳断点
        best_break = end
        search_start = max(start + chunk_size // 2, start + chunk_size - 100)

        for sep in _SEPARATORS:
            pos = text.rfind(sep, search_start, min(end + 80, len(text)))
            if pos > search_start:
                best_break = pos + len(sep)
                break

        chunk_text = text[start:best_break].strip()
        if chunk_text:
            chunks.append(chunk_text)

        start = best_break - chunk_overlap
        if start <= 0 or start >= best_break:
            start = best_break

    return chunks
