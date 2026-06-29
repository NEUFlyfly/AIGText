"""
RAG 模块 — 文档加载器

职责:
  - 从 data/documents/ 目录加载 .txt / .md 文件
  - 提取纯文本内容
  - 保留文档来源元信息 (文件名、路径)
"""

import hashlib
import os
from typing import List, Dict


def _normalize_source(path: str) -> str:
    return path.replace("\\", "/")


def _legacy_doc_id(source: str) -> str:
    source_hash = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"legacy_{source_hash}"


def load_documents(directory: str) -> List[Dict[str, str]]:
    """加载指定目录下的所有 .txt 和 .md 文件。

    Args:
        directory: 文档目录路径

    Returns:
        [{"source": "filename.txt", "content": "全文内容...", ...metadata}, ...]
    """
    docs: List[Dict[str, str]] = []

    if not os.path.isdir(directory):
        return docs

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, OSError):
            continue

        if content.strip():
            source = _normalize_source(filename)
            docs.append({
                "doc_id": _legacy_doc_id(source),
                "coarse_category": "legacy",
                "sub_category": "legacy",
                "asset_type": "text",
                "source": source,
                "content": content,
            })

    return docs
