"""
RAG 模块 — 文档加载器

职责:
  - 从 data/documents/ 目录加载 .txt / .md 文件
  - 提取纯文本内容
  - 保留文档来源元信息 (文件名、路径)
"""

import os
from typing import List, Dict


def load_documents(directory: str) -> List[Dict[str, str]]:
    """加载指定目录下的所有 .txt 和 .md 文件。

    Args:
        directory: 文档目录路径

    Returns:
        [{"source": "filename.txt", "content": "全文内容..."}, ...]
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
            docs.append({"source": filename, "content": content})

    return docs
