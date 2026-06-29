"""
RAG 模块 — IoT 知识加载器

职责:
  - 从 data/iot_knowledge/ 层级目录加载 IoT 子类文档
  - 提取 document.md 文本内容
  - 保留 doc_id、粗分类、子分类和来源路径元信息
"""

import json
import os
from typing import cast


def _normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/")


def _load_taxonomy_entries(taxonomy_path: str) -> list[dict[str, str]]:
    taxonomy_entries: list[dict[str, str]] = []

    if not os.path.isfile(taxonomy_path):
        return taxonomy_entries

    try:
        with open(taxonomy_path, "r", encoding="utf-8") as taxonomy_file:
            taxonomy_data = cast(object, json.load(taxonomy_file))
    except (json.JSONDecodeError, OSError):
        return taxonomy_entries

    if not isinstance(taxonomy_data, list):
        return taxonomy_entries

    for taxonomy_entry in cast(list[object], taxonomy_data):
        if not isinstance(taxonomy_entry, dict):
            continue

        taxonomy_mapping = cast(dict[object, object], taxonomy_entry)
        doc_id = taxonomy_mapping.get("doc_id")
        coarse_category = taxonomy_mapping.get("coarse_category")
        sub_category = taxonomy_mapping.get("sub_category")
        document_path = taxonomy_mapping.get("document_path")

        if (
            isinstance(doc_id, str)
            and isinstance(coarse_category, str)
            and isinstance(sub_category, str)
            and isinstance(document_path, str)
        ):
            taxonomy_entries.append({
                "doc_id": doc_id,
                "coarse_category": coarse_category,
                "sub_category": sub_category,
                "document_path": _normalize_relative_path(document_path),
            })

    return taxonomy_entries


def _candidate_taxonomy_paths(directory: str) -> list[str]:
    paths = [os.path.join(directory, "taxonomy.json")]
    default_taxonomy_path = os.path.join("data", "iot_knowledge", "taxonomy.json")
    if os.path.normpath(paths[0]) != os.path.normpath(default_taxonomy_path):
        paths.append(default_taxonomy_path)
    return paths


def _load_taxonomy_by_document_path(directory: str) -> dict[str, dict[str, str]]:
    taxonomy_by_document_path: dict[str, dict[str, str]] = {}

    for taxonomy_path in _candidate_taxonomy_paths(directory):
        for taxonomy_entry in _load_taxonomy_entries(taxonomy_path):
            document_path = taxonomy_entry["document_path"]
            taxonomy_by_document_path[document_path] = taxonomy_entry

    return taxonomy_by_document_path


def _read_document(document_path: str) -> str | None:
    try:
        with open(document_path, "r", encoding="utf-8") as document_file:
            content = document_file.read()
    except (UnicodeDecodeError, OSError):
        return None

    if not content.strip():
        return None
    return content


def _append_document(
    docs: list[dict[str, str]],
    seen_sources: set[str],
    source: str,
    content: str,
    doc_id: str,
    coarse_category: str,
    sub_category: str,
) -> None:
    if source in seen_sources:
        return

    docs.append(
        {
            "doc_id": doc_id,
            "coarse_category": coarse_category,
            "sub_category": sub_category,
            "asset_type": "text",
            "source": source,
            "content": content,
        }
    )
    seen_sources.add(source)


def load_iot_documents(directory: str = "data/iot_knowledge") -> list[dict[str, str]]:
    """加载 IoT 知识目录下的所有 document.md 文件。

    Args:
        directory: IoT 知识根目录路径

    Returns:
        [{"doc_id": "...", "coarse_category": "...", "sub_category": "...", "source": "...", "content": "..."}, ...]
    """
    docs: list[dict[str, str]] = []

    if not os.path.isdir(directory):
        return docs

    seen_sources: set[str] = set()
    taxonomy_by_document_path = _load_taxonomy_by_document_path(directory)

    local_taxonomy_path = os.path.join(directory, "taxonomy.json")
    for taxonomy_entry in _load_taxonomy_entries(local_taxonomy_path):
        document_path = taxonomy_entry["document_path"]
        content = _read_document(document_path)
        if content is None:
            continue
        _append_document(
            docs,
            seen_sources,
            document_path,
            content,
            taxonomy_entry["doc_id"],
            taxonomy_entry["coarse_category"],
            taxonomy_entry["sub_category"],
        )

    for coarse_category in sorted(os.listdir(directory)):
        coarse_category_path = os.path.join(directory, coarse_category)
        if not os.path.isdir(coarse_category_path):
            continue

        for sub_category in sorted(os.listdir(coarse_category_path)):
            sub_category_path = os.path.join(coarse_category_path, sub_category)
            if not os.path.isdir(sub_category_path):
                continue

            document_path = os.path.join(sub_category_path, "document.md")
            if not os.path.isfile(document_path):
                continue

            content = _read_document(document_path)
            if content is None:
                continue

            source = _normalize_relative_path(os.path.relpath(document_path))
            taxonomy_entry = taxonomy_by_document_path.get(source, {})
            doc_id = taxonomy_entry.get("doc_id", f"{coarse_category}/{sub_category}")
            metadata_coarse_category = taxonomy_entry.get("coarse_category", coarse_category)
            metadata_sub_category = taxonomy_entry.get("sub_category", sub_category)

            _append_document(
                docs,
                seen_sources,
                source,
                content,
                doc_id,
                metadata_coarse_category,
                metadata_sub_category,
            )

    return docs
