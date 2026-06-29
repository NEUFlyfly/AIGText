"""Visual RAG asset readiness checker.

Reports whether required local assets exist without downloading anything.
Distinguishes user-provided assets (models, taxonomy, images, docs) from
generated artifacts (vector indexes).  Exit code is always 0 in report mode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from config.settings import settings


@dataclass
class AssetStatus:
    label: str
    exists: bool = False
    detail: str = ""


@dataclass
class AssetReport:
    user_provided: dict[str, AssetStatus] = field(default_factory=dict)
    generated: dict[str, AssetStatus] = field(default_factory=dict)

    def has_any_missing(self) -> bool:
        for status in self.user_provided.values():
            if not status.exists:
                return True
        for status in self.generated.values():
            if not status.exists:
                return True
        return False


def _dir_has_files(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    try:
        return any(item.is_file() for item in p.iterdir())
    except OSError:
        return False


def _check_dir(label: str, path: str) -> AssetStatus:
    if not Path(path).exists():
        return AssetStatus(label=label, exists=False, detail=f"path absent: {path}")
    if not Path(path).is_dir():
        return AssetStatus(label=label, exists=False, detail=f"not a directory: {path}")
    if not _dir_has_files(path):
        return AssetStatus(label=label, exists=False, detail=f"directory empty: {path}")
    return AssetStatus(label=label, exists=True, detail=path)


def _check_taxonomy_content(taxonomy_path: str) -> AssetStatus:
    if not Path(taxonomy_path).is_file():
        return AssetStatus(label="TAXONOMY", exists=False, detail=f"file absent: {taxonomy_path}")
    try:
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = cast(object, json.load(f))
    except (json.JSONDecodeError, OSError) as exc:
        return AssetStatus(label="TAXONOMY", exists=False, detail=f"invalid JSON: {exc}")
    if not isinstance(data, list):
        return AssetStatus(label="TAXONOMY", exists=False, detail="not a JSON array")
    taxonomy_entries = cast(list[object], data)
    entry_count = len(taxonomy_entries)
    if entry_count == 0:
        return AssetStatus(label="TAXONOMY", exists=False, detail="taxonomy array is empty")
    return AssetStatus(label="TAXONOMY", exists=True, detail=f"{taxonomy_path} ({entry_count} entries)")


def _load_taxonomy_entries(taxonomy_path: str) -> list[dict[str, str]]:
    if not Path(taxonomy_path).is_file():
        return []
    try:
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            data = cast(object, json.load(f))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    entries: list[dict[str, str]] = []
    for item in cast(list[object], data):
        if not isinstance(item, dict):
            continue
        entry = cast(dict[object, object], item)
        doc_id = entry.get("doc_id")
        image_dir = entry.get("image_dir")
        document_path = entry.get("document_path")
        if all(isinstance(v, str) and v for v in (doc_id, image_dir, document_path)):
            entries.append({
                "doc_id": cast(str, doc_id),
                "image_dir": cast(str, image_dir),
                "document_path": cast(str, document_path),
            })
    return entries


def _check_images(taxonomy_path: str) -> AssetStatus:
    entries = _load_taxonomy_entries(taxonomy_path)
    if not entries:
        return AssetStatus(label="IMAGES", exists=False, detail="no taxonomy entries to inspect")
    missing_dirs: list[str] = []
    missing_images: list[str] = []
    total_dirs = 0
    total_images = 0
    for entry in entries:
        total_dirs += 1
        image_dir_path = Path(entry["image_dir"])
        if not image_dir_path.is_dir():
            missing_dirs.append(entry["image_dir"])
            continue
        image_files = [
            p for p in image_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        total_images += len(image_files)
        if not image_files:
            missing_images.append(entry["image_dir"])

    detail_parts: list[str] = []
    if missing_dirs:
        detail_parts.append(f"missing dirs: {', '.join(missing_dirs[:3])}")
    if missing_images:
        detail_parts.append(f"empty dirs: {', '.join(missing_images[:3])}")
    if detail_parts:
        return AssetStatus(
            label="IMAGES",
            exists=False,
            detail="; ".join(detail_parts),
        )
    return AssetStatus(
        label="IMAGES",
        exists=True,
        detail=f"{total_dirs} dirs, {total_images} images",
    )


def _check_docs(taxonomy_path: str) -> AssetStatus:
    entries = _load_taxonomy_entries(taxonomy_path)
    if not entries:
        return AssetStatus(label="DOCS", exists=False, detail="no taxonomy entries to inspect")
    missing: list[str] = []
    present = 0
    for entry in entries:
        doc_path = Path(entry["document_path"])
        if doc_path.is_file():
            present += 1
        else:
            missing.append(entry["document_path"])
    if missing:
        return AssetStatus(
            label="DOCS",
            exists=False,
            detail=f"missing: {', '.join(missing[:3])}",
        )
    return AssetStatus(label="DOCS", exists=True, detail=f"{present} documents")


def check_assets(
    *,
    text_embedding_path: str | None = None,
    visual_embedding_path: str | None = None,
    taxonomy_path: str | None = None,
    chroma_text_path: str | None = None,
    chroma_visual_path: str | None = None,
) -> AssetReport:
    report = AssetReport()

    text_path = text_embedding_path or settings.TEXT_EMBEDDING_MODEL_PATH
    visual_path = visual_embedding_path or settings.VISUAL_EMBEDDING_MODEL_PATH
    taxonomy = taxonomy_path or os.path.join(settings.IOT_DOCUMENTS_DIR, "taxonomy.json")
    chroma_t = chroma_text_path or settings.CHROMA_TEXT_PATH
    chroma_v = chroma_visual_path or settings.CHROMA_VISUAL_PATH

    report.user_provided["TEXT_EMBEDDING_MODEL"] = _check_dir("TEXT_EMBEDDING_MODEL", text_path)
    report.user_provided["VISUAL_EMBEDDING_MODEL"] = _check_dir("VISUAL_EMBEDDING_MODEL", visual_path)
    report.user_provided["TAXONOMY"] = _check_taxonomy_content(taxonomy)
    report.user_provided["IMAGES"] = _check_images(taxonomy)
    report.user_provided["DOCS"] = _check_docs(taxonomy)

    report.generated["TEXT_INDEX"] = _check_dir("TEXT_INDEX", chroma_t)
    report.generated["VISUAL_INDEX"] = _check_dir("VISUAL_INDEX", chroma_v)

    return report


def format_report(report: AssetReport) -> str:
    lines: list[str] = []
    lines.append("=" * 58)
    lines.append("  Visual RAG Asset Readiness Report")
    lines.append("=" * 58)

    lines.append("")
    lines.append("  User-provided assets:")
    lines.append("  ----------------------")
    for key in ("TEXT_EMBEDDING_MODEL", "VISUAL_EMBEDDING_MODEL", "TAXONOMY", "IMAGES", "DOCS"):
        status = report.user_provided.get(key)
        if status is None:
            continue
        flag = "OK     " if status.exists else "MISSING"
        lines.append(f"  [{flag}] {status.label}")
        lines.append(f"          {status.detail}")

    lines.append("")
    lines.append("  Generated artifacts:")
    lines.append("  --------------------")
    for key in ("TEXT_INDEX", "VISUAL_INDEX"):
        status = report.generated.get(key)
        if status is None:
            continue
        flag = "OK     " if status.exists else "MISSING"
        lines.append(f"  [{flag}] {status.label}")
        lines.append(f"          {status.detail}")

    lines.append("")
    lines.append("=" * 58)
    return "\n".join(lines)


def main() -> None:
    report = check_assets()
    print(format_report(report))


if __name__ == "__main__":
    main()
