"""Index IoT reference images into the separate visual vector store."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from PIL import Image

from config.settings import settings

from .iot_loader import load_iot_documents
from .visual_embedder import VisualEmbedder
from .visual_store import ImageMetadata, InMemoryVisualStore, VisualVectorStore


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FIXTURE_IMAGE_COLORS = {
    "fixture_temp_sensor": (24, 176, 96),
    "fixture_pir_sensor": (224, 128, 32),
    "fixture_ptz_camera": (64, 128, 224),
}

VisualImageRecord: TypeAlias = dict[str, str]


class VisualStoreWriter(Protocol):
    def clear(self) -> None:
        ...

    def upsert_images(
        self,
        image_metadatas: list[ImageMetadata],
        embeddings: list[list[float]],
    ) -> None:
        ...

    @property
    def count(self) -> int:
        ...


class VisualImageEmbedder(Protocol):
    def embed_images(self, image_inputs: Iterable[object]) -> list[list[float]]:
        ...


@dataclass
class VisualIndexReport:
    indexed_count: int = 0
    rejected_count: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VisualIndexCliArgs:
    fixtures: bool
    iot_dir: str | None
    taxonomy: str | None
    persist_dir: str


def build_visual_index(
    *,
    iot_documents_dir: str = settings.IOT_DOCUMENTS_DIR,
    taxonomy_path: str | None = None,
    persist_dir: str = settings.CHROMA_VISUAL_PATH,
    fixture_mode: bool = False,
    store: VisualStoreWriter | None = None,
    embedder: VisualImageEmbedder | None = None,
) -> VisualIndexReport:
    if fixture_mode:
        _ = ensure_fixture_images(taxonomy_path or _default_taxonomy_path())

    records, errors = collect_visual_image_records(
        iot_documents_dir=iot_documents_dir,
        taxonomy_path=taxonomy_path,
    )
    report = VisualIndexReport(rejected_count=len(errors), errors=errors)

    if not records:
        return report

    active_embedder = embedder or (
        _DeterministicFixtureEmbedder() if fixture_mode else VisualEmbedder()
    )
    active_store = store or (
        InMemoryVisualStore() if fixture_mode else VisualVectorStore(persist_dir=persist_dir)
    )

    image_paths = [record["image_path"] for record in records]
    embeddings = active_embedder.embed_images(image_paths)
    metadatas = [image_metadata_from_record(record) for record in records]

    active_store.clear()
    active_store.upsert_images(metadatas, embeddings)
    report.indexed_count = len(metadatas)
    return report


def collect_visual_image_records(
    *,
    iot_documents_dir: str = settings.IOT_DOCUMENTS_DIR,
    taxonomy_path: str | None = None,
) -> tuple[list[VisualImageRecord], list[str]]:
    taxonomy_entries = _load_taxonomy(taxonomy_path or _default_taxonomy_path())
    text_document_ids = _load_text_document_ids(iot_documents_dir, taxonomy_entries)
    records: list[VisualImageRecord] = []
    errors: list[str] = []

    for entry in taxonomy_entries:
        doc_id = entry["doc_id"]
        document_path = entry["document_path"]
        if doc_id not in text_document_ids:
            errors.append(f"Missing matching text document for doc_id={doc_id}")
            continue
        if not Path(document_path).is_file():
            errors.append(f"Missing document_path for doc_id={doc_id}: {document_path}")
            continue

        image_dir = Path(entry["image_dir"])
        if not image_dir.is_dir():
            errors.append(f"Missing image_dir for doc_id={doc_id}: {image_dir}")
            continue

        image_paths = _iter_image_paths(image_dir)
        if not image_paths:
            errors.append(f"No reference images found for doc_id={doc_id}: {image_dir}")
            continue

        for image_path in image_paths:
            image_path_text = _normalize_relative_path(str(image_path))
            records.append({
                "doc_id": doc_id,
                "coarse_category": entry["coarse_category"],
                "sub_category": entry["sub_category"],
                "asset_type": "image",
                "image_id": _image_id(doc_id, image_path_text),
                "image_path": image_path_text,
            })

    return records, errors


def _load_text_document_ids(
    iot_documents_dir: str,
    taxonomy_entries: list[dict[str, str]],
) -> set[str]:
    text_document_ids = {
        document["doc_id"]
        for document in load_iot_documents(iot_documents_dir)
    }
    for entry in taxonomy_entries:
        content = _read_text_document(entry["document_path"])
        if content is not None:
            text_document_ids.add(entry["doc_id"])
    return text_document_ids


def image_metadata_from_record(record: VisualImageRecord) -> ImageMetadata:
    return {
        "doc_id": record["doc_id"],
        "coarse_category": record["coarse_category"],
        "sub_category": record["sub_category"],
        "asset_type": "image",
        "image_id": record["image_id"],
        "image_path": record["image_path"],
    }


def ensure_fixture_images(taxonomy_path: str | Path | None = None) -> int:
    created_or_existing = 0
    for entry in _load_taxonomy(str(taxonomy_path or _default_taxonomy_path())):
        doc_id = entry["doc_id"]
        if doc_id not in FIXTURE_IMAGE_COLORS:
            continue

        image_dir = Path(entry["image_dir"])
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{doc_id}.png"
        if not image_path.exists():
            image = Image.new("RGB", (12, 12), color=FIXTURE_IMAGE_COLORS[doc_id])
            image.save(image_path, format="PNG")
        created_or_existing += 1

    return created_or_existing


def _read_text_document(document_path: str) -> str | None:
    try:
        content = Path(document_path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    if not content.strip():
        return None
    return content


class _DeterministicFixtureEmbedder:
    def embed_images(self, image_inputs: Iterable[object]) -> list[list[float]]:
        return [_stable_unit_vector(str(image_input)) for image_input in image_inputs]


def _stable_unit_vector(value: str) -> list[float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    vector = [float(digest[index] + 1) for index in range(8)]
    norm = math.sqrt(sum(component * component for component in vector))
    return [component / norm for component in vector]


def _load_taxonomy(taxonomy_path: str) -> list[dict[str, str]]:
    try:
        with open(taxonomy_path, "r", encoding="utf-8") as taxonomy_file:
            raw_data = cast(object, json.load(taxonomy_file))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(raw_data, list):
        return []

    entries: list[dict[str, str]] = []
    for raw_entry in cast(list[object], raw_data):
        if not isinstance(raw_entry, dict):
            continue
        entry = cast(dict[object, object], raw_entry)
        required_values = {
            "doc_id": entry.get("doc_id"),
            "coarse_category": entry.get("coarse_category"),
            "sub_category": entry.get("sub_category"),
            "image_dir": entry.get("image_dir"),
            "document_path": entry.get("document_path"),
        }
        if all(isinstance(value, str) and value for value in required_values.values()):
            entries.append(cast(dict[str, str], required_values))
    return entries


def _iter_image_paths(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _image_id(doc_id: str, image_path: str) -> str:
    path_hash = hashlib.sha1(image_path.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}__{path_hash}"


def _normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/")


def _default_taxonomy_path() -> str:
    return os.path.join(settings.IOT_DOCUMENTS_DIR, "taxonomy.json")


def _parse_args(argv: Sequence[str] | None = None) -> VisualIndexCliArgs:
    parser = argparse.ArgumentParser(description="Build the Visual RAG image vector index.")
    _ = parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Index deterministic fixture images with fake offline embeddings.",
    )
    _ = parser.add_argument("--iot-dir", default=None, help="IoT knowledge directory to validate text docs.")
    _ = parser.add_argument("--taxonomy", default=None, help="Path to taxonomy.json.")
    _ = parser.add_argument("--persist-dir", default=settings.CHROMA_VISUAL_PATH)
    raw_args = cast(dict[str, object], vars(parser.parse_args(argv)))
    iot_dir = raw_args.get("iot_dir")
    taxonomy = raw_args.get("taxonomy")
    persist_dir = raw_args.get("persist_dir")
    return VisualIndexCliArgs(
        fixtures=raw_args.get("fixtures") is True,
        iot_dir=iot_dir if isinstance(iot_dir, str) else None,
        taxonomy=taxonomy if isinstance(taxonomy, str) else None,
        persist_dir=persist_dir if isinstance(persist_dir, str) else settings.CHROMA_VISUAL_PATH,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    iot_documents_dir = args.iot_dir or (
        "tests/fixtures/iot_knowledge" if args.fixtures else settings.IOT_DOCUMENTS_DIR
    )

    report = build_visual_index(
        iot_documents_dir=iot_documents_dir,
        taxonomy_path=args.taxonomy,
        persist_dir=args.persist_dir,
        fixture_mode=args.fixtures,
    )

    for error in report.errors:
        print(f"Rejected image asset: {error}")
    print(f"Indexed images: {report.indexed_count}")
    if report.indexed_count <= 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
