import json
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from src.rag.visual_index import build_visual_index, ensure_fixture_images
from src.rag.visual_store import InMemoryVisualStore, VisualVectorStore


REQUIRED_IMAGE_METADATA = {
    "doc_id",
    "coarse_category",
    "sub_category",
    "asset_type",
    "image_id",
    "image_path",
}


class FakeVisualEmbedder:
    def embed_images(self, image_inputs: Iterable[object]) -> list[list[float]]:
        paths = list(image_inputs)
        return [[1.0, 0.0, 0.0] for _ in paths]


def test_fixture_visual_index_writes_image_metadata() -> None:
    fixture_count = ensure_fixture_images()
    store = InMemoryVisualStore()

    report = build_visual_index(
        iot_documents_dir="tests/fixtures/iot_knowledge",
        fixture_mode=True,
        store=store,
        embedder=FakeVisualEmbedder(),
    )

    assert fixture_count > 0
    assert report.indexed_count >= fixture_count
    assert report.errors == []
    assert store.count == report.indexed_count
    assert VisualVectorStore.COLLECTION_NAME != "aigtext_docs"

    indexed_doc_ids = {metadata["doc_id"] for metadata in store.image_metadatas}
    assert indexed_doc_ids == {
        "fixture_temp_sensor",
        "fixture_pir_sensor",
        "fixture_ptz_camera",
    }

    for metadata in store.image_metadatas:
        assert REQUIRED_IMAGE_METADATA <= set(metadata)
        assert metadata["asset_type"] == "image"
        assert metadata["image_id"].startswith(metadata["doc_id"])
        assert Path(metadata["image_path"]).is_file()
        assert metadata["image_path"].endswith(".png")


def test_image_without_document_rejected(tmp_path: Path) -> None:
    image_dir = tmp_path / "orphan" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "orphan.png"
    Image.new("RGB", (8, 8), color=(200, 10, 10)).save(image_path, format="PNG")

    document_path = tmp_path / "orphan" / "document.md"
    taxonomy_path = tmp_path / "taxonomy.json"
    _ = taxonomy_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "missing_doc_id",
                    "coarse_category": "智能传感器",
                    "sub_category": "缺失文档设备",
                    "image_dir": str(image_dir),
                    "document_path": str(document_path),
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    store = InMemoryVisualStore()

    report = build_visual_index(
        iot_documents_dir=str(tmp_path),
        taxonomy_path=str(taxonomy_path),
        store=store,
        embedder=FakeVisualEmbedder(),
    )

    assert report.indexed_count == 0
    assert store.count == 0
    assert report.rejected_count == 1
    assert report.errors == ["Missing matching text document for doc_id=missing_doc_id"]
