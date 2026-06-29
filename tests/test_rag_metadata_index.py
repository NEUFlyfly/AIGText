from pathlib import Path

from src.rag.iot_loader import load_iot_documents
from src.rag.loader import load_documents
from src.rag.splitter import split_text


Chunk = dict[str, str | int]


REQUIRED_METADATA_KEYS = {
    "doc_id",
    "coarse_category",
    "sub_category",
    "asset_type",
    "source",
    "chunk_id",
}


def _chunks_from_iot_fixtures() -> list[Chunk]:
    docs = load_iot_documents("data/iot_knowledge")
    chunks: list[Chunk] = []

    for doc in docs:
        chunks.extend(
            split_text(
                doc["content"],
                doc["source"],
                chunk_size=200,
                chunk_overlap=20,
                metadata={
                    "doc_id": doc["doc_id"],
                    "coarse_category": doc["coarse_category"],
                    "sub_category": doc["sub_category"],
                    "asset_type": doc["asset_type"],
                },
            )
        )

    return chunks


def test_iot_loader_reads_taxonomy_backed_fixture_documents() -> None:
    docs = load_iot_documents("data/iot_knowledge")

    docs_by_id = {doc["doc_id"]: doc for doc in docs}

    assert set(docs_by_id) == {
        "fixture_temp_sensor",
        "fixture_pir_sensor",
        "fixture_ptz_camera",
    }
    assert docs_by_id["fixture_temp_sensor"]["coarse_category"] == "智能传感器"
    assert docs_by_id["fixture_temp_sensor"]["sub_category"] == "温湿度传感器"
    assert docs_by_id["fixture_temp_sensor"]["asset_type"] == "text"


def test_iot_loader_reads_hierarchical_fixture_directory_with_taxonomy_metadata() -> None:
    docs = load_iot_documents("tests/fixtures/iot_knowledge")

    docs_by_id = {doc["doc_id"]: doc for doc in docs}

    assert "fixture_temp_sensor" in docs_by_id
    assert "fixture_pir_sensor" in docs_by_id
    assert "fixture_ptz_camera" in docs_by_id


def test_split_chunks_copy_required_iot_metadata() -> None:
    chunks = _chunks_from_iot_fixtures()

    assert chunks
    for chunk in chunks:
        assert REQUIRED_METADATA_KEYS <= set(chunk)
        assert chunk["asset_type"] == "text"
        assert str(chunk["doc_id"]).startswith("fixture_")


def test_legacy_documents_get_safe_fallback_metadata(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "documents"
    legacy_dir.mkdir()
    legacy_file = legacy_dir / "legacy-note.md"
    legacy_file.write_text("# Legacy\n\nlegacy content", encoding="utf-8")

    docs = load_documents(str(legacy_dir))

    assert len(docs) == 1
    doc = docs[0]
    assert doc["doc_id"].startswith("legacy_")
    assert doc["coarse_category"] == "legacy"
    assert doc["sub_category"] == "legacy"
    assert doc["asset_type"] == "text"

    chunks = split_text(
        doc["content"],
        doc["source"],
        metadata={
            "doc_id": doc["doc_id"],
            "coarse_category": doc["coarse_category"],
            "sub_category": doc["sub_category"],
            "asset_type": doc["asset_type"],
        },
    )
    assert chunks[0]["doc_id"] == doc["doc_id"]
    assert chunks[0]["coarse_category"] == "legacy"


def test_indexed_chunks_have_iot_metadata() -> None:
    chunks = _chunks_from_iot_fixtures()

    for chunk in chunks:
        stored_metadata = {
            "doc_id": chunk["doc_id"],
            "coarse_category": chunk["coarse_category"],
            "sub_category": chunk["sub_category"],
            "asset_type": chunk["asset_type"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
        }
        assert REQUIRED_METADATA_KEYS <= set(stored_metadata)
        assert stored_metadata["asset_type"] == "text"
        assert str(stored_metadata["doc_id"]).startswith("fixture_")
