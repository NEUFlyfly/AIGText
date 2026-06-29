import importlib
from pathlib import Path
from collections.abc import Callable
from typing import Protocol, cast

import pytest


class _AssetStatus(Protocol):
    exists: bool


class _AssetReport(Protocol):
    user_provided: dict[str, _AssetStatus]
    generated: dict[str, _AssetStatus]


asset_checker = importlib.import_module("src.rag.check_visual_rag_assets")
check_assets = cast(Callable[..., _AssetReport], getattr(asset_checker, "check_assets"))
format_report = cast(Callable[[_AssetReport], str], getattr(asset_checker, "format_report"))
main = cast(Callable[[], None], getattr(asset_checker, "main"))


def _write_taxonomy(root: Path) -> Path:
    document_path = root / "iot" / "sensor" / "document.md"
    image_dir = root / "iot" / "sensor" / "images"
    document_path.parent.mkdir(parents=True)
    image_dir.mkdir()
    _ = document_path.write_text("# Sensor\n\nFixture document", encoding="utf-8")
    _ = (image_dir / "sensor.png").write_bytes(b"fixture image")

    taxonomy_path = root / "taxonomy.json"
    _ = taxonomy_path.write_text(
        """
[
  {
    "doc_id": "sensor",
    "coarse_category": "sensor",
    "sub_category": "temperature",
    "aliases": ["temp"],
    "description": "fixture taxonomy entry",
    "image_dir": "%s",
    "document_path": "%s"
  }
]
""".strip()
        % (image_dir.as_posix(), document_path.as_posix()),
        encoding="utf-8",
    )
    return taxonomy_path


def test_missing_visual_model_reported(tmp_path: Path) -> None:
    taxonomy_path = _write_taxonomy(tmp_path)
    text_model_path = tmp_path / "models" / "embedding"
    text_model_path.mkdir(parents=True)
    _ = (text_model_path / "config.json").write_text("{}", encoding="utf-8")
    visual_model_path = tmp_path / "models" / "visual_embedding" / "clip-ViT-B-32"

    report = check_assets(
        text_embedding_path=str(text_model_path),
        visual_embedding_path=str(visual_model_path),
        taxonomy_path=str(taxonomy_path),
        chroma_text_path=str(tmp_path / "vectorstore"),
        chroma_visual_path=str(tmp_path / "visual_vectorstore"),
    )
    output = format_report(report)

    assert report.user_provided["VISUAL_EMBEDDING_MODEL"].exists is False
    assert "VISUAL_EMBEDDING_MODEL" in output
    assert "MISSING" in output
    assert "models/visual_embedding/clip-ViT-B-32" in output.replace("\\", "/")


def test_generated_indexes_reported_separately(tmp_path: Path) -> None:
    taxonomy_path = _write_taxonomy(tmp_path)

    report = check_assets(
        text_embedding_path=str(tmp_path / "missing_text_model"),
        visual_embedding_path=str(tmp_path / "missing_visual_model"),
        taxonomy_path=str(taxonomy_path),
        chroma_text_path=str(tmp_path / "vectorstore"),
        chroma_visual_path=str(tmp_path / "visual_vectorstore"),
    )
    output = format_report(report)

    assert "TEXT_INDEX" not in report.user_provided
    assert "VISUAL_INDEX" not in report.user_provided
    assert set(report.generated) == {"TEXT_INDEX", "VISUAL_INDEX"}
    assert output.index("Generated artifacts:") < output.index("TEXT_INDEX")
    assert output.index("Generated artifacts:") < output.index("VISUAL_INDEX")


def test_main_report_only_exits_zero_when_assets_are_missing(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    output = capsys.readouterr().out

    assert "Visual RAG Asset Readiness Report" in output
    assert "MISSING" in output
