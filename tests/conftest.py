"""Pytest fixtures for AIGText tests.

All Chroma fixtures use tmp_path (pytest temporary directories),
NEVER the production data/vectorstore or data/visual_vectorstore.
"""

from collections.abc import Iterator
from pathlib import Path
import sys

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def temp_chroma_text_dir(tmp_path: Path) -> Iterator[Path]:
    """Yield a temporary directory intended for a text ChromaDB instance."""
    chroma_dir = tmp_path / "chroma_text"
    chroma_dir.mkdir()
    yield chroma_dir


@pytest.fixture
def temp_chroma_visual_dir(tmp_path: Path) -> Iterator[Path]:
    """Yield a temporary directory intended for a visual ChromaDB instance."""
    chroma_dir = tmp_path / "chroma_visual"
    chroma_dir.mkdir()
    yield chroma_dir


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create and return a tiny valid JPEG file (10x10 solid green pixel)."""
    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (10, 10), color=(0, 255, 0))
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def sample_image_bytes(sample_image: Path) -> bytes:
    """Read sample_image and return its raw bytes."""
    return sample_image.read_bytes()
