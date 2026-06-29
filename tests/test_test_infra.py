"""Verify that test infrastructure fixtures behave correctly."""

from pathlib import Path

from PIL import Image

PRODUCTION_VECTORSTORE = Path("data/vectorstore").resolve()


class TestChromaFixtures:
    def test_text_dir_is_not_production(self, temp_chroma_text_dir: Path):
        """The text Chroma fixture must NOT point at the production vectorstore."""
        resolved = temp_chroma_text_dir.resolve()
        assert resolved != PRODUCTION_VECTORSTORE, (
            f"temp_chroma_text_dir resolved to {resolved}, "
            f"which is the production vectorstore at {PRODUCTION_VECTORSTORE}"
        )
        assert str(PRODUCTION_VECTORSTORE) not in str(resolved), (
            "temp_chroma_text_dir is inside the production vectorstore tree"
        )

    def test_visual_dir_is_not_production(self, temp_chroma_visual_dir: Path):
        """The visual Chroma fixture must NOT point at the production vectorstore."""
        resolved = temp_chroma_visual_dir.resolve()
        assert resolved != PRODUCTION_VECTORSTORE, (
            f"temp_chroma_visual_dir resolved to {resolved}, "
            f"which is the production vectorstore at {PRODUCTION_VECTORSTORE}"
        )


class TestSampleImage:
    def test_is_valid_jpeg(self, sample_image: Path):
        """sample_image must be openable by Pillow as a valid JPEG."""
        img = Image.open(sample_image)
        assert img.format == "JPEG"
        assert img.size == (10, 10)

    def test_sample_image_bytes_is_readable(self, sample_image_bytes: bytes):
        """sample_image_bytes must return non-empty bytes."""
        assert len(sample_image_bytes) > 0
        assert sample_image_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes
