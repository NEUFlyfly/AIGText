import os
import subprocess
import sys
from math import isclose
from pathlib import Path

import pytest
from PIL import Image

from src.rag.visual_embedder import VisualEmbedder


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self, vector: object) -> None:
        self.vector: object = vector
        self.images: list[Image.Image] = []

    def encode(
        self,
        images: list[Image.Image],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object:
        assert normalize_embeddings is True
        assert show_progress_bar is False
        self.images.extend(images)
        if len(images) == 1:
            return self.vector
        return [self.vector for _ in images]


class TensorLike:
    def __init__(self, values: list[list[float]]) -> None:
        self._values: list[list[float]] = values

    def detach(self) -> "TensorLike":
        return self

    def cpu(self) -> "TensorLike":
        return self

    def numpy(self) -> "TensorLike":
        return self

    def tolist(self) -> list[list[float]]:
        return self._values


def test_visual_embedder_import_is_lazy() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.rag.visual_embedder import VisualEmbedder; print('ok')",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_invalid_image_bytes_raise_clear_error() -> None:
    embedder = VisualEmbedder(model=FakeModel([1.0, 0.0]))

    with pytest.raises(ValueError, match="Invalid image bytes"):
        _ = embedder.embed_image(b"not an image")


def test_missing_image_path_raises_clear_error(tmp_path: Path) -> None:
    embedder = VisualEmbedder(model=FakeModel([1.0, 0.0]))

    with pytest.raises(FileNotFoundError, match="Image path does not exist"):
        _ = embedder.embed_image(tmp_path / "missing.jpg")


def test_corrupt_image_file_raises_clear_error(tmp_path: Path) -> None:
    image_path = tmp_path / "corrupt.jpg"
    _ = image_path.write_bytes(b"not an image")
    embedder = VisualEmbedder(model=FakeModel([1.0, 0.0]))

    with pytest.raises(ValueError, match="Invalid image file"):
        _ = embedder.embed_image(image_path)


def test_unsupported_input_type_raises_clear_error() -> None:
    embedder = VisualEmbedder(model=FakeModel([1.0, 0.0]))

    with pytest.raises(TypeError, match="Unsupported image input type"):
        _ = embedder.embed_image(123)


def assert_rgb_input_is_normalized(image_input: object) -> None:
    fake_model = FakeModel([3.0, 4.0])
    embedder = VisualEmbedder(model=fake_model)

    vector = embedder.embed_image(image_input)

    assert_vector_close(vector, [0.6, 0.8])
    assert len(fake_model.images) == 1
    assert fake_model.images[0].mode == "RGB"
    assert fake_model.images[0].size == (10, 10)


def assert_vector_close(vector: list[float], expected: list[float]) -> None:
    assert len(vector) == len(expected)
    for actual_value, expected_value in zip(vector, expected):
        assert isclose(actual_value, expected_value)


def test_path_input_is_decoded_as_rgb_and_vector_is_normalized(sample_image: Path) -> None:
    assert_rgb_input_is_normalized(sample_image)


def test_bytes_input_is_decoded_as_rgb_and_vector_is_normalized(sample_image_bytes: bytes) -> None:
    assert_rgb_input_is_normalized(sample_image_bytes)


def test_pil_input_is_converted_to_rgb_and_vector_is_normalized(sample_image: Path) -> None:
    with Image.open(sample_image) as image:
        assert_rgb_input_is_normalized(image)


def test_fake_encoder_batch_output_is_normalized_for_tensor_like_values(
    sample_image: Path,
    sample_image_bytes: bytes,
) -> None:
    def fake_encoder(images: list[Image.Image]) -> object:
        assert len(images) == 2
        return TensorLike([[3.0, 4.0], [5.0, 12.0]])

    embedder = VisualEmbedder(encoder=fake_encoder)

    vectors = embedder.embed_images([sample_image, sample_image_bytes])

    assert len(vectors) == 2
    assert_vector_close(vectors[0], [0.6, 0.8])
    assert_vector_close(vectors[1], [5.0 / 13.0, 12.0 / 13.0])
