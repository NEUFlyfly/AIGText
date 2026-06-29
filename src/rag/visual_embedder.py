"""Lazy local visual embedding wrapper for Visual RAG."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Iterable, Sequence
from io import BytesIO
from pathlib import Path
from typing import Callable, Protocol, TypeGuard, cast

from PIL import Image, UnidentifiedImageError

from config.settings import settings


ImageInput = bytes | str | Path | Image.Image
ImageEncoder = Callable[[list[Image.Image]], object]


class ImageEmbeddingModel(Protocol):
    def encode(
        self,
        images: list[Image.Image],
        *,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object: ...


class SentenceTransformerFactory(Protocol):
    def __call__(self, model_name_or_path: str, *, local_files_only: bool) -> ImageEmbeddingModel: ...


class VisualEmbedder:
    """Image embedding adapter that loads the visual model only on first use."""

    def __init__(
        self,
        model_name: str | None = None,
        model_path: str | Path | None = None,
        model: ImageEmbeddingModel | None = None,
        encoder: ImageEncoder | None = None,
    ) -> None:
        self._model_name: str = model_name or settings.VISUAL_EMBEDDING_MODEL_NAME
        self._model_path: Path = Path(model_path or settings.VISUAL_EMBEDDING_MODEL_PATH)
        self._model: ImageEmbeddingModel | None = model
        self._encoder: ImageEncoder | None = encoder

    @property
    def model(self) -> ImageEmbeddingModel:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> ImageEmbeddingModel:
        if not self._model_path.exists():
            message = (
                f"Visual embedding model path does not exist: {self._model_path}. "
                f"Place {self._model_name} weights there or pass a test encoder/model."
            )
            raise FileNotFoundError(
                message
            )

        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            message = (
                "sentence-transformers is required for visual embeddings. Install project "
                "requirements or pass a test encoder/model."
            )
            raise ImportError(
                message
            ) from exc

        sentence_transformer = cast(
            SentenceTransformerFactory,
            getattr(module, "SentenceTransformer"),
        )

        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return sentence_transformer(str(self._model_path), local_files_only=True)
        finally:
            if previous_offline is None:
                _ = os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline

    def embed_image(self, image_input: object) -> list[float]:
        images = [self._normalize_image_input(image_input)]
        encoded = self._encode_images(images)
        vectors = self._vectors_from_encoded(encoded, expected_count=1)
        return self._normalize_vector(vectors[0])

    def embed_images(self, image_inputs: Iterable[object]) -> list[list[float]]:
        images = [self._normalize_image_input(image_input) for image_input in image_inputs]
        if not images:
            return []
        encoded = self._encode_images(images)
        vectors = self._vectors_from_encoded(encoded, expected_count=len(images))
        return [self._normalize_vector(vector) for vector in vectors]

    def _encode_images(self, images: list[Image.Image]) -> object:
        if self._encoder is not None:
            return self._encoder(images)
        return self.model.encode(
            images,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _normalize_image_input(self, image_input: object) -> Image.Image:
        if isinstance(image_input, bytes):
            if not image_input:
                raise ValueError("Invalid image bytes: input is empty")
            try:
                with Image.open(BytesIO(image_input)) as image:
                    _ = image.load()
                    return image.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ValueError("Invalid image bytes: could not decode image") from exc

        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            try:
                with Image.open(image_path) as image:
                    _ = image.load()
                    return image.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ValueError(f"Invalid image file: {image_path}") from exc

        if isinstance(image_input, Image.Image):
            try:
                _ = image_input.load()
                return image_input.convert("RGB")
            except (OSError, ValueError) as exc:
                raise ValueError("Invalid PIL image: could not decode image") from exc

        message = (
            f"Unsupported image input type: {type(image_input).__name__}. "
            "Expected bytes, path, or PIL Image."
        )
        raise TypeError(message)

    def _vectors_from_encoded(
        self,
        encoded: object,
        expected_count: int,
    ) -> list[object]:
        converted = self._to_python(encoded)

        if self._is_flat_vector(converted):
            if expected_count != 1:
                raise ValueError(
                    f"Visual embedding model returned one vector for {expected_count} images"
                )
            return [converted]

        if isinstance(converted, tuple):
            converted = list(converted)

        if not isinstance(converted, list):
            raise TypeError(
                f"Visual embedding model returned unsupported output type: {type(encoded).__name__}"
            )

        vectors = [self._to_python(vector) for vector in converted]
        if len(vectors) != expected_count:
            raise ValueError(
                f"Visual embedding model returned {len(vectors)} vectors for {expected_count} images"
            )
        return vectors

    def _normalize_vector(self, vector: object) -> list[float]:
        converted = self._to_python(vector)
        if isinstance(converted, tuple):
            converted = list(converted)
        if not self._is_flat_vector(converted):
            raise TypeError("Visual embedding vector must be a non-empty numeric sequence")

        floats = [float(value) for value in converted]
        norm = math.sqrt(sum(value * value for value in floats))
        if norm == 0.0:
            raise ValueError("Visual embedding model returned a zero vector")
        return [value / norm for value in floats]

    def _to_python(self, value: object) -> object:
        for method_name in ("detach", "cpu", "numpy", "tolist"):
            method = getattr(value, method_name, None)
            if callable(method):
                value = method()
        return value

    def _is_flat_vector(self, value: object) -> TypeGuard[Sequence[int | float]]:
        if not isinstance(value, (list, tuple)) or not value:
            return False
        return all(isinstance(item, (int, float)) for item in value)
