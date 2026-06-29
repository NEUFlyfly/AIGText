import os

import pytest

from src.rag import embedder as embedder_module


def test_local_embedding_load_failure_does_not_retry_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def failing_sentence_transformer(
        model_name: str,
        **kwargs: object,
    ) -> object:
        calls.append((model_name, dict(kwargs)))
        raise OSError("local model files are missing")

    monkeypatch.setattr(embedder_module, "_has_sentence_transformers", True)
    monkeypatch.setattr(
        embedder_module,
        "_sentence_transformer_factory",
        lambda: failing_sentence_transformer,
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    text_embedder = embedder_module.Embedder("local-test-embedding-model")

    with pytest.raises(RuntimeError) as exc_info:
        _ = text_embedder.model

    assert calls == [("local-test-embedding-model", {"local_files_only": True})]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    message = str(exc_info.value)
    assert "not available locally" in message
    assert "models" in message
    assert "embedding" in message
    assert "online model downloads are disabled" in message
