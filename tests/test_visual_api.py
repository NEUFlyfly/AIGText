import http.client
import json
import sys
import threading
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from src import front_server


CANONICAL_KEYS = {
    "status",
    "coarse_category",
    "coarse_confidence",
    "visual_candidates",
    "retrieved_chunks",
    "answer",
    "errors",
}


class FakeVisionBackendClient:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload: dict[str, object] = payload or _visual_payload()
        self.calls: list[dict[str, object]] = []
        self.health_calls = 0

    def search(
        self,
        image_bytes: bytes,
        *,
        question: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, object]:
        self.calls.append({
            "image_bytes": image_bytes,
            "question": question,
            "top_k": top_k,
        })
        return self.payload

    def health(self) -> dict[str, object]:
        self.health_calls += 1
        return {"status": "ok", "model_loaded": True}


class FailingVisionBackendClient:
    def __init__(self, message: str = "host b is down") -> None:
        self.message = message
        self.calls = 0

    def search(
        self,
        image_bytes: bytes,
        *,
        question: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, object]:
        from src.vision.vision_client import VisionBackendUnavailable

        self.calls += 1
        raise VisionBackendUnavailable(self.message)

    def health(self) -> dict[str, object]:
        from src.vision.vision_client import VisionBackendUnavailable

        raise VisionBackendUnavailable(self.message)


class FakeTextPipeline:
    def __init__(
        self,
        chunks: list[dict[str, object]] | None = None,
        *,
        ready: bool = True,
    ) -> None:
        self._chunks = chunks if chunks is not None else _text_chunks()
        self.is_ready = ready
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        coarse_category: str | None = None,
        sub_category: str | None = None,
        where: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append({
            "query": query,
            "doc_ids": doc_ids,
            "coarse_category": coarse_category,
            "sub_category": sub_category,
            "where": where,
        })
        if doc_ids is None:
            return list(self._chunks)
        allowed_doc_ids = set(doc_ids)
        return [
            chunk for chunk in self._chunks
            if str(chunk.get("doc_id")) in allowed_doc_ids
        ]


@pytest.fixture
def visual_api_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    servers: list[HTTPServer] = []
    threads: list[threading.Thread] = []

    def start(
        vision_client: object,
        text_pipeline: object | None = None,
        answer_provider: Callable[[str], str] | None = None,
        *,
        static_dir: Path | None = None,
    ) -> str:
        front_server.FrontendHandler.static_dir = str(static_dir or tmp_path)
        front_server.FrontendHandler.backend_url = "http://127.0.0.1:1"
        monkeypatch.setattr(front_server, "_vision_backend_client", vision_client)
        monkeypatch.setattr(front_server, "_rag_pipeline", text_pipeline or FakeTextPipeline())
        monkeypatch.setattr(front_server, "_llama_answer_provider", answer_provider)
        server = HTTPServer(("127.0.0.1", 0), front_server.FrontendHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
        host, port = server.server_address[:2]
        return f"{host}:{port}"

    yield start

    for server in servers:
        server.shutdown()
        server.server_close()
    for thread in threads:
        thread.join(timeout=2)


def test_visual_query_happy_path_returns_fixed_schema_and_local_rag(
    sample_image_bytes: bytes,
    visual_api_server,
) -> None:
    vision_client = FakeVisionBackendClient()
    text_pipeline = FakeTextPipeline()
    prompts: list[str] = []

    def answer_provider(prompt: str) -> str:
        prompts.append(prompt)
        return "mock answer"

    address = visual_api_server(vision_client, text_pipeline, answer_provider)
    status_code, payload = _post_multipart(
        address,
        "/api/vision/query",
        image_bytes=sample_image_bytes,
    )

    assert status_code == 200
    assert set(payload) == CANONICAL_KEYS
    assert payload["status"] == "OK"
    assert payload["coarse_category"] == "sensor"
    assert payload["coarse_confidence"] == 0.91
    assert payload["visual_candidates"] == [_serialized_candidate()]
    assert payload["retrieved_chunks"] == [_serialized_chunk()]
    assert payload["answer"] == "mock answer"
    assert payload["errors"] == []
    assert vision_client.calls[0]["question"] == front_server.DEFAULT_VISUAL_QUESTION
    assert vision_client.calls[0]["top_k"] is None
    assert text_pipeline.calls == [{
        "query": front_server.DEFAULT_VISUAL_QUESTION,
        "doc_ids": ["fixture_temp_sensor"],
        "coarse_category": None,
        "sub_category": None,
        "where": None,
    }]
    assert len(prompts) == 1
    assert "doc_id=fixture_temp_sensor" in prompts[0]
    assert "Temperature sensors collect temperature and humidity." in prompts[0]


def test_visual_query_accepts_question_alias_and_top_k(
    sample_image_bytes: bytes,
    visual_api_server,
) -> None:
    vision_client = FakeVisionBackendClient()
    text_pipeline = FakeTextPipeline()
    address = visual_api_server(vision_client, text_pipeline, lambda prompt: "answer")

    status_code, payload = _post_multipart(
        address,
        "/api/vision/query",
        image_bytes=sample_image_bytes,
        query="How is this installed?",
        top_k=2,
    )

    assert status_code == 200
    assert payload["answer"] == "answer"
    assert vision_client.calls[0]["question"] == "How is this installed?"
    assert vision_client.calls[0]["top_k"] == 2
    assert text_pipeline.calls[0]["query"] == "How is this installed?"


def test_visual_classify_is_compatibility_alias_with_device_class(
    sample_image_bytes: bytes,
    visual_api_server,
) -> None:
    address = visual_api_server(
        FakeVisionBackendClient(),
        FakeTextPipeline(),
        lambda prompt: "alias answer",
    )

    status_code, payload = _post_multipart(
        address,
        "/api/vision/classify",
        image_bytes=sample_image_bytes,
    )

    assert status_code == 200
    assert set(payload) == CANONICAL_KEYS | {"device_class"}
    assert payload["status"] == "OK"
    assert payload["device_class"] == "temp_sensor"
    assert payload["answer"] == "alias answer"


def test_visual_query_rejects_invalid_image(visual_api_server) -> None:
    vision_client = FakeVisionBackendClient()
    text_pipeline = FakeTextPipeline()
    address = visual_api_server(
        vision_client,
        text_pipeline,
        lambda prompt: "should not be called",
    )

    status_code, payload = _post_multipart(
        address,
        "/api/vision/query",
        image_bytes=b"not an image",
    )

    assert status_code == 400
    assert payload["errors"][0]["code"] == "INVALID_IMAGE"
    assert vision_client.calls == []
    assert text_pipeline.calls == []


def test_visual_query_maps_backend_unavailable_to_retryable_503(
    sample_image_bytes: bytes,
    visual_api_server,
) -> None:
    vision_client = FailingVisionBackendClient("host b timeout")
    text_pipeline = FakeTextPipeline()
    address = visual_api_server(
        vision_client,
        text_pipeline,
        lambda prompt: "should not be called",
    )

    status_code, payload = _post_multipart(
        address,
        "/api/vision/query",
        image_bytes=sample_image_bytes,
    )

    assert status_code == 503
    assert payload == {
        "status": "VISION_BACKEND_UNAVAILABLE",
        "coarse_category": None,
        "coarse_confidence": None,
        "visual_candidates": [],
        "retrieved_chunks": [],
        "answer": None,
        "errors": [{
            "code": "VISION_BACKEND_UNAVAILABLE",
            "message": "host b timeout",
            "retryable": True,
        }],
    }
    assert vision_client.calls == 1
    assert text_pipeline.calls == []


def test_visual_query_maps_host_b_index_not_ready_to_503(
    sample_image_bytes: bytes,
    visual_api_server,
) -> None:
    vision_client = FakeVisionBackendClient(_visual_payload(status="INDEX_NOT_READY"))
    address = visual_api_server(
        vision_client,
        FakeTextPipeline(),
        lambda prompt: "should not be called",
    )

    status_code, payload = _post_multipart(
        address,
        "/api/vision/query",
        image_bytes=sample_image_bytes,
    )

    assert status_code == 503
    assert payload["status"] == "INDEX_NOT_READY"
    assert payload["errors"][0]["code"] == "INDEX_NOT_READY"
    assert payload["answer"] is None


def test_visual_query_exposes_low_confidence_status(
    sample_image_bytes: bytes,
    visual_api_server,
) -> None:
    vision_client = FakeVisionBackendClient(_visual_payload(coarse_status="LOW_CONFIDENCE"))
    address = visual_api_server(
        vision_client,
        FakeTextPipeline(),
        lambda prompt: "low confidence answer",
    )

    status_code, payload = _post_multipart(
        address,
        "/api/vision/query",
        image_bytes=sample_image_bytes,
    )

    assert status_code == 200
    assert payload["status"] == "LOW_CONFIDENCE"
    assert payload["answer"] == "low confidence answer"


def test_visual_query_with_missing_coarse_classifier_still_returns_usable_payload(
    sample_image_bytes: bytes,
    visual_api_server,
) -> None:
    vision_client = FakeVisionBackendClient(_visual_payload(coarse_status="MODEL_NOT_READY"))
    address = visual_api_server(
        vision_client,
        FakeTextPipeline(),
        lambda prompt: "degraded classifier answer",
    )

    status_code, payload = _post_multipart(
        address,
        "/api/vision/query",
        image_bytes=sample_image_bytes,
    )

    assert status_code == 200
    assert payload["status"] == "MODEL_NOT_READY"
    assert payload["answer"] == "degraded classifier answer"
    assert payload["visual_candidates"]
    assert payload["retrieved_chunks"]
    assert payload["errors"] == []


def test_dual_host_path_does_not_load_local_visual_modules(
    sample_image_bytes: bytes,
    visual_api_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("src.rag.visual_retriever", None)
    sys.modules.pop("src.rag.visual_embedder", None)

    def fail_local_visual_pipeline() -> None:
        raise AssertionError("local visual pipeline must not be used")

    monkeypatch.setattr(
        front_server,
        "_get_visual_rag_pipeline",
        fail_local_visual_pipeline,
        raising=False,
    )
    address = visual_api_server(
        FakeVisionBackendClient(),
        FakeTextPipeline(),
        lambda prompt: "answer",
    )

    status_code, payload = _post_multipart(
        address,
        "/api/vision/query",
        image_bytes=sample_image_bytes,
    )

    assert status_code == 200
    assert payload["answer"] == "answer"
    assert "src.rag.visual_retriever" not in sys.modules
    assert "src.rag.visual_embedder" not in sys.modules


def test_frontend_js_does_not_expose_host_b_url_or_api_key(
    visual_api_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_b_url = "http://host-b.internal:18081"
    host_b_api_key = "super-secret-host-b-token"
    monkeypatch.setenv("VISION_BACKEND_URL", host_b_url)
    monkeypatch.setenv("VISION_BACKEND_API_KEY", host_b_api_key)
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    address = visual_api_server(
        FakeVisionBackendClient(),
        FakeTextPipeline(),
        lambda prompt: "answer",
        static_dir=frontend_dir,
    )

    status_code, body = _get_text(address, "/js/chat.js")

    assert status_code == 200
    assert host_b_url not in body
    assert host_b_api_key not in body
    assert "VISION_BACKEND" not in body
    assert "/api/vision/query" in body


def _post_multipart(
    address: str,
    path: str,
    *,
    image_bytes: bytes,
    question: str | None = None,
    query: str | None = None,
    top_k: int | None = None,
) -> tuple[int, dict[str, Any]]:
    boundary = "----aigtext-visual-api-test"
    body = _multipart_body(
        boundary,
        image_bytes=image_bytes,
        question=question,
        query=query,
        top_k=top_k,
    )
    connection = http.client.HTTPConnection(address, timeout=10)
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        response_body = response.read().decode("utf-8")
        return response.status, json.loads(response_body)
    finally:
        connection.close()


def _get_text(address: str, path: str) -> tuple[int, str]:
    connection = http.client.HTTPConnection(address, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


def _multipart_body(
    boundary: str,
    *,
    image_bytes: bytes,
    question: str | None,
    query: str | None,
    top_k: int | None,
) -> bytes:
    boundary_bytes = boundary.encode("ascii")
    parts = [
        b"--" + boundary_bytes,
        (
            b'Content-Disposition: form-data; name="image"; '
            b'filename="photo.jpg"'
        ),
        b"Content-Type: image/jpeg",
        b"",
        image_bytes,
    ]
    for name, value in (
        ("question", question),
        ("query", query),
        ("top_k", None if top_k is None else str(top_k)),
    ):
        if value is None:
            continue
        parts.extend([
            b"--" + boundary_bytes,
            f'Content-Disposition: form-data; name="{name}"'.encode("ascii"),
            b"Content-Type: text/plain; charset=utf-8",
            b"",
            value.encode("utf-8"),
        ])
    parts.extend([b"--" + boundary_bytes + b"--", b""])
    return b"\r\n".join(parts)


def _visual_payload(
    *,
    status: str = "OK",
    coarse_status: str = "OK",
) -> dict[str, object]:
    candidates = [] if status == "INDEX_NOT_READY" else [_candidate()]
    return {
        "status": status,
        "coarse_category": "sensor",
        "coarse_confidence": 0.91,
        "coarse_status": coarse_status,
        "visual_candidates": candidates,
    }


def _candidate() -> dict[str, object]:
    return {
        "doc_id": "fixture_temp_sensor",
        "sub_category": "temp_sensor",
        "coarse_category": "sensor",
        "score": 0.94,
        "evidence_image_id": "temp-image-a",
        "evidence_image_path": "ignored-by-api",
        "matched_image_count": 2,
        "status": "OK",
    }


def _text_chunks() -> list[dict[str, object]]:
    return [_serialized_chunk()]


def _serialized_candidate() -> dict[str, object]:
    return {
        "doc_id": "fixture_temp_sensor",
        "sub_category": "temp_sensor",
        "coarse_category": "sensor",
        "score": 0.94,
        "evidence_image_id": "temp-image-a",
    }


def _serialized_chunk() -> dict[str, object]:
    return {
        "doc_id": "fixture_temp_sensor",
        "source": "tests/fixtures/iot_knowledge/sensor/temp_sensor/document.md",
        "chunk_id": 0,
        "score": 0.88,
        "text": "Temperature sensors collect temperature and humidity.",
    }
