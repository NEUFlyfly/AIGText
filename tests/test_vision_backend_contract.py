import http.client
import json
import threading
from collections.abc import Iterator
from http.server import HTTPServer
from typing import cast

import pytest

from src.vision import vision_server


CONTRACT_KEYS = {
    "status",
    "contract_version",
    "request_id",
    "coarse_category",
    "coarse_confidence",
    "coarse_status",
    "visual_candidates",
    "errors",
}
FORBIDDEN_TOP_LEVEL_KEYS = {"retrieved_chunks", "augmented_prompt", "answer"}


class FakeVisualRetriever:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload: dict[str, object] = payload if payload is not None else _retrieval_payload()
        self.calls: list[dict[str, object]] = []

    def retrieve(self, image_bytes: bytes, *, top_k: int | None = None) -> dict[str, object]:
        self.calls.append({"image_bytes": image_bytes, "top_k": top_k})
        return self.payload


class FailingVisualRetriever:
    def retrieve(self, image_bytes: bytes, *, top_k: int | None = None) -> dict[str, object]:
        _ = (image_bytes, top_k)
        raise FileNotFoundError("models/visual_embedding/clip-ViT-B-32 is missing")


@pytest.fixture
def vision_backend_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.delenv("VISION_BACKEND_MODE", raising=False)
    monkeypatch.delenv("VISION_BACKEND_API_KEY", raising=False)
    monkeypatch.delenv("VISION_API_KEY", raising=False)
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    monkeypatch.setattr(vision_server, "_visual_retriever", None)

    server = HTTPServer(("127.0.0.1", 0), vision_server.VisionBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]

    yield f"{host}:{port}"

    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_health_reports_private_visual_backend_role(
    vision_backend_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISION_BACKEND_MODE", "stub")

    status_code, payload = _get_json(vision_backend_server, "/health")

    assert status_code == 200
    assert payload["status"] == "ok"
    assert payload["service"] == "vision-backend"
    assert payload["role"] == "host-b-private-visual-search"
    assert payload["mode"] == "stub"
    assert payload["contract_version"] == "visual-search.v1"


def test_stub_search_returns_visual_only_contract(
    sample_image_bytes: bytes,
    vision_backend_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISION_BACKEND_MODE", "stub")
    monkeypatch.setenv("VISION_BACKEND_API_KEY", "secret-key")

    status_code, payload = _post_multipart(
        vision_backend_server,
        "/api/vision/search",
        image_bytes=sample_image_bytes,
        question="What is this device?",
        top_k=1,
        headers={
            "X-Internal-API-Key": "secret-key",
            "X-Request-ID": "request-123",
        },
    )

    assert status_code == 200
    assert set(payload) == CONTRACT_KEYS
    assert not FORBIDDEN_TOP_LEVEL_KEYS & set(payload)
    assert payload["status"] == "OK"
    assert payload["contract_version"] == "visual-search.v1"
    assert payload["request_id"] == "request-123"
    assert payload["coarse_category"] == "stub_iot_device"
    assert payload["coarse_confidence"] == 0.88
    assert payload["coarse_status"] == "OK"
    assert payload["errors"] == []
    visual_candidates = cast(list[dict[str, object]], payload["visual_candidates"])
    assert visual_candidates == [{
        "doc_id": "stub_visual_device",
        "sub_category": "stub_sensor",
        "coarse_category": "stub_iot_device",
        "score": 0.91,
        "evidence_image_id": "stub-image-a",
        "matched_image_count": 1,
        "status": "OK",
    }]


def test_search_validates_internal_api_key_when_configured(
    vision_backend_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VISION_BACKEND_API_KEY", "secret-key")

    status_code, payload = _post_multipart(
        vision_backend_server,
        "/api/vision/search",
        image_bytes=b"not parsed because auth fails first",
    )

    assert status_code == 401
    assert set(payload) == CONTRACT_KEYS
    assert payload["status"] == "UNAUTHORIZED"
    assert payload["visual_candidates"] == []
    assert payload["errors"] == [{"code": "UNAUTHORIZED", "message": "Invalid internal API key"}]


def test_production_search_uses_visual_retriever_and_sanitizes_candidates(
    sample_image_bytes: bytes,
    vision_backend_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = FakeVisualRetriever()
    monkeypatch.setattr(vision_server, "_visual_retriever", retriever)

    status_code, payload = _post_multipart(
        vision_backend_server,
        "/api/vision/search",
        image_bytes=sample_image_bytes,
        top_k=5,
        headers={"X-Request-ID": "prod-request"},
    )

    assert status_code == 200
    assert set(payload) == CONTRACT_KEYS
    assert payload["request_id"] == "prod-request"
    assert payload["status"] == "OK"
    assert retriever.calls == [{"image_bytes": sample_image_bytes, "top_k": 5}]
    visual_candidates = cast(list[dict[str, object]], payload["visual_candidates"])
    assert visual_candidates == [{
        "doc_id": "fixture_temp_sensor",
        "sub_category": "temp_sensor",
        "coarse_category": "sensor",
        "score": 0.94,
        "evidence_image_id": "temp-image-a",
        "matched_image_count": 2,
        "status": "OK",
    }]
    assert "evidence_image_path" not in visual_candidates[0]
    assert not FORBIDDEN_TOP_LEVEL_KEYS & set(payload)


def test_production_missing_model_fails_gracefully_without_paths(
    sample_image_bytes: bytes,
    vision_backend_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vision_server, "_visual_retriever", FailingVisualRetriever())

    status_code, payload = _post_multipart(
        vision_backend_server,
        "/api/vision/search",
        image_bytes=sample_image_bytes,
        headers={"X-Request-ID": "missing-model"},
    )

    assert status_code == 503
    assert set(payload) == CONTRACT_KEYS
    assert payload["status"] == "MODEL_NOT_READY"
    assert payload["request_id"] == "missing-model"
    assert payload["visual_candidates"] == []
    assert payload["errors"] == [{"code": "MODEL_NOT_READY", "message": "Visual model is not ready"}]
    assert "models/" not in json.dumps(payload)


def _get_json(address: str, path: str) -> tuple[int, dict[str, object]]:
    connection = http.client.HTTPConnection(address, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        response_body = response.read().decode("utf-8")
        return response.status, cast(dict[str, object], json.loads(response_body))
    finally:
        connection.close()


def _post_multipart(
    address: str,
    path: str,
    *,
    image_bytes: bytes,
    question: str | None = None,
    top_k: int | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    boundary = "----aigtext-vision-backend-test"
    body = _multipart_body(
        boundary,
        image_bytes=image_bytes,
        question=question,
        top_k=top_k,
    )
    request_headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    if headers is not None:
        request_headers.update(headers)

    connection = http.client.HTTPConnection(address, timeout=10)
    try:
        connection.request("POST", path, body=body, headers=request_headers)
        response = connection.getresponse()
        response_body = response.read().decode("utf-8")
        return response.status, cast(dict[str, object], json.loads(response_body))
    finally:
        connection.close()


def _multipart_body(
    boundary: str,
    *,
    image_bytes: bytes,
    question: str | None,
    top_k: int | None,
) -> bytes:
    boundary_bytes = boundary.encode("ascii")
    parts = [
        b"--" + boundary_bytes,
        b'Content-Disposition: form-data; name="image"; filename="photo.jpg"',
        b"Content-Type: image/jpeg",
        b"",
        image_bytes,
    ]
    if question is not None:
        parts.extend([
            b"--" + boundary_bytes,
            b'Content-Disposition: form-data; name="question"',
            b"Content-Type: text/plain; charset=utf-8",
            b"",
            question.encode("utf-8"),
        ])
    if top_k is not None:
        parts.extend([
            b"--" + boundary_bytes,
            b'Content-Disposition: form-data; name="top_k"',
            b"Content-Type: text/plain; charset=utf-8",
            b"",
            str(top_k).encode("ascii"),
        ])
    parts.extend([b"--" + boundary_bytes + b"--", b""])
    return b"\r\n".join(parts)


def _retrieval_payload() -> dict[str, object]:
    return {
        "status": "OK",
        "coarse_category": "sensor",
        "coarse_confidence": 0.91,
        "coarse_status": "OK",
        "visual_candidates": [{
            "doc_id": "fixture_temp_sensor",
            "sub_category": "temp_sensor",
            "coarse_category": "sensor",
            "score": 0.94,
            "evidence_image_id": "temp-image-a",
            "evidence_image_path": "tests/fixtures/iot_knowledge/sensor/temp_sensor/images/a.jpg",
            "matched_image_count": 2,
            "status": "OK",
        }],
        "retrieved_chunks": [{"text": "must not leak"}],
        "augmented_prompt": "must not leak",
        "answer": "must not leak",
    }
