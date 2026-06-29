#!/usr/bin/env python3
"""Private Host B visual-only backend service."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import os
import threading
import uuid
from typing import Protocol, TypedDict, cast

from config.settings import settings


CONTRACT_VERSION = "visual-search.v1"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8091
STUB_MODE = "stub"
PRODUCTION_MODE = "production"
_READY_STATUSES = {"OK", "NO_VISUAL_MATCH", "LOW_CONFIDENCE", "MODEL_NOT_READY"}
_HTTP_503_STATUSES = {"MODEL_NOT_READY", "INDEX_NOT_READY"}
_visual_retriever: object | None = None
_VISUAL_RETRIEVER_LOCK = threading.Lock()


class MultipartSearchForm(TypedDict):
    image: bytes
    top_k: int


class VisualSearchBackend(Protocol):
    def retrieve(
        self,
        image_bytes: bytes,
        *,
        top_k: int | None = None,
    ) -> dict[str, object]:
        ...


class VisionBackendHandler(BaseHTTPRequestHandler):
    """HTTP handler for private Host B visual search."""

    server_version: str = "AIGTextVisionBackend/1.0"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, _health_payload())
            return
        self._send_json(404, {"status": "NOT_FOUND", "errors": [_error("NOT_FOUND", "Not found")]})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/api/vision/search":
            self._send_json(404, {"status": "NOT_FOUND", "errors": [_error("NOT_FOUND", "Not found")]})
            return

        request_id = _request_id_from_header(self.headers.get("X-Request-ID"))
        auth_error = _authorization_error(self.headers.get("X-Internal-API-Key"))
        if auth_error is not None:
            self._send_json(401, _contract_error_payload(auth_error, request_id=request_id))
            return

        try:
            form = _parse_multipart_request(self)
        except ValueError as exc:
            self._send_json(
                400,
                _contract_error_payload(
                    "INVALID_REQUEST",
                    request_id=request_id,
                    message=str(exc),
                ),
            )
            return

        payload = _run_visual_search(
            form["image"],
            top_k=form["top_k"],
            request_id=request_id,
        )
        status_code = 503 if payload["status"] in _HTTP_503_STATUSES else 200
        self._send_json(status_code, payload)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Internal-API-Key, X-Request-ID",
        )


def _health_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "vision-backend",
        "role": "host-b-private-visual-search",
        "mode": _backend_mode(),
        "contract_version": CONTRACT_VERSION,
    }


def _run_visual_search(
    image_bytes: bytes,
    *,
    top_k: int,
    request_id: str,
) -> dict[str, object]:
    if _backend_mode() == STUB_MODE:
        return _stub_search_payload(top_k=top_k, request_id=request_id)

    try:
        retriever = _get_visual_retriever()
        result = retriever.retrieve(image_bytes, top_k=top_k)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        return _production_exception_payload(exc, request_id=request_id)

    return _contract_payload_from_retrieval(result, request_id=request_id)


def _get_visual_retriever() -> VisualSearchBackend:
    global _visual_retriever
    if _visual_retriever is not None:
        return cast(VisualSearchBackend, _visual_retriever)

    with _VISUAL_RETRIEVER_LOCK:
        if _visual_retriever is None:
            from src.rag.visual_retriever import VisualRetriever
            _visual_retriever = VisualRetriever()
    return cast(VisualSearchBackend, _visual_retriever)


def _contract_payload_from_retrieval(
    result: dict[str, object],
    *,
    request_id: str,
) -> dict[str, object]:
    status = _response_status(result)
    errors = [] if status in _READY_STATUSES else [_error(status, _status_message(status))]
    if status in _HTTP_503_STATUSES:
        errors = [_error(status, _status_message(status))]

    return {
        "status": status,
        "contract_version": CONTRACT_VERSION,
        "request_id": request_id,
        "coarse_category": _string_or_none(result.get("coarse_category")),
        "coarse_confidence": _float_or_none(result.get("coarse_confidence")),
        "coarse_status": _string_or_none(result.get("coarse_status")) or "UNKNOWN",
        "visual_candidates": _serialize_visual_candidates(result.get("visual_candidates")),
        "errors": errors,
    }


def _response_status(result: dict[str, object]) -> str:
    status = _string_or_none(result.get("status")) or "MODEL_NOT_READY"
    coarse_status = _string_or_none(result.get("coarse_status"))
    if status in _HTTP_503_STATUSES:
        return status
    if status == "OK" and coarse_status == "MODEL_NOT_READY":
        return "MODEL_NOT_READY"
    if status == "OK" and coarse_status == "LOW_CONFIDENCE":
        return "LOW_CONFIDENCE"
    return status


def _production_exception_payload(
    exc: Exception,
    *,
    request_id: str,
) -> dict[str, object]:
    code = "INDEX_NOT_READY" if isinstance(exc, ImportError) else "MODEL_NOT_READY"
    return _contract_error_payload(code, request_id=request_id, message=_status_message(code))


def _stub_search_payload(*, top_k: int, request_id: str) -> dict[str, object]:
    candidates = [
        {
            "doc_id": "stub_visual_device",
            "sub_category": "stub_sensor",
            "coarse_category": "stub_iot_device",
            "score": 0.91,
            "evidence_image_id": "stub-image-a",
            "matched_image_count": 1,
            "status": "OK",
        },
        {
            "doc_id": "stub_visual_gateway",
            "sub_category": "stub_gateway",
            "coarse_category": "stub_iot_device",
            "score": 0.82,
            "evidence_image_id": "stub-image-b",
            "matched_image_count": 1,
            "status": "OK",
        },
    ]
    return {
        "status": "OK",
        "contract_version": CONTRACT_VERSION,
        "request_id": request_id,
        "coarse_category": "stub_iot_device",
        "coarse_confidence": 0.88,
        "coarse_status": "OK",
        "visual_candidates": candidates[:top_k],
        "errors": [],
    }


def _contract_error_payload(
    code: str,
    *,
    request_id: str,
    message: str | None = None,
) -> dict[str, object]:
    return {
        "status": code,
        "contract_version": CONTRACT_VERSION,
        "request_id": request_id,
        "coarse_category": None,
        "coarse_confidence": None,
        "coarse_status": "UNKNOWN",
        "visual_candidates": [],
        "errors": [_error(code, message or _status_message(code))],
    }


def _serialize_visual_candidates(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    candidates: list[dict[str, object]] = []
    for raw_item in cast(list[object], value):
        if not isinstance(raw_item, dict):
            continue
        item = cast(dict[str, object], raw_item)
        candidates.append({
            "doc_id": _string_or_none(item.get("doc_id")) or "",
            "sub_category": _string_or_none(item.get("sub_category")) or "",
            "coarse_category": _string_or_none(item.get("coarse_category")) or "",
            "score": _float_or_none(item.get("score")) or 0.0,
            "evidence_image_id": _string_or_none(item.get("evidence_image_id")) or "",
            "matched_image_count": _int_or_zero(item.get("matched_image_count")),
            "status": _string_or_none(item.get("status")) or "OK",
        })
    return candidates


def _parse_multipart_request(handler: BaseHTTPRequestHandler) -> MultipartSearchForm:
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("Expected multipart/form-data with an image field")

    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("Invalid Content-Length") from exc
    if content_length <= 0:
        raise ValueError("Missing multipart request body")

    body = handler.rfile.read(content_length)
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + content_type.encode("utf-8")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not message.is_multipart():
        raise ValueError("Invalid multipart/form-data body")

    image_bytes: bytes | None = None
    top_k = settings.VISUAL_TOP_K
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        raw_payload = part.get_payload(decode=True)
        payload = raw_payload if isinstance(raw_payload, bytes) else b""
        if name == "image":
            image_bytes = payload
        elif name == "top_k":
            top_k = _parse_top_k(payload, part.get_content_charset() or "utf-8")
        elif name == "question":
            continue

    if not image_bytes:
        raise ValueError("Missing required image upload")
    _validate_image_upload(image_bytes)
    return {"image": image_bytes, "top_k": top_k}


def _parse_top_k(payload: bytes, charset: str) -> int:
    try:
        top_k = int(payload.decode(charset, errors="replace").strip())
    except ValueError as exc:
        raise ValueError("top_k must be a positive integer") from exc
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return min(top_k, _top_k_limit())


def _validate_image_upload(image_bytes: bytes) -> None:
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("Invalid image upload") from exc


def _authorization_error(header_value: str | None) -> str | None:
    expected_key = (
        os.environ.get("VISION_BACKEND_API_KEY")
        or os.environ.get("VISION_API_KEY")
        or os.environ.get("INTERNAL_API_KEY")
    )
    if not expected_key:
        return None
    if header_value == expected_key:
        return None
    return "UNAUTHORIZED"


def _request_id_from_header(header_value: str | None) -> str:
    if isinstance(header_value, str) and header_value.strip():
        return header_value.strip()
    return uuid.uuid4().hex


def _backend_mode() -> str:
    mode = os.environ.get("VISION_BACKEND_MODE", PRODUCTION_MODE).strip().lower()
    return STUB_MODE if mode == STUB_MODE else PRODUCTION_MODE


def _top_k_limit() -> int:
    try:
        configured_limit = int(os.environ.get("VISUAL_TOP_K_MAX", "20"))
    except ValueError:
        return 20
    return configured_limit if configured_limit > 0 else 20


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _status_message(status: str) -> str:
    if status == "INDEX_NOT_READY":
        return "Visual index is not ready"
    if status == "MODEL_NOT_READY":
        return "Visual model is not ready"
    if status == "UNAUTHORIZED":
        return "Invalid internal API key"
    if status == "INVALID_REQUEST":
        return "Invalid visual search request"
    return status


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def main(
    *,
    host: str | None = None,
    port: int | None = None,
    backend_mode: str | None = None,
    fallback_mode: str | None = None,
    api_key: str | None = None,
    top_k_max: int | None = None,
) -> None:
    if backend_mode is not None:
        os.environ["VISION_BACKEND_MODE"] = backend_mode
    if fallback_mode is not None:
        os.environ["VISION_FALLBACK_MODE"] = fallback_mode
    if api_key is not None:
        os.environ["VISION_API_KEY"] = api_key
    if top_k_max is not None:
        os.environ["VISUAL_TOP_K_MAX"] = str(top_k_max)

    if host is None or port is None:
        parser = argparse.ArgumentParser(description="AIGText private Host B visual backend")
        _ = parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host")
        _ = parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
        args = parser.parse_args()
        host = cast(str, args.host)
        port = cast(int, args.port)

    server = HTTPServer((host, port), VisionBackendHandler)
    print(f"AIGText vision backend listening on http://{host}:{port}")
    print(f"mode={_backend_mode()} contract={CONTRACT_VERSION}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVision backend stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
