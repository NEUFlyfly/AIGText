"""HTTP client for Host B visual candidate search."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid

from config.settings import settings


class VisionBackendUnavailable(RuntimeError):
    """Raised when Host B cannot be reached or times out."""


class VisionBackendResponseError(RuntimeError):
    """Raised when Host B returns an unusable response."""


class VisionBackendClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._base_url = (base_url or settings.VISION_BACKEND_URL).rstrip("/")
        self._api_key = api_key if api_key is not None else settings.VISION_BACKEND_API_KEY
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.VISION_BACKEND_TIMEOUT_SECONDS
        )

    def search(
        self,
        image_bytes: bytes,
        *,
        question: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, object]:
        fields: dict[str, str] = {}
        if question:
            fields["question"] = question
        if top_k is not None:
            fields["top_k"] = str(top_k)
        body, content_type = _multipart_body(
            image_bytes=image_bytes,
            fields=fields,
        )
        request = urllib.request.Request(
            f"{self._base_url}/api/vision/search",
            data=body,
            headers=self._headers(content_type),
        )
        return self._json_request(request)

    def health(self) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self._base_url}/health",
            headers=self._headers(),
        )
        return self._json_request(request)

    def _json_request(self, request: urllib.request.Request) -> dict[str, object]:
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                raw_body = response.read()
        except TimeoutError as exc:
            raise VisionBackendUnavailable("vision backend timed out") from exc
        except urllib.error.URLError as exc:
            raise VisionBackendUnavailable("vision backend unavailable") from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise VisionBackendResponseError("vision backend returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise VisionBackendResponseError("vision backend returned a non-object payload")
        return payload

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers


def _multipart_body(
    *,
    image_bytes: bytes,
    fields: dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"----aigtext-hostb-{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    parts = [
        b"--" + boundary_bytes,
        (
            b'Content-Disposition: form-data; name="image"; '
            b'filename="image.jpg"'
        ),
        b"Content-Type: image/jpeg",
        b"",
        image_bytes,
    ]
    for name, value in fields.items():
        parts.extend([
            b"--" + boundary_bytes,
            f'Content-Disposition: form-data; name="{name}"'.encode("ascii"),
            b"Content-Type: text/plain; charset=utf-8",
            b"",
            value.encode("utf-8"),
        ])
    parts.extend([b"--" + boundary_bytes + b"--", b""])
    body = b"\r\n".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"
