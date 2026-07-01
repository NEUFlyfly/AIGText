"""HTTP client for Host B visual candidate search."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid

from config.settings import settings

logger = logging.getLogger("vision_client")


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
        url = f"{self._base_url}/api/vision/search"
        headers = self._headers(content_type)
        logger.info(
            "[vision] POST %s image=%dB fields=%s timeout=%.1fs api_key=%s",
            url, len(image_bytes), list(fields), self._timeout_seconds,
            "yes" if self._api_key else "no",
        )
        request = urllib.request.Request(url, data=body, headers=headers)
        return self._json_request(request)

    def health(self) -> dict[str, object]:
        url = f"{self._base_url}/health"
        logger.info("[vision] health GET %s (timeout=%.1fs)", url, self._timeout_seconds)
        request = urllib.request.Request(url, headers=self._headers())
        return self._json_request(request)

    def _json_request(self, request: urllib.request.Request) -> dict[str, object]:
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                raw_body = response.read()
            elapsed = time.monotonic() - t0
            status = response.status
            logger.info(
                "[vision] %s %s -> %d body=%dB elapsed=%.2fs",
                request.get_method(), request.full_url, status, len(raw_body), elapsed,
            )
        except TimeoutError as exc:
            elapsed = time.monotonic() - t0
            logger.error(
                "[vision] TIMEOUT %s %s after %.2fs: %s",
                request.get_method(), request.full_url, elapsed, exc,
            )
            raise VisionBackendUnavailable("vision backend timed out") from exc
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - t0
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.error(
                "[vision] HTTP ERROR %s %s -> %s %s (elapsed=%.2fs): %s",
                request.get_method(), request.full_url, exc.code, exc.reason,
                elapsed, error_body[:500],
            )
            raise VisionBackendUnavailable(
                f"vision backend HTTP {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            elapsed = time.monotonic() - t0
            logger.error(
                "[vision] CONNECTION ERROR %s %s after %.2fs: %s",
                request.get_method(), request.full_url, elapsed, exc,
            )
            raise VisionBackendUnavailable("vision backend unavailable") from exc

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            logger.error(
                "[vision] BAD JSON from %s (len=%d): %s",
                request.full_url, len(raw_body), exc,
            )
            raise VisionBackendResponseError("vision backend returned invalid JSON") from exc
        if not isinstance(payload, dict):
            logger.error(
                "[vision] NON-OBJECT response from %s: %s",
                request.full_url, type(payload).__name__,
            )
            raise VisionBackendResponseError("vision backend returned a non-object payload")

        coarse = payload.get("coarse_category", "")
        cand_count = len(payload.get("visual_candidates", []))
        first_id = ""
        if payload.get("visual_candidates"):
            first_id = payload["visual_candidates"][0].get("doc_id", "")
        logger.info(
            "[vision] decoded: coarse=%s candidates=%d first_doc=%s",
            coarse, cand_count, first_id,
        )
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
