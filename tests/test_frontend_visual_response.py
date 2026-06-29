"""Deterministic tests for frontend Visual RAG response parsing and error mapping.

These tests mirror the JS-side response parsing and error display logic from
frontend/js/chat.js, serving as executable specifications without a browser.

Tests cover:
- Parsing visual_candidates (labels + scores) and answer text
- Legacy device_class compatibility
- Error code to user-facing display text mapping
- Edge cases: empty candidates, null answer, missing fields
"""

from __future__ import annotations

from typing import Any

import pytest

# ============================================================================
# JS-mirror parsing functions (executable specification of chat.js behavior)
# ============================================================================

# Canonical keys required in every Visual RAG API response (sync with test_visual_api.py)
CANONICAL_KEYS = frozenset({
    "status",
    "coarse_category",
    "coarse_confidence",
    "visual_candidates",
    "retrieved_chunks",
    "answer",
    "errors",
})

# Error codes the frontend must handle with display text
VISUAL_ERROR_CODES = frozenset({
    "INVALID_IMAGE",
    "MODEL_NOT_READY",
    "INDEX_NOT_READY",
    "NO_VISUAL_MATCH",
})

# Chinese display text mapping — mirrors visualErrorDisplayText() in chat.js
VISUAL_ERROR_DISPLAY: dict[str, str] = {
    "INVALID_IMAGE": "图片无效，请重新拍摄清晰的设备照片",
    "MODEL_NOT_READY": "识别模型未就绪，请稍后重试",
    "INDEX_NOT_READY": "知识库索引未就绪，请稍后重试",
    "NO_VISUAL_MATCH": "未识别到匹配的设备，请尝试不同角度或光线",
}


def parse_visual_rag_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Mirrors the JS-side response parsing in displayVisualRagResponse().

    Extracts candidate summary (label + score) list, answer text, coarse info,
    and legacy device_class compatibility from a /api/vision/query response.

    Returns a dict with keys:
        status, answer, coarse_category, coarse_confidence,
        candidates: [{"label": str, "score": int}],
        device_class: str | None,
        errors: list[dict],
    """
    result: dict[str, Any] = {
        "status": string_or_none(payload.get("status")),
        "answer": string_or_none(payload.get("answer")),
        "coarse_category": string_or_none(payload.get("coarse_category")),
        "coarse_confidence": float_or_none(payload.get("coarse_confidence")),
        "device_class": string_or_none(payload.get("device_class")),
        "candidates": [],
        "errors": ensure_list(payload.get("errors")),
    }

    # Parse visual_candidates into display-friendly labels + percentage scores
    for c in ensure_list(payload.get("visual_candidates")):
        if not isinstance(c, dict):
            continue
        label = (
            string_or_none(c.get("sub_category"))
            or string_or_none(c.get("coarse_category"))
            or "unknown"
        )
        score = float_or_none(c.get("score")) or 0.0
        result["candidates"].append({
            "label": label,
            "score": round(score * 100),
        })

    # Legacy device_class compatibility: derive from first candidate
    if not result["device_class"] and result["candidates"]:
        result["device_class"] = result["candidates"][0]["label"]

    return result


def visual_error_display_text(error_code: str) -> str:
    """Mirrors the JS-side visualErrorDisplayText() mapping.

    Returns the user-facing Chinese error text for known error codes,
    or a generic fallback message for unknown codes.
    """
    return VISUAL_ERROR_DISPLAY.get(error_code, "识别服务异常，请重试")


def should_trigger_chat_request(result: dict[str, Any]) -> bool:
    """Returns True when the frontend should send a follow-up /api/chat request.

    Mirrors the JS logic: when the Visual RAG response already includes
    an 'answer', the frontend MUST NOT send a duplicate /api/chat request.
    """
    has_answer = bool(result.get("answer"))
    has_errors = len(ensure_list(result.get("errors", []))) > 0
    status = result.get("status")
    # Statuses that explicitly mean "no chat follow-up"
    non_chat_statuses = {"MODEL_NOT_READY", "INDEX_NOT_READY", "NO_VISUAL_MATCH"}
    is_non_chat = status in non_chat_statuses or status is None
    return not has_answer and not has_errors and not is_non_chat


def is_visual_response_complete(payload: dict[str, Any]) -> bool:
    """Checks that a Visual RAG response contains all canonical keys."""
    required = CANONICAL_KEYS
    return required.issubset(set(payload.keys()))


# ============================================================================
# Helpers
# ============================================================================

def string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ensure_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


# ============================================================================
# Response Parser Tests
# ============================================================================

class TestVisualRagResponseParser:
    """Tests for parse_visual_rag_response — the JS-side response parser."""

    def test_extracts_candidate_labels_and_scores(self) -> None:
        payload = {
            "status": "OK",
            "coarse_category": "sensor",
            "coarse_confidence": 0.91,
            "visual_candidates": [
                {
                    "doc_id": "fixture_temp_sensor",
                    "sub_category": "temp_sensor",
                    "coarse_category": "sensor",
                    "score": 0.94,
                    "evidence_image_id": "temp-image-a",
                },
                {
                    "doc_id": "fixture_humidity_sensor",
                    "sub_category": "humidity_sensor",
                    "coarse_category": "sensor",
                    "score": 0.82,
                    "evidence_image_id": "humid-image-b",
                },
            ],
            "retrieved_chunks": [],
            "answer": "This is a temperature sensor used for environmental monitoring.",
            "errors": [],
        }

        parsed = parse_visual_rag_response(payload)

        assert parsed["answer"] == "This is a temperature sensor used for environmental monitoring."
        assert parsed["coarse_category"] == "sensor"
        assert parsed["coarse_confidence"] == 0.91
        assert len(parsed["candidates"]) == 2
        assert parsed["candidates"][0] == {"label": "temp_sensor", "score": 94}
        assert parsed["candidates"][1] == {"label": "humidity_sensor", "score": 82}

    def test_derives_device_class_from_first_candidate(self) -> None:
        """Legacy compatibility: device_class = first candidate sub_category."""
        payload = {
            "status": "OK",
            "visual_candidates": [
                {"sub_category": "smart_camera", "score": 0.95},
                {"sub_category": "ip_camera", "score": 0.72},
            ],
            "answer": "answer text",
        }

        parsed = parse_visual_rag_response(payload)
        assert parsed["device_class"] == "smart_camera"

    def test_preserves_explicit_device_class_over_candidate(self) -> None:
        """When API returns device_class, use it instead of deriving."""
        payload = {
            "status": "OK",
            "device_class": "legacy_device_label",
            "visual_candidates": [
                {"sub_category": "temp_sensor", "score": 0.94},
            ],
            "answer": "answer text",
        }

        parsed = parse_visual_rag_response(payload)
        assert parsed["device_class"] == "legacy_device_label"

    def test_handles_empty_candidates(self) -> None:
        payload = {
            "status": "NO_VISUAL_MATCH",
            "coarse_category": None,
            "coarse_confidence": None,
            "visual_candidates": [],
            "retrieved_chunks": [],
            "answer": None,
            "errors": [],
        }

        parsed = parse_visual_rag_response(payload)
        assert parsed["candidates"] == []
        assert parsed["answer"] is None
        assert parsed["device_class"] is None
        assert parsed["coarse_category"] is None

    def test_handles_null_answer(self) -> None:
        payload = {
            "status": "OK",
            "visual_candidates": [{"sub_category": "sensor", "score": 0.88}],
            "answer": None,
        }

        parsed = parse_visual_rag_response(payload)
        assert parsed["answer"] is None
        assert len(parsed["candidates"]) == 1

    def test_handles_missing_candidate_sub_category(self) -> None:
        """Falls back to coarse_category when sub_category missing."""
        payload = {
            "status": "OK",
            "visual_candidates": [
                {
                    "doc_id": "fixture_unknown",
                    "coarse_category": "smart_lighting",
                    "score": 0.65,
                },
            ],
            "answer": "fallback answer",
        }

        parsed = parse_visual_rag_response(payload)
        assert parsed["candidates"][0]["label"] == "smart_lighting"

    def test_handles_candidate_with_zero_score(self) -> None:
        payload = {
            "status": "NO_VISUAL_MATCH",
            "visual_candidates": [
                {"sub_category": "low_confidence_match", "score": 0.0},
            ],
            "answer": None,
        }

        parsed = parse_visual_rag_response(payload)
        assert parsed["candidates"][0]["score"] == 0

    def test_handles_candidate_with_null_score(self) -> None:
        payload = {
            "status": "NO_VISUAL_MATCH",
            "visual_candidates": [
                {"sub_category": "no_score", "score": None},
            ],
            "answer": None,
        }

        parsed = parse_visual_rag_response(payload)
        assert parsed["candidates"][0]["score"] == 0

    def test_handles_non_list_candidates(self) -> None:
        payload = {
            "status": "OK",
            "visual_candidates": None,  # type: ignore[dict-item]
            "answer": "answer",
        }

        parsed = parse_visual_rag_response(payload)
        assert parsed["candidates"] == []

    def test_handles_every_field_null_or_missing(self) -> None:
        """Parser should never throw on maximally empty payload."""
        parsed = parse_visual_rag_response({})
        assert parsed["status"] is None
        assert parsed["answer"] is None
        assert parsed["candidates"] == []
        assert parsed["device_class"] is None
        assert parsed["errors"] == []


# ============================================================================
# Error Display Tests
# ============================================================================

class TestVisualApiErrorDisplayState:
    """Tests for visual_error_display_text — error code to display text mapping."""

    def test_invalid_image_maps_to_clear_message(self) -> None:
        text = visual_error_display_text("INVALID_IMAGE")
        assert "无效" in text
        assert "重新拍摄" in text

    def test_model_not_ready_maps_to_clear_message(self) -> None:
        text = visual_error_display_text("MODEL_NOT_READY")
        assert "未就绪" in text
        assert "重试" in text

    def test_index_not_ready_maps_to_clear_message(self) -> None:
        text = visual_error_display_text("INDEX_NOT_READY")
        assert "索引" in text
        assert "未就绪" in text

    def test_no_visual_match_maps_to_clear_message(self) -> None:
        text = visual_error_display_text("NO_VISUAL_MATCH")
        assert any(w in text for w in ("匹配", "识别"))
        assert "角度" in text

    def test_unknown_error_code_returns_generic_fallback(self) -> None:
        text = visual_error_display_text("SOME_UNKNOWN_CODE")
        assert "异常" in text
        assert "重试" in text

    def test_all_known_codes_have_non_empty_messages(self) -> None:
        for code in VISUAL_ERROR_CODES:
            msg = visual_error_display_text(code)
            assert msg, f"Error code {code!r} must have a non-empty message"
            # Must not return the raw code as the message
            assert msg != code, (
                f"Error code {code!r} maps to itself instead of a display message"
            )

    def test_empty_error_code_returns_generic_fallback(self) -> None:
        text = visual_error_display_text("")
        assert "异常" in text

    def test_all_known_codes_are_distinct(self) -> None:
        """Each error code should map to a unique display message."""
        messages = {code: visual_error_display_text(code) for code in VISUAL_ERROR_CODES}
        # All should be different from each other
        unique_messages = set(messages.values())
        assert len(unique_messages) == len(VISUAL_ERROR_CODES), (
            f"Error messages are not unique: {messages}"
        )


# ============================================================================
# Duplicate Chat Request Prevention Tests
# ============================================================================

class TestNoDuplicateChatRequest:
    """Tests that the frontend does NOT send a duplicate /api/chat when
    the Visual RAG response already includes an answer."""

    def test_answer_present_means_no_chat_request(self) -> None:
        result = {
            "status": "OK",
            "answer": "This is a comprehensive answer from Visual RAG.",
            "visual_candidates": [{"sub_category": "sensor", "score": 0.9}],
            "errors": [],
        }
        assert not should_trigger_chat_request(result), (
            "When answer is present, must NOT trigger duplicate /api/chat"
        )

    def test_answer_null_but_no_errors_means_chat_request(self) -> None:
        """If somehow answer is null and there are no errors, it may be a
        pre-answer scenario. Frontend may decide to trigger chat (legacy)."""
        result = {
            "status": "OK",
            "answer": None,
            "visual_candidates": [{"sub_category": "sensor", "score": 0.9}],
            "errors": [],
        }
        assert should_trigger_chat_request(result), (
            "Without answer and without errors, fallback chat might be needed"
        )

    def test_errors_present_means_no_chat_request(self) -> None:
        result = {
            "status": "MODEL_NOT_READY",
            "answer": None,
            "visual_candidates": [],
            "errors": [{"code": "MODEL_NOT_READY", "message": "model missing"}],
        }
        assert not should_trigger_chat_request(result), (
            "When errors are present, must NOT trigger /api/chat"
        )

    def test_no_visual_match_no_chat_request(self) -> None:
        result = {
            "status": "NO_VISUAL_MATCH",
            "answer": None,
            "visual_candidates": [],
            "errors": [],
        }
        # NO_VISUAL_MATCH without answer means there's nothing to chat about
        assert not should_trigger_chat_request(result), (
            "NO_VISUAL_MATCH without answer should not trigger chat"
        )

    def test_missing_status_no_chat_request(self) -> None:
        result: dict[str, Any] = {
            "answer": None,
            "visual_candidates": [],
            "errors": [],
        }
        assert not should_trigger_chat_request(result), (
            "Missing status should be treated as server not ready"
        )


# ============================================================================
# Response Completeness Tests (integration with canonical API schema)
# ============================================================================

class TestVisualResponseCompleteness:
    """Tests that Visual RAG responses from the API contain all required fields."""

    def test_canonical_keys_cover_all_required_fields(self) -> None:
        """Ensures the canonical key set is exactly what the frontend expects."""
        expected = {
            "status",
            "coarse_category",
            "coarse_confidence",
            "visual_candidates",
            "retrieved_chunks",
            "answer",
            "errors",
        }
        assert CANONICAL_KEYS == expected, (
            "CANONICAL_KEYS must match the frontend's expected response schema"
        )

    def test_happy_path_payload_is_complete(self) -> None:
        payload = {
            "status": "OK",
            "coarse_category": "sensor",
            "coarse_confidence": 0.91,
            "visual_candidates": [{"sub_category": "sensor", "score": 0.9}],
            "retrieved_chunks": [],
            "answer": "answer",
            "errors": [],
        }
        assert is_visual_response_complete(payload)

    def test_error_payload_is_complete(self) -> None:
        payload = {
            "status": "MODEL_NOT_READY",
            "coarse_category": None,
            "coarse_confidence": None,
            "visual_candidates": [],
            "retrieved_chunks": [],
            "answer": None,
            "errors": [{"code": "MODEL_NOT_READY", "message": "missing"}],
        }
        assert is_visual_response_complete(payload)
