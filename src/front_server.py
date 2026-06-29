#!/usr/bin/env python3
"""
AIGText Frontend Server
— 提供前端静态页面 + 反向代理 llama.cpp API + RAG 检索增强
— 绑定 0.0.0.0，支持局域网设备访问

用法:
  python src/front_server.py
  python src/front_server.py --port 9090 --backend http://127.0.0.1:18080

端点:
  GET  /              → frontend/index.html
  GET  /api/health    → 代理 llama-server 健康检查
  POST /api/chat      → 代理 chat/completions（SSE 流式 & 非流式）
                       当 X-RAG-Enabled: true 时，自动注入检索上下文
"""

import argparse
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Callable


# ------------------------------------------------------------------
# RAG 管线 — 后台线程加载（避免首次请求阻塞）
# ------------------------------------------------------------------

_rag_pipeline = None
_rag_lock = threading.Lock()

DEFAULT_VISUAL_QUESTION = "请介绍一下这个物联网设备"
_VISUAL_READY_STATUSES = {
    "OK",
    "LOW_CONFIDENCE",
    "NO_VISUAL_MATCH",
    "NO_TEXT_CHUNKS",
    "MODEL_NOT_READY",
    "INDEX_NOT_READY",
    "VISION_BACKEND_UNAVAILABLE",
}
_VISUAL_HTTP_503_STATUSES = {
    "MODEL_NOT_READY",
    "INDEX_NOT_READY",
    "VISION_BACKEND_UNAVAILABLE",
}
_PROMPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompt",
)
_VISUAL_RAG_PROMPT_FILE = os.path.join(_PROMPT_DIR, "visual_rag_prompt.txt")
_vision_backend_client = None
_vision_backend_lock = threading.Lock()
_llama_answer_provider: Callable[[str], str] | None = None


def _get_rag_pipeline():
    """获取已加载的 RAG 管线（非阻塞）。未就绪时返回 None。"""
    global _rag_pipeline
    if _rag_pipeline is not None:
        return _rag_pipeline
    return None


def _preload_rag():
    """在后台线程中加载嵌入模型（10-30s），避免首次 RAG 请求阻塞。"""
    global _rag_pipeline
    sys.stderr.write("[RAG] 正在后台加载嵌入模型...\n")
    _load_rag_pipeline()


def _preload_rag_sync():
    """同步加载 RAG 嵌入模型，阻塞直到就绪。"""
    global _rag_pipeline
    _load_rag_pipeline()


def _load_rag_pipeline():
    """加载 RAG 管线（嵌入模型 + 向量库），由 _preload_rag 或 _preload_rag_sync 调用。"""
    global _rag_pipeline
    try:
        from src.rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        with _rag_lock:
            _rag_pipeline = pipeline
        chunk_count = pipeline.chunk_count
        if pipeline.is_ready:
            sys.stderr.write(f"[RAG] 就绪 ({chunk_count} chunks)\n")
        else:
            sys.stderr.write(
                f"[RAG] 嵌入模型加载完成，但向量库为空\n"
                f"[RAG] 请运行: python -m src.rag.index\n"
            )
    except Exception as e:
        with _rag_lock:
            _rag_pipeline = None
        sys.stderr.write(f"[RAG] 加载失败: {e}\n")


class FrontendHandler(SimpleHTTPRequestHandler):
    """处理静态文件 + API 代理。"""

    backend_url: str = "http://127.0.0.1:18080"
    static_dir: str = os.getcwd()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.static_dir, **kwargs)

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def do_GET(self):
        if self.path == "/api/health":
            self._proxy_health()
        else:
            super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/chat":
            self._proxy_chat()
        elif path == "/api/vision/query":
            self._handle_visual_query(include_device_class=False)
        elif path == "/api/vision/classify":
            self._handle_visual_query(include_device_class=True)
        else:
            self._send_cors()
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ------------------------------------------------------------------
    # API 代理
    # ------------------------------------------------------------------

    def _proxy_health(self):
        vision_backend = _vision_backend_health_payload()
        try:
            req = urllib.request.Request(f"{self.backend_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                backend_data = json.loads(resp.read())
                # 注入 RAG 就绪状态
                pipeline = _get_rag_pipeline()
                backend_data["rag_ready"] = (
                    pipeline is not None and pipeline.is_ready
                )
                backend_data["vision_backend"] = vision_backend
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(backend_data).encode())
        except Exception as e:
            pipeline = _get_rag_pipeline()
            self.send_response(503)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({
                    "status": "error",
                    "message": str(e),
                    "rag_ready": (
                        pipeline is not None and pipeline.is_ready
                    ),
                    "vision_backend": vision_backend,
                }).encode()
            )

    def _proxy_chat(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_cors()
            self.send_error(400, "Invalid JSON")
            return

        # ---- RAG 检索增强 ----
        rag_enabled = self.headers.get("X-RAG-Enabled", "").lower() == "true"
        augmented = False
        if rag_enabled:
            pipeline = _get_rag_pipeline()
            if pipeline and pipeline.is_ready:
                messages = payload.get("messages", [])
                # 找到最后一条 user 消息
                user_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        user_idx = i
                        break
                if user_idx is not None:
                    user_query = messages[user_idx].get("content", "")
                    chunks = pipeline.retrieve(user_query)
                    if chunks:
                        augmented_text = pipeline.augment(user_query, chunks)
                        messages[user_idx] = {
                            "role": "user",
                            "content": augmented_text,
                        }
                        augmented = True
                        msg = f"[RAG] 检索到 {len(chunks)} 个相关片段"
                    else:
                        msg = "[RAG] 未检索到相关文档，使用原始查询"
                    sys.stderr.write(f"[{self.log_date_time_string()}] {msg}\n")
            elif rag_enabled:
                sys.stderr.write(
                    f"[{self.log_date_time_string()}] [RAG] 向量库未就绪，请先运行: python -m src.rag.index\n"
                )

        is_stream = payload.get("stream", False)
        backend_body = json.dumps(payload).encode("utf-8")
        backend_req = urllib.request.Request(
            f"{self.backend_url}/v1/chat/completions",
            data=backend_body,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(backend_req, timeout=300) as backend_resp:
                if is_stream:
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    try:
                        for raw_line in backend_resp:
                            self.wfile.write(raw_line)
                            self.wfile.flush()
                    except Exception:
                        pass  # client disconnected, ignore
                    # HTTP/1.0 — connection closes naturally here
                else:
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(backend_resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = e.read() if e.fp else b'{"error": "backend error"}'
            self.wfile.write(body)
        except Exception as e:
            self.send_response(502)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_visual_query(self, *, include_device_class: bool) -> None:
        try:
            image_bytes, question, top_k = _parse_visual_multipart_request(self)
        except ValueError as exc:
            self._send_json(400, _visual_error_response("INVALID_IMAGE", str(exc)))
            return

        try:
            payload = _run_dual_host_visual_query(image_bytes, question, top_k)
        except _vision_backend_unavailable_errors() as exc:
            self._send_json(
                503,
                _visual_error_response(
                    "VISION_BACKEND_UNAVAILABLE",
                    str(exc),
                    retryable=True,
                ),
            )
            return
        except Exception as exc:
            self._send_json(503, _visual_error_response("MODEL_NOT_READY", str(exc)))
            return

        status = _visual_api_status(payload)
        response_payload = _visual_response_payload(
            payload,
            status=status,
            answer=None,
            errors=[],
            include_device_class=include_device_class,
        )

        if _visual_pipeline_status(payload) in _VISUAL_HTTP_503_STATUSES:
            response_payload["errors"] = [
                {"code": status, "message": _visual_status_message(status)}
            ]
            self._send_json(503, response_payload)
            return

        prompt = _string_or_none(payload.get("augmented_prompt"))
        if prompt:
            try:
                response_payload["answer"] = _request_llama_answer(self.backend_url, prompt)
            except Exception as exc:
                error_payload = _visual_response_payload(
                    payload,
                    status="MODEL_NOT_READY",
                    answer=None,
                    errors=[{"code": "MODEL_NOT_READY", "message": str(exc)}],
                    include_device_class=include_device_class,
                )
                self._send_json(503, error_payload)
                return

        self._send_json(200, response_payload)

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self._send_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, X-RAG-Enabled"
        )

    def log_message(self, format: str, *args) -> None:
        # 静默静态资源日志，只打印 API 调用
        msg = format % args
        if "/api/" in msg or "404" in msg:
            sys.stderr.write(f"[{self.log_date_time_string()}] {msg}\n")


# ------------------------------------------------------------------
# Visual RAG API helpers
# ------------------------------------------------------------------

def _get_vision_backend_client():
    global _vision_backend_client
    if _vision_backend_client is not None:
        return _vision_backend_client

    with _vision_backend_lock:
        if _vision_backend_client is None:
            from src.vision.vision_client import VisionBackendClient
            _vision_backend_client = VisionBackendClient()
    return _vision_backend_client


def _parse_visual_multipart_request(
    handler: SimpleHTTPRequestHandler,
) -> tuple[bytes, str, int | None]:
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("Expected multipart/form-data with an image field")

    content_length = int(handler.headers.get("Content-Length", 0))
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
    question_value: str | None = None
    query_value: str | None = None
    top_k: int | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name == "image":
            raw_image = part.get_payload(decode=True)
            image_bytes = raw_image if isinstance(raw_image, bytes) else b""
        elif name in {"question", "query", "top_k"}:
            raw_value = part.get_payload(decode=True)
            value_bytes = raw_value if isinstance(raw_value, bytes) else b""
            charset = part.get_content_charset() or "utf-8"
            decoded_value = value_bytes.decode(charset, errors="replace").strip()
            if name == "question" and decoded_value:
                question_value = decoded_value
            elif name == "query" and decoded_value:
                query_value = decoded_value
            elif name == "top_k" and decoded_value:
                top_k = _parse_positive_int(decoded_value, "top_k")

    if not image_bytes:
        raise ValueError("Missing required image upload")
    _validate_image_upload(image_bytes)
    return image_bytes, question_value or query_value or DEFAULT_VISUAL_QUESTION, top_k


def _parse_positive_int(value: str, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}") from exc
    if parsed <= 0:
        raise ValueError(f"Invalid {field_name}")
    return parsed


def _validate_image_upload(image_bytes: bytes) -> None:
    try:
        from PIL import Image
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("INVALID_IMAGE") from exc


def _run_dual_host_visual_query(
    image_bytes: bytes,
    question: str,
    top_k: int | None,
) -> dict[str, object]:
    visual_result = _get_vision_backend_client().search(
        image_bytes,
        question=question,
        top_k=top_k,
    )
    visual_candidates = _candidate_list(visual_result.get("visual_candidates"))

    if not visual_candidates:
        return _visual_payload(
            visual_result=visual_result,
            visual_candidates=[],
            retrieved_chunks=[],
            augmented_prompt="",
            status=_empty_visual_status(visual_result),
        )

    doc_ids = _candidate_doc_ids(visual_candidates)
    text_pipeline = _get_rag_pipeline()
    retrieved_chunks: list[dict[str, object]] = []
    if text_pipeline is not None and text_pipeline.is_ready:
        retrieved_chunks = [
            dict(chunk)
            for chunk in text_pipeline.retrieve(question, doc_ids=doc_ids)
            if isinstance(chunk, dict)
        ]

    return _visual_payload(
        visual_result=visual_result,
        visual_candidates=visual_candidates,
        retrieved_chunks=retrieved_chunks,
        augmented_prompt=_augment_visual_prompt(
            question,
            visual_candidates,
            retrieved_chunks,
        ),
        status="OK" if retrieved_chunks else "NO_TEXT_CHUNKS",
    )


def _visual_payload(
    *,
    visual_result: dict[str, object],
    visual_candidates: list[dict[str, object]],
    retrieved_chunks: list[dict[str, object]],
    augmented_prompt: str,
    status: str,
) -> dict[str, object]:
    return {
        "coarse_category": _string_or_none(visual_result.get("coarse_category")),
        "coarse_confidence": _float_or_none(visual_result.get("coarse_confidence")),
        "coarse_status": _string_or_none(visual_result.get("coarse_status")),
        "visual_candidates": visual_candidates,
        "retrieved_chunks": retrieved_chunks,
        "augmented_prompt": augmented_prompt,
        "status": status,
    }


def _candidate_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    candidates: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if not _string_or_none(item.get("doc_id")):
            continue
        candidates.append(dict(item))
    return candidates


def _candidate_doc_ids(candidates: list[dict[str, object]]) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        doc_id = _string_or_none(candidate.get("doc_id"))
        if not doc_id or doc_id in seen:
            continue
        doc_ids.append(doc_id)
        seen.add(doc_id)
    return doc_ids


def _empty_visual_status(visual_result: dict[str, object]) -> str:
    status = _string_or_none(visual_result.get("status"))
    if status and status != "OK":
        return status
    return "NO_VISUAL_MATCH"


def _augment_visual_prompt(
    question: str,
    visual_candidates: list[dict[str, object]],
    chunks: list[dict[str, object]],
) -> str:
    context = _format_text_sources(chunks)
    if not context:
        context = "未检索到可引用的候选文档文本片段。"
    return _load_visual_prompt_template().format(
        candidate_summary=_format_candidate_summary(visual_candidates),
        context=context,
        query=question,
    )


def _load_visual_prompt_template() -> str:
    try:
        with open(_VISUAL_RAG_PROMPT_FILE, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read()
    except FileNotFoundError:
        return (
            "你是一个物联网设备识别与说明助手。\n\n"
            "视觉候选设备：\n{candidate_summary}\n\n"
            "参考资料：\n{context}\n\n"
            "用户问题：{query}\n\n"
            "请基于参考资料回答，并使用 [1]、[2] 这样的来源编号。"
        )


def _format_candidate_summary(candidates: list[dict[str, object]]) -> str:
    if not candidates:
        return "无视觉候选设备。"

    parts: list[str] = []
    for index, candidate in enumerate(candidates, 1):
        score = _float_or_none(candidate.get("score")) or 0.0
        matched_image_count = int(
            _float_or_none(candidate.get("matched_image_count")) or 0.0
        )
        parts.append("".join([
            f"{index}. doc_id={_string_or_none(candidate.get('doc_id')) or ''}; ",
            f"粗类别={_string_or_none(candidate.get('coarse_category')) or ''}; ",
            f"子类别={_string_or_none(candidate.get('sub_category')) or ''}; ",
            f"视觉分数={score:.2f}; ",
            f"证据图片={_string_or_none(candidate.get('evidence_image_id')) or ''}; ",
            f"匹配图片数={matched_image_count}",
        ]))
    return "\n".join(parts)


def _format_text_sources(chunks: list[dict[str, object]]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        score = _float_or_none(chunk.get("score")) or 0.0
        parts.append("".join([
            f"[{index}] {_string_or_none(chunk.get('text')) or ''}\n",
            f"   (来源: {_string_or_none(chunk.get('source')) or ''}, ",
            f"相关度: {score:.2f})",
        ]))
    return "\n\n".join(parts)


def _vision_backend_health_payload() -> dict[str, object]:
    try:
        payload = _get_vision_backend_client().health()
    except _vision_backend_unavailable_errors() as exc:
        return {
            "status": "unavailable",
            "reachable": False,
            "error_code": "VISION_BACKEND_UNAVAILABLE",
            "message": str(exc),
        }
    except Exception as exc:
        return {
            "status": "error",
            "reachable": False,
            "error_code": "VISION_BACKEND_ERROR",
            "message": str(exc),
        }

    status = _string_or_none(payload.get("status")) or "ok"
    return {
        "status": status,
        "reachable": True,
    }


def _vision_backend_unavailable_errors() -> tuple[type[BaseException], ...]:
    try:
        from src.vision.vision_client import VisionBackendUnavailable
        return (VisionBackendUnavailable, TimeoutError)
    except Exception:
        return (TimeoutError,)


def _visual_api_status(payload: dict[str, object]) -> str:
    status = _visual_pipeline_status(payload)
    coarse_status = _string_or_none(payload.get("coarse_status"))
    if status in _VISUAL_HTTP_503_STATUSES:
        return status
    if coarse_status in _VISUAL_HTTP_503_STATUSES:
        return coarse_status
    if status == "OK" and coarse_status == "LOW_CONFIDENCE":
        return "LOW_CONFIDENCE"
    if status in _VISUAL_READY_STATUSES:
        return status
    return "MODEL_NOT_READY"


def _visual_pipeline_status(payload: dict[str, object]) -> str:
    return _string_or_none(payload.get("status")) or "MODEL_NOT_READY"


def _visual_response_payload(
    payload: dict[str, object],
    *,
    status: str,
    answer: str | None,
    errors: list[dict[str, object]],
    include_device_class: bool,
) -> dict[str, object]:
    visual_candidates = _serialize_visual_candidates(
        payload.get("visual_candidates")
    )
    response: dict[str, object] = {
        "status": status,
        "coarse_category": _string_or_none(payload.get("coarse_category")),
        "coarse_confidence": _float_or_none(payload.get("coarse_confidence")),
        "visual_candidates": visual_candidates,
        "retrieved_chunks": _serialize_retrieved_chunks(
            payload.get("retrieved_chunks")
        ),
        "answer": answer,
        "errors": errors,
    }
    if include_device_class:
        response["device_class"] = _device_class(visual_candidates)
    return response


def _visual_error_response(
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> dict[str, object]:
    error: dict[str, object] = {"code": code, "message": message}
    if retryable:
        error["retryable"] = True
    return {
        "status": code if code in _VISUAL_READY_STATUSES else "MODEL_NOT_READY",
        "coarse_category": None,
        "coarse_confidence": None,
        "visual_candidates": [],
        "retrieved_chunks": [],
        "answer": None,
        "errors": [error],
    }


def _serialize_visual_candidates(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    candidates: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        candidates.append({
            "doc_id": _string_or_none(item.get("doc_id")) or "",
            "sub_category": _string_or_none(item.get("sub_category")) or "",
            "coarse_category": _string_or_none(item.get("coarse_category")) or "",
            "score": _float_or_none(item.get("score")) or 0.0,
            "evidence_image_id": _string_or_none(item.get("evidence_image_id")) or "",
        })
    return candidates


def _serialize_retrieved_chunks(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    chunks: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        chunks.append({
            "doc_id": _string_or_none(item.get("doc_id")) or "",
            "source": _string_or_none(item.get("source")) or "",
            "chunk_id": _int_or_zero(item.get("chunk_id")),
            "score": _float_or_none(item.get("score")) or 0.0,
            "text": _string_or_none(item.get("text")) or "",
        })
    return chunks


def _device_class(candidates: list[dict[str, object]]) -> str | None:
    if not candidates:
        return None
    first = candidates[0]
    sub_category = _string_or_none(first.get("sub_category"))
    if sub_category:
        return sub_category
    return _string_or_none(first.get("coarse_category"))


def _request_llama_answer(backend_url: str, prompt: str) -> str:
    if _llama_answer_provider is not None:
        return _llama_answer_provider(prompt)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False,
    }
    backend_req = urllib.request.Request(
        f"{backend_url}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(backend_req, timeout=300) as backend_resp:
        raw_body = backend_resp.read()
    response_payload = json.loads(raw_body)
    answer = _extract_llama_answer(response_payload)
    if answer is None:
        raise RuntimeError("llama backend response did not include an answer")
    return answer


def _extract_llama_answer(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first_choice.get("text")
            if isinstance(text, str):
                return text
    content = payload.get("content")
    if isinstance(content, str):
        return content
    return None


def _visual_status_message(status: str) -> str:
    if status == "VISION_BACKEND_UNAVAILABLE":
        return "Vision backend is unavailable"
    if status == "INDEX_NOT_READY":
        return "Visual or text index is not ready"
    if status == "MODEL_NOT_READY":
        return "Visual RAG model is not ready"
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


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_static = os.path.join(project_root, "frontend")

    parser = argparse.ArgumentParser(
        description="AIGText Frontend Server — 前端 + API 代理",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="监听端口 (默认: 8080)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="绑定地址 (默认: 0.0.0.0, 允许局域网访问)",
    )
    parser.add_argument(
        "--backend",
        default="http://127.0.0.1:18080",
        help="llama-server 地址 (默认: http://127.0.0.1:18080)",
    )
    parser.add_argument(
        "--static",
        default=default_static,
        help=f"前端文件目录 (默认: {default_static})",
    )
    args = parser.parse_args()

    static_dir = os.path.abspath(args.static)
    if not os.path.isdir(static_dir):
        print(f"[ERROR] 静态文件目录不存在: {static_dir}", file=sys.stderr)
        sys.exit(1)

    FrontendHandler.backend_url = args.backend.rstrip("/")
    FrontendHandler.static_dir = static_dir

    server = HTTPServer((args.host, args.port), FrontendHandler)

    # 获取本机局域网 IP
    local_ips = _get_local_ips()

    print()
    print("=" * 52)
    print("  AIGText — Frontend Server")
    print("=" * 52)
    print(f"  后端服务:  {FrontendHandler.backend_url}")
    print(f"  静态目录:  {static_dir}")
    print("=" * 52)
    print()
    print("  本机访问:")
    print(f"    http://localhost:{args.port}")
    print(f"    http://127.0.0.1:{args.port}")
    if local_ips:
        print()
        print("  局域网访问 (分享给同事/手机):")
        for ip in local_ips:
            print(f"    http://{ip}:{args.port}")
    else:
        print()
        print("  [WARN] 未检测到局域网 IP，请检查网络连接")
    print()
    print("  [注意] 如局域网设备无法访问，请在 Windows 防火墙中")
    print(f"         允许 Python 通过端口 {args.port}：")
    print("         控制面板 → 防火墙 → 高级设置 → 入站规则 → 新建规则")
    print(f"         端口 → TCP → {args.port} → 允许连接")
    print("=" * 52)
    print()
    print("按 Ctrl+C 停止服务")
    print()

    # 同步加载 RAG 嵌入模型，确保就绪后再启动服务
    sys.stderr.write("[RAG] 正在加载嵌入模型（约 10-30s）...\n")
    sys.stderr.flush()
    _preload_rag_sync()

    # 自动打开浏览器
    _open_browser(f"http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
        server.shutdown()


def _open_browser(url: str) -> None:
    """尝试打开默认浏览器（失败静默忽略）。"""
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _get_local_ips() -> list[str]:
    """获取本机所有局域网 IPv4 地址（排除 127.x.x.x）。"""
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip != "127.0.0.1":
                ips.append(ip)
    except Exception:
        pass

    # 去重并保持顺序
    seen = set()
    unique = []
    for ip in ips:
        if ip not in seen:
            seen.add(ip)
            unique.append(ip)
    return unique


if __name__ == "__main__":
    main()
