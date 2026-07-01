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
import logging
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("front_server")
from socketserver import ThreadingMixIn
from typing import Callable


# 确保项目根目录在 sys.path 以支持 `python src/front_server.py` 直接运行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.prompt_loader import render_prompt  # noqa: E402


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server — prevents long-running requests (e.g. 3D
    modeling) from blocking concurrent requests (e.g. chat, health check)."""

    allow_reuse_address = True
    daemon_threads = True


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
    data_dir: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.static_dir, **kwargs)

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------

    def do_GET(self):
        if self.path == "/api/health":
            self._proxy_health()
        elif self.path == "/api/ping":
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/api/conversations":
            self._list_conversations()
        elif self.path.startswith("/api/conversations/"):
            conv_id = self.path[len("/api/conversations/"):]
            self._get_conversation(conv_id)
        elif self.path == "/api/learning/messages":
            self._handle_learning_messages()
        elif self.path == "/api/knowledge-graph":
            self._serve_knowledge_graph()
        elif self.path.startswith("/data/") or self.path == "/data":
            self._serve_data_file()
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
        elif path == "/api/vision/compare":
            self._handle_visual_compare()
        elif path == "/api/conversations":
            self._create_conversation()
        elif path.startswith("/api/conversations/"):
            conv_id = path[len("/api/conversations/"):]
            self._save_conversation(conv_id)
        else:
            self._send_cors()
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/conversations/"):
            conv_id = path[len("/api/conversations/"):]
            self._delete_conversation(conv_id)
        else:
            self._send_cors()
            self.send_error(404, "Not Found")

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
                body = json.dumps(backend_data).encode("utf-8")
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, OSError) as e:
            # 客户端在响应完成前断开连接，正常现象，静默忽略
            logger.debug("[health] client disconnected during write: %s", e)
        except Exception as e:
            logger.warning("[health] backend proxy error: %s", e)
            try:
                pipeline = _get_rag_pipeline()
                err_body = json.dumps({
                    "status": "error",
                    "message": str(e),
                    "rag_ready": (
                        pipeline is not None and pipeline.is_ready
                    ),
                    "vision_backend": vision_backend,
                }).encode()
                self.send_response(503)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
            except (ConnectionAbortedError, BrokenPipeError, OSError):
                pass

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
        voice_mode = payload.get("voice_mode", False)
        augmented = False
        if rag_enabled:
            pipeline = _get_rag_pipeline()
            if pipeline and pipeline.is_ready:
                try:
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
                            # 语音模式使用 voice_prompt 模板（无 CoT，简洁口语）
                            augment_kwargs = {"prompt_name": "voice_prompt"} if voice_mode else {}
                            augmented_text = pipeline.augment(user_query, chunks, **augment_kwargs)
                            messages[user_idx] = {
                                "role": "user",
                                "content": augmented_text,
                            }
                            augmented = True
                            msg = f"[RAG] 检索到 {len(chunks)} 个相关片段"
                        else:
                            msg = "[RAG] 未检索到相关文档，使用原始查询"
                        sys.stderr.write(f"[{self.log_date_time_string()}] {msg}\n")
                except Exception as e:
                    import traceback
                    sys.stderr.write(
                        f"[{self.log_date_time_string()}] [RAG] 检索失败: {e}\n"
                        f"{traceback.format_exc()}\n"
                    )
                    # RAG 失败不影响正常对话
            elif rag_enabled:
                sys.stderr.write(
                    f"[{self.log_date_time_string()}] [RAG] 向量库未就绪，请先运行: python -m src.rag.index\n"
                )

        # 语音模式：注入简洁口语系统提示，禁止 CoT
        if voice_mode:
            try:
                messages = payload.get("messages", [])
                voice_system = render_prompt("voice_prompt", context="", query="")
                has_system = messages and messages[0].get("role") == "system"
                if has_system:
                    messages[0]["content"] = voice_system
                else:
                    messages.insert(0, {"role": "system", "content": voice_system})
                sys.stderr.write(f"[{self.log_date_time_string()}] [Voice] 已注入语音模式系统提示（无 CoT）\n")
            except Exception:
                pass  # 非致命

        # 移除前端专属字段，避免传入 LLM 后端
        payload.pop("voice_mode", None)

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
            self._send_json(_visual_error_response("INVALID_IMAGE", str(exc)), 400)
            return

        try:
            payload = _run_dual_host_visual_query(image_bytes, question, top_k)
        except _vision_backend_unavailable_errors() as exc:
            self._send_json(
                _visual_error_response(
                    "VISION_BACKEND_UNAVAILABLE",
                    str(exc),
                    retryable=True,
                ),
                503,
            )
            return
        except Exception as exc:
            self._send_json(_visual_error_response("MODEL_NOT_READY", str(exc)), 503)
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
            self._send_json(response_payload, 503)
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
                self._send_json(error_payload, 503)
                return

        self._send_json(response_payload, 200)

    def _handle_visual_compare(self) -> None:
        """POST /api/vision/compare — 多图设备对比（支持流式输出）"""
        try:
            images, question, top_k, is_stream = _parse_compare_multipart(self)
        except ValueError as exc:
            self._send_json(_visual_error_response("INVALID_IMAGE", str(exc)), 400)
            return

        visual_results: list[dict[str, object]] = []

        try:
            for i, img_bytes in enumerate(images):
                try:
                    result = _run_dual_host_visual_query(img_bytes, question, top_k)
                    candidates = _candidate_list(result.get("visual_candidates"))
                    raw_chunks = result.get("retrieved_chunks", [])
                    chunks = [
                        dict(chunk)
                        for chunk in raw_chunks
                        if isinstance(chunk, dict)
                    ] if isinstance(raw_chunks, list) else []
                    visual_results.append({
                        "image_index": i,
                        "candidates": candidates,
                        "chunks": chunks,
                        "status": result.get("status", "OK"),
                    })
                except Exception as exc:
                    visual_results.append({
                        "image_index": i,
                        "candidates": [],
                        "chunks": [],
                        "status": "VISION_BACKEND_UNAVAILABLE",
                        "error": str(exc),
                    })
        except _vision_backend_unavailable_errors() as exc:
            self._send_json(
                _visual_error_response("VISION_BACKEND_UNAVAILABLE", str(exc), retryable=True),
                503,
            )
            return

        if not any(r.get("candidates") for r in visual_results):
            if is_stream:
                self._send_json({
                    "status": "NO_VISUAL_MATCH",
                    "message": "所有图片均未识别到设备",
                    "visual_results": visual_results,
                    "answer": "",
                }, 200)
                return
            self._send_json({
                "status": "NO_VISUAL_MATCH",
                "message": "所有图片均未识别到设备",
                "visual_results": visual_results,
                "answer": "",
            }, 200)
            return

        # 构建对比提示词
        parts: list[str] = []
        for vr in visual_results:
            i = vr["image_index"]
            cands = vr.get("candidates", [])
            chunks_list = vr.get("chunks", [])
            img_num = int(i) + 1 if isinstance(i, int) else 1
            parts.append(f"## 图片 {img_num}\n")
            if isinstance(cands, list) and cands:
                parts.append("视觉识别结果：\n")
                parts.append(_format_candidate_summary(cands))
                parts.append("\n")
            if isinstance(chunks_list, list) and chunks_list:
                parts.append("参考资料：\n")
                parts.append(_format_text_sources(chunks_list))
                parts.append("\n")
            else:
                parts.append("参考资料：未检索到可引用的文档文本片段。\n")
        comparison_data = "\n".join(parts)

        prompt = render_prompt("compare_prompt", comparison_data=comparison_data, query=question or "")

        llm_payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": is_stream,
        }
        llm_body = json.dumps(llm_payload, ensure_ascii=False).encode("utf-8")
        llm_req = urllib.request.Request(
            f"{self.backend_url}/v1/chat/completions",
            data=llm_body,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(llm_req, timeout=300) as llm_resp:
                if is_stream:
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    try:
                        for raw_line in llm_resp:
                            self.wfile.write(raw_line)
                            self.wfile.flush()
                    except Exception:
                        pass  # 客户端断开
                else:
                    raw_body = llm_resp.read()
                    response_payload = json.loads(raw_body)
                    answer = _extract_llama_answer(response_payload)
                    if answer is None:
                        self._send_json({
                            "status": "MODEL_NOT_READY",
                            "message": "LLM backend response did not include an answer",
                            "visual_results": visual_results,
                            "answer": "",
                        }, 503)
                        return
                    self._send_json({
                        "status": "OK",
                        "visual_results": visual_results,
                        "answer": answer,
                    }, 200)
        except urllib.error.HTTPError as exc:
            body = exc.read() if exc.fp else b'{"error":"llm backend error"}'
            self._send_cors()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self._send_json({
                "status": "MODEL_NOT_READY",
                "message": str(exc),
                "visual_results": visual_results,
                "answer": "",
            }, 503)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, X-RAG-Enabled"
        )

    # ---- JSON helpers ----
    def _send_json(self, data, status=200):
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        payload = json.dumps(data).encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    # ---- Conversation API ----
    def _handle_learning_messages(self):
        """GET /api/learning/messages — 返回所有对话的所有消息（一次查询，避免 N+1 请求）"""
        from src.db import database as db
        self._send_json(db.get_all_messages())

    def _list_conversations(self):
        """GET /api/conversations — 返回所有对话列表"""
        from src.db import database as db
        self._send_json(db.list_conversations())

    def _get_conversation(self, conv_id: str):
        """GET /api/conversations/:id — 返回单个对话含消息"""
        from src.db import database as db
        conv = db.get_conversation(conv_id)
        if not conv:
            self._send_json({"error": "Conversation not found"}, 404)
        else:
            self._send_json(conv)

    def _create_conversation(self):
        """POST /api/conversations — 创建新对话"""
        try:
            body = self._read_json_body()
        except Exception as e:
            self._send_json({"error": "Invalid JSON: " + str(e)}, 400)
            return
        from src.db import database as db
        title = body.get("title") or "新对话"
        conv = db.create_conversation(title=title)
        self._send_json(conv, 201)

    def _save_conversation(self, conv_id: str):
        """POST /api/conversations/:id — 保存/更新对话消息"""
        try:
            body = self._read_json_body()
        except Exception as e:
            self._send_json({"error": "Invalid JSON: " + str(e)}, 400)
            return
        from src.db import database as db
        messages = body.get("messages", [])
        db.save_messages(conv_id, messages)
        if body.get("title"):
            db.update_conversation_title(conv_id, body["title"])
        self._send_json({"ok": True})

    def _delete_conversation(self, conv_id: str):
        """DELETE /api/conversations/:id — 删除对话"""
        from src.db import database as db
        db.delete_conversation(conv_id)
        self._send_json({"ok": True})

    # ---- Knowledge Graph API ----
    def _serve_knowledge_graph(self):
        """GET /api/knowledge-graph — 返回物联网设备知识图谱数据"""
        graph_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "iot_knowledge", "iot_taxonomy.json"
        )
        if not os.path.isfile(graph_path):
            self._send_json({"error": "Knowledge graph not found"}, 404)
            return
        with open(graph_path, "r", encoding="utf-8") as f:
            import re
            data = json.loads(re.sub(r'^\s*\/\/.*$', '', f.read(), flags=re.MULTILINE))
        self._send_json(data)

    # ---- 静态数据文件服务 ----
    def _serve_data_file(self):
        """GET /data/<rel_path> — 将 project_root/data/ 目录作为静态目录提供服务。

        允许前端通过 fetch("/data/...") 直接读取项目根目录下的 data/ 文件
        (如 taxonomy.json), 而不是通过 API 接口。"""
        import urllib.parse

        path = self.path.split("?", 1)[0]
        # 解析 URL 路径, 获取 /data/ 之后的相对路径
        rel_path = urllib.parse.unquote(path[len("/data/"):]) if path.startswith("/data/") else ""

        # 路径规范化, 防止 ../.. 等路径穿越攻击
        try:
            import posixpath
            rel_normalized = posixpath.normpath(rel_path)
            # normpath 不会去掉开头的 .. 或 /, 需要额外过滤
            if rel_normalized.startswith("..") or rel_normalized.startswith("/"):
                raise ValueError("invalid path")
        except ValueError:
            self.send_error(400, "Invalid path")
            return

        # 解析绝对路径并验证在 data_dir 范围内
        if self.data_dir and os.path.isdir(self.data_dir):
            full_path = os.path.normpath(os.path.join(self.data_dir, rel_normalized))
            data_dir_real = os.path.realpath(self.data_dir)
            full_path_real = os.path.realpath(full_path)
            if not full_path_real.startswith(data_dir_real + os.sep) and full_path_real != data_dir_real:
                self.send_error(403, "Forbidden")
                return
            if not os.path.isfile(full_path_real):
                self.send_error(404, "File not found")
                return
        else:
            self.send_error(500, "Data directory not configured")
            return

        # 猜测 MIME 类型
        import mimetypes
        mime_type, _ = mimetypes.guess_type(full_path_real)
        if not mime_type:
            mime_type = "application/octet-stream"

        try:
            with open(full_path_real, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(500, "Error reading file")
            return

        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        # 静默静态资源日志，只打印 API 调用
        msg = format % args
        if "/api/" in msg or "404" in msg:
            sys.stderr.write(f"[{self.log_date_time_string()}] {msg}\n")


# ------------------------------------------------------------------
# Visual RAG API helpers
# ------------------------------------------------------------------

def _get_vision_backend_client():
    """Get or create vision backend client singleton."""
    global _vision_backend_client
    if _vision_backend_client is not None:
        return _vision_backend_client

    with _vision_backend_lock:
        if _vision_backend_client is None:
            from src.vision.vision_client import VisionBackendClient
            _vision_backend_client = VisionBackendClient()
            logger.info(
                "[vision] client created: base_url=%s timeout=%.1fs",
                _vision_backend_client._base_url,
                _vision_backend_client._timeout_seconds,
            )
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


def _parse_compare_multipart(
    handler: SimpleHTTPRequestHandler,
) -> tuple[list[bytes], str, int | None, bool]:
    """解析对比模式的多图片 multipart 请求，返回 (图片列表, 问题, top_k, stream)。"""
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("Expected multipart/form-data with image fields")

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

    images: list[bytes] = []
    question_value: str | None = None
    query_value: str | None = None
    top_k: int | None = None
    stream_value: bool = False

    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        img_names = {"image_0", "image_1", "image_2"}
        if name in img_names:
            raw_image = part.get_payload(decode=True)
            img_bytes = raw_image if isinstance(raw_image, bytes) else b""
            if img_bytes:
                images.append(img_bytes)
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
        elif name == "stream":
            raw_value = part.get_payload(decode=True)
            value_bytes = raw_value if isinstance(raw_value, bytes) else b""
            stream_value = value_bytes.decode("utf-8", errors="replace").strip().lower() == "true"

    if len(images) < 2:
        raise ValueError("对比模式需要至少 2 张图片")
    if len(images) > 3:
        images = images[:3]

    for img_bytes in images:
        _validate_image_upload(img_bytes)

    return images, question_value or query_value or DEFAULT_VISUAL_QUESTION, top_k, stream_value


def _run_dual_host_visual_query(
    image_bytes: bytes,
    question: str,
    top_k: int | None,
) -> dict[str, object]:
    logger.info(
        "[vision] query start: image=%dB question=%r top_k=%s",
        len(image_bytes),
        question[:80],
        top_k,
    )
    try:
        visual_result = _get_vision_backend_client().search(
            image_bytes,
            question=question,
            top_k=top_k,
        )
    except _vision_backend_unavailable_errors() as exc:
        logger.error("[vision] query FAILED (unavailable): %s", exc)
        raise
    except Exception as exc:
        logger.error("[vision] query ERROR: %s", exc, exc_info=True)
        raise

    visual_candidates = _candidate_list(visual_result.get("visual_candidates"))
    logger.info(
        "[vision] query got %d candidate(s), doc_ids=%s",
        len(visual_candidates),
        _candidate_doc_ids(visual_candidates),
    )

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
    return render_prompt(
        "visual_rag_prompt",
        candidate_summary=_format_candidate_summary(visual_candidates),
        context=context,
        query=question,
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
        logger.info("[vision] health check result: %s", json.dumps(payload, ensure_ascii=False))
    except _vision_backend_unavailable_errors() as exc:
        logger.warning("[vision] health check FAILED (unavailable): %s", exc)
        return {
            "status": "unavailable",
            "reachable": False,
            "error_code": "VISION_BACKEND_UNAVAILABLE",
            "message": str(exc),
        }
    except Exception as exc:
        logger.error("[vision] health check ERROR: %s", exc, exc_info=True)
        return {
            "status": "error",
            "reachable": False,
            "error_code": "VISION_BACKEND_ERROR",
            "message": str(exc),
        }

    status = _string_or_none(payload.get("status")) or "ok"
    logger.info("[vision] health: status=%s reachable=True", status)
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
# 入口辅助函数
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
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
    FrontendHandler.data_dir = os.path.join(project_root, "data")

    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)

    # 获取本机局域网 IP
    local_ips = _get_local_ips()

    lan_urls = ", ".join(f"http://{ip}:{args.port}" for ip in local_ips) if local_ips else "无"
    print(f"AIGText http://localhost:{args.port}  LAN: {lan_urls}", file=sys.stderr)

    # RAG 后台加载，不阻塞 HTTP 服务启动
    threading.Thread(target=_preload_rag, name="rag-loader", daemon=True).start()

    # 初始化对话数据库
    try:
        from src.db.database import init_db
        init_db()
        sys.stderr.write("[DB] 数据库已初始化\n")
    except Exception as e:
        sys.stderr.write(f"[DB] 数据库初始化失败: {e}\n")

    # 自动打开浏览器（延迟 0.5s 避免 serve_forever 尚未就绪的竞态）
    url = f"http://localhost:{args.port}"
    threading.Timer(0.5, lambda u=url: _open_browser(u)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
        server.shutdown()


if __name__ == "__main__":
    main()
