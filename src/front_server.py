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
import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler


# ------------------------------------------------------------------
# RAG 管线 — 后台线程加载（避免首次请求阻塞）
# ------------------------------------------------------------------

_rag_pipeline = None
_rag_lock = threading.Lock()


def _get_rag_pipeline():
    """获取已加载的 RAG 管线（非阻塞）。未就绪时返回 None。"""
    global _rag_pipeline
    if _rag_pipeline is not None:
        return _rag_pipeline if _rag_pipeline is not False else None
    return None


def _preload_rag():
    """在后台线程中加载嵌入模型（10-30s），避免首次 RAG 请求阻塞。"""
    global _rag_pipeline
    sys.stderr.write("[RAG] 正在后台加载嵌入模型...\n")
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
                f"[RAG] 就绪（向量库为空，请运行 python -m src.rag.index）\n"
            )
    except Exception as e:
        with _rag_lock:
            _rag_pipeline = False
        sys.stderr.write(f"[RAG] 加载失败: {e}\n")


class FrontendHandler(SimpleHTTPRequestHandler):
    """处理静态文件 + API 代理。"""

    backend_url: str = "http://127.0.0.1:18080"

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
        if self.path == "/api/chat":
            self._proxy_chat()
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
        try:
            req = urllib.request.Request(f"{self.backend_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp.read())
        except Exception as e:
            self.send_response(503)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"status": "error", "message": str(e)}).encode()
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

    # 后台预加载 RAG 嵌入模型（避免首次 RAG 请求阻塞 10-30s）
    threading.Thread(target=_preload_rag, daemon=True).start()

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


def _get_local_ips() -> list:
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
