#!/usr/bin/env python3
"""
本地模型客户端封装 — LlamaCppChatClient

通过 llama-server 的 OpenAI 兼容 HTTP API 提供聊天与健康检查。
不包含终端输出、键盘处理、CLI 交互逻辑。
"""

import json
import urllib.request
import urllib.error
from typing import Generator, Optional, Dict, Any, List


class LlamaCppChatClient:
    """封装 llama.cpp server (OpenAI-compatible API) 的聊天客户端。"""

    def __init__(
        self,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """检查 server 是否可用。"""
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 非流式聊天
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """非流式聊天。返回完整回复文本；连接/协议异常时返回 None。"""
        payload = {
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
                choices = body.get("choices", [])
                if not choices:
                    return None
                return choices[0].get("message", {}).get("content", "")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 流式聊天
    # ------------------------------------------------------------------

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """流式聊天（SSE）。逐 token 产出文本片段。

        调用方负责：终端输出、ESC 中断检测、文本累积。
        发生连接/协议异常时向上抛出，由调用方处理。
        """
        payload = {
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                    if choice.get("finish_reason") == "stop":
                        return
                except (json.JSONDecodeError, KeyError):
                    continue
