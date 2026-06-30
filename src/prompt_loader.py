"""
Central prompt loader — 按文件名从 prompt/ 目录加载提示词模板。

模板使用 Python str.format() 语法，以 {变量名} 作为占位符。

用法:
    from src.prompt_loader import render_prompt

    text = render_prompt("rag_prompt", context=..., query=...)
"""

from __future__ import annotations

import os
from functools import lru_cache


# 项目根目录: src/prompt_loader.py 的上一级
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROMPT_DIR = os.path.join(_PROJECT_ROOT, "prompt")


class PromptNotFoundError(FileNotFoundError):
    """提示词模板文件不存在时抛出。"""


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """按名称加载 prompt/{name}.txt 模板原文。

    Args:
        name: 模板文件名 (不含 .txt 后缀).

    Returns:
        原始模板字符串.

    Raises:
        PromptNotFoundError: 文件不存在时抛出.
    """
    path = os.path.join(_PROMPT_DIR, f"{name}.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as exc:
        raise PromptNotFoundError(
            f"Prompt template not found: {path}"
        ) from exc


def render_prompt(name: str, **variables: object) -> str:
    """加载模板并替换 {变量名} 占位符。

    Args:
        name: 模板名 (不含 .txt).
        **variables: str.format() 所需的变量键值对.

    Returns:
        渲染后的 prompt 字符串.
    """
    template = load_prompt(name)
    return template.format(**variables)


def clear_cache() -> None:
    """清空模板缓存 (用于热更新或测试场景)。"""
    load_prompt.cache_clear()
