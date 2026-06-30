#!/usr/bin/env python3
"""
AIGText — CLI 交互聊天客户端
通过 HTTP API 连接 llama-server (OpenAI 兼容接口)

用法:
  python -m src.lang.chat                          # 默认连接 localhost:18080
  python -m src.lang.chat --port 8080              # 指定端口
  python -m src.lang.chat --url http://x:8080      # 指定完整 URL
  python -m src.lang.chat --system "你是一个助手"   # 自定义系统提示
"""

import argparse
import sys
import urllib.error
import platform

from .model_client import LlamaCppChatClient

# 非阻塞键盘检测
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import msvcrt as _msvcrt
else:
    import select as _select
    import termios as _termios
    import tty as _tty


# RAG 管线（延迟加载）
_rag_pipeline = None


def _get_rag_pipeline():
    """延迟初始化 RAG 管线。"""
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            from ..rag.pipeline import RAGPipeline

            _rag_pipeline = RAGPipeline()
        except Exception as e:
            print(f"[警告] RAG 初始化失败: {e}", file=sys.stderr)
            _rag_pipeline = False
    return _rag_pipeline if _rag_pipeline is not False else None


def kbhit() -> bool:
    """检测是否有按键按下（非阻塞）"""
    if IS_WINDOWS:
        return _msvcrt.kbhit()
    else:
        fd = sys.stdin.fileno()
        old_settings = _termios.tcgetattr(fd)
        try:
            _tty.setcbreak(fd)
            r, _, _ = _select.select([sys.stdin], [], [], 0)
            return bool(r)
        finally:
            _termios.tcsetattr(fd, _termios.TCSADRAIN, old_settings)


def getch() -> bytes:
    """读取按键字符"""
    if IS_WINDOWS:
        return _msvcrt.getch()
    else:
        return sys.stdin.read(1).encode()


def safe_input(prompt: str) -> str:
    """跨平台安全输入，解决 git-bash + readline 的中文标点问题"""
    if platform.system() == "Windows":
        try:
            import readline
            readline.set_history_length(0)
        except ImportError:
            pass
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return sys.stdin.readline().strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AIGText — CLI 聊天客户端 (连接 llama-server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
对话命令:
  /exit     退出
  /clear    清空上下文
  /help     显示帮助
  /rag on   开启 RAG 检索增强
  /rag off  关闭 RAG 检索增强
  /reindex  重建文档索引

示例:
  python src/chat.py
  python src/chat.py --port 9090 --system "你是 Python 专家"
""",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server 主机 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18080,
                        help="Server 端口 (默认: 18080)")
    parser.add_argument("--url", default=None,
                        help="完整 Server URL (覆盖 host/port)")
    parser.add_argument("--system",
                        default=None,
                        help="系统提示词 (默认从 prompt/system_prompt.txt 加载)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度 (默认: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="最大输出 token (默认: 2048)")
    return parser


def print_banner(server_url: str, rag_enabled: bool, chunk_count: int = 0):
    print("=" * 60)
    print("  AIGText — CLI 交互聊天")
    print("=" * 60)
    print(f"  Server: {server_url}")
    if rag_enabled and chunk_count > 0:
        print(f"  RAG: 已开启 ({chunk_count} chunks 已索引)")
    print("  命令: /exit 退出 | /new 新对话 | /help 帮助")
    print("  RAG: /rag on 开启 | /rag off 关闭 | /reindex 重建索引")
    print("  按 ESC 可中断 AI 输出")
    print("=" * 60)
    print()


def main():
    args = build_parser().parse_args()

    if args.url:
        server_url = args.url.rstrip("/")
    else:
        server_url = f"http://{args.host}:{args.port}"

    # Health check
    client = LlamaCppChatClient(server_url, args.temperature, args.max_tokens)

    if not client.health_check():
        print(f"[错误] 无法连接到 server: {server_url}", file=sys.stderr)
        print("请确保 llama-server 正在运行。启动方式:", file=sys.stderr)
        print("  bash scripts/chat.sh", file=sys.stderr)
        sys.exit(1)

    # RAG 状态
    rag_enabled = False
    pipeline = _get_rag_pipeline()
    if pipeline and pipeline.is_ready:
        rag_enabled = True

    print_banner(server_url, rag_enabled, pipeline.chunk_count if pipeline else 0)

    # 从 prompt/system_prompt.txt 加载默认系统提示词 (CLI --system 未指定时)
    system_prompt = args.system
    if system_prompt is None:
        from ..prompt_loader import load_prompt

        system_prompt = load_prompt("system_prompt").strip()

    # 对话历史
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = safe_input("\nFlyfly: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        # ---- 命令处理 ----
        if user_input == "/exit":
            print("\n再见！")
            break

        elif user_input in ("/clear", "/new"):
            messages = [{"role": "system", "content": system_prompt}]
            print("[新对话已开始]")
            continue

        elif user_input == "/help":
            print("\n命令:")
            print("  /exit     退出")
            print("  /new      开始新对话")
            print("  /help     显示帮助")
            print("  /rag on   开启 RAG 检索增强")
            print("  /rag off  关闭 RAG 检索增强")
            print("  /reindex  重建文档索引")
            continue

        elif user_input == "/rag on":
            p = _get_rag_pipeline()
            if p is None:
                print("[RAG] 初始化失败，请确保已安装 chromadb 和 sentence-transformers")
                continue
            if not p.is_ready:
                print(f"[RAG] 向量库为空 (0 chunks)。请先运行: python -m src.rag.index")
                continue
            rag_enabled = True
            print(f"[RAG] 已开启 ({p.chunk_count} chunks 已索引)")
            continue

        elif user_input == "/rag off":
            rag_enabled = False
            print("[RAG] 已关闭")
            continue

        elif user_input == "/reindex":
            print("[RAG] 正在重建索引...")
            try:
                from ..rag.index import build_index
                result = build_index()
                if result >= 0:
                    # 重置 pipeline 以使用新索引
                    global _rag_pipeline
                    _rag_pipeline = None
                    p = _get_rag_pipeline()
                    if p:
                        rag_enabled = True
                        print(f"[RAG] 索引重建完成！({p.chunk_count} chunks)")
                else:
                    print("[RAG] 索引重建失败")
            except Exception as e:
                print(f"[RAG] 索引重建出错: {e}")
            continue

        # ---- RAG 检索（仅当轮增强，不污染历史）----
        augmented = None
        if rag_enabled:
            pipeline = _get_rag_pipeline()
            if pipeline and pipeline.is_ready:
                chunks = pipeline.retrieve(user_input)
                if chunks:
                    augmented = pipeline.augment(user_input, chunks)
                    print(f"[RAG] 检索到 {len(chunks)} 个相关片段")
                else:
                    print("[RAG] 未检索到相关文档，使用原始查询")

        # 历史只记录用户原始提问
        messages.append({"role": "user", "content": user_input})

        # 发给模型的 messages：RAG 开启时注入参考上下文到 user 消息
        # 不替换 system prompt，保持模型原有对话能力
        api_messages = messages.copy()
        if augmented:
            api_messages[-1] = {"role": "user", "content": augmented}

        print("\nAssistant: ", end="", flush=True)

        try:
            full_text = ""
            interrupted = False
            for chunk in client.stream_chat(api_messages):
                if kbhit() and getch() == b'\x1b':
                    interrupted = True
                    break
                print(chunk, end="", flush=True)
                full_text += chunk
            print()
            if interrupted:
                print("[输出已中断]")

            if full_text:
                messages.append({"role": "assistant", "content": full_text})
            else:
                messages.pop()

        except urllib.error.URLError as e:
            print(f"\n[错误] 连接失败: {e}", file=sys.stderr)
            messages.pop()
        except Exception as e:
            print(f"\n[错误] 请求失败: {e}", file=sys.stderr)
            messages.pop()


if __name__ == "__main__":
    main()
