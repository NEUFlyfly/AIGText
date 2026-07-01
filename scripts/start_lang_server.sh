#!/usr/bin/env bash
# =============================================================
# AIGText — 一键启动服务
#
# 启动流程:
#   1. 如果有 llama-server + GGUF 模型 → 启动 LLM (可选)
#   2. 启动前端 Python 服务器 (0.0.0.0:8080)
#      提供: 静态页面 / 3D建模API / RAG / 聊天代理
#   3. Ctrl+C 停止全部服务
#
# 用法:
#   bash scripts/start_lang_server.sh
#   bash scripts/start_lang_server.sh --port 9090
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd -P)"

# --- 使用 conda 环境的 Python（如果存在）---
PYTHON="D:/anaconda_envs/pytorch_env/python.exe"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python"
fi

LLAMA_PORT=18080
FRONTEND_PORT=8080
VISION_BACKEND_URL=""
LOG_DIR="$REPO_ROOT/logs"
BIN_DIR="$REPO_ROOT/bin"
LLAMA_PID=""
FRONTEND_PID=""

# --- 清理所有子进程 ---
cleanup() {
    echo ""
    echo "[INFO] 正在停止服务..."
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [ -n "${LLAMA_PID:-}" ] && kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "[INFO] 关闭 llama-server (PID: $LLAMA_PID)..."
        kill "$LLAMA_PID" 2>/dev/null || true
        wait "$LLAMA_PID" 2>/dev/null || true
    fi
    echo "[INFO] 所有服务已停止。"
}
trap cleanup EXIT INT TERM

check_llama() {
    curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1
}

# --- 解析参数 ---
next_is_port=0
next_is_vision_url=0
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        echo "AIGText 一键启动脚本"
        echo "用法: bash scripts/start_lang_server.sh [选项]"
        echo ""
        echo "  --port PORT          前端端口 (默认: 8080)"
        echo "  --vision-url URL     视觉后端地址 (默认: http://127.0.0.1:9091)"
        echo "  --help, -h           显示此帮助"
        exit 0
    fi
    if [ "$arg" = "--port" ]; then
        next_is_port=1
    elif [ "$next_is_port" = "1" ]; then
        FRONTEND_PORT="$arg"
        next_is_port=0
    elif [ "$arg" = "--vision-url" ]; then
        next_is_vision_url=1
    elif [ "$next_is_vision_url" = "1" ]; then
        VISION_BACKEND_URL="$arg"
        next_is_vision_url=0
    fi
done

# --- 查找 llama-server 和 GGUF 模型 ---
LLAMA_BIN="$BIN_DIR/llama-server.exe"
if [ ! -f "$LLAMA_BIN" ]; then
    LLAMA_BIN="$BIN_DIR/llama-server"
fi
MODEL=""
for candidate in "$REPO_ROOT/models/lang/"*.gguf; do
    if [ -f "$candidate" ]; then
        MODEL="$candidate"
        break
    fi
done

# =============================================================
# 1. 尝试启动 LLM（如果资源就绪）
# =============================================================
HAS_LLM=false

if check_llama; then
    echo "========================================"
    echo "  AIGText — llama-server 已在运行"
    echo "========================================"
    echo "  LLM 端口: $LLAMA_PORT (复用已有)"
    echo "  前端端口: $FRONTEND_PORT"
    echo "----------------------------------------"
    HAS_LLM=true
elif [ -f "$LLAMA_BIN" ] && [ -n "$MODEL" ]; then
    echo "========================================"
    echo "  AIGText — 启动全部服务"
    echo "========================================"
    echo "  LLM 模型: $(basename "$MODEL")"
    echo "  LLM 端口: $LLAMA_PORT (仅本地)"
    echo "  前端端口: $FRONTEND_PORT"
    echo "----------------------------------------"

    mkdir -p "$LOG_DIR"
    "$LLAMA_BIN" \
        -m "$MODEL" \
        -ngl -1 \
        -c 4096 \
        --port "$LLAMA_PORT" \
        --host 127.0.0.1 \
        > "$LOG_DIR/server.log" 2>&1 &
    LLAMA_PID=$!

    echo "[llama] 等待模型加载 (PID: $LLAMA_PID)..."
    LLAMA_READY=false
    for i in $(seq 1 180); do
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "[WARN] llama-server 进程意外退出"
            echo "[WARN] 最后日志:"
            tail -5 "$LOG_DIR/server.log" 2>/dev/null || true
            LLAMA_PID=""
            break
        fi
        if check_llama; then
            echo "[llama] 就绪 ✓"
            HAS_LLM=true
            LLAMA_READY=true
            break
        fi
        # 每 10 秒打印一次进度
        if [ $((i % 10)) -eq 0 ]; then
            echo "  ... 等待中 (${i}s)"
        fi
        sleep 1
    done

    if [ "$LLAMA_READY" != "true" ]; then
        echo "[WARN] LLM 启动超时或失败，继续启动前端（不影响 3D 建模）"
        if kill -0 "$LLAMA_PID" 2>/dev/null; then
            kill "$LLAMA_PID" 2>/dev/null || true
            wait "$LLAMA_PID" 2>/dev/null || true
        fi
        LLAMA_PID=""
    fi
else
    echo "========================================"
    echo "  AIGText — 启动前端服务"
    echo "========================================"
    if [ ! -f "$LLAMA_BIN" ]; then
        echo "  [跳过] llama-server 未安装 (不影响 3D 建模)"
    fi
    if [ -z "$MODEL" ]; then
        echo "  [跳过] GGUF 模型未找到 (不影响 3D 建模)"
    fi
    echo "  前端端口: $FRONTEND_PORT"
    echo "----------------------------------------"
fi

# =============================================================
# 2. 启动前端服务器（必定执行）
# =============================================================

# 确保输出目录存在
mkdir -p "$REPO_ROOT/data/models" "$REPO_ROOT/data/temp/models"

BACKEND_URL="http://127.0.0.1:${LLAMA_PORT}"
if [ "$HAS_LLM" != "true" ]; then
    BACKEND_URL="http://127.0.0.1:18080"
fi

echo ""
echo "[frontend] 启动 (端口 $FRONTEND_PORT)..."
if [ -n "$VISION_BACKEND_URL" ]; then
    echo "  视觉后端: $VISION_BACKEND_URL (来自 --vision-url)"
else
    VISION_BACKEND_URL="http://127.0.0.1:9091"
    echo "  视觉后端: $VISION_BACKEND_URL (默认，使用 --vision-url 修改)"
fi
echo "  聊天页面: http://localhost:$FRONTEND_PORT/chat.html"
echo "  Ctrl+C 停止全部服务"
echo ""

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VISION_BACKEND_URL

# 前台运行 Python 服务器（不用 exec，保持 bash 存活以处理信号）
"$PYTHON" "$REPO_ROOT/src/front_server.py" \
    --port "$FRONTEND_PORT" \
    --host 0.0.0.0 \
    --backend "$BACKEND_URL" \
    --static "$REPO_ROOT/frontend" &
FRONTEND_PID=$!

# 等待前端进程，同时转发信号
wait "$FRONTEND_PID" || true
