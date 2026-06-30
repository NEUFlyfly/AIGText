#!/usr/bin/env bash
# =============================================================
# AIGText — 模拟视觉后端 + RAG 物联网知识库启动脚本
#
# 启动流程:
#   1. 如果有 llama-server + GGUF 模型 → 启动 LLM (可选)
#   2. 从 iot_taxonomy.json 生成 document.md + flat taxonomy.json
#   3. 重建向量索引（RAG）
#   4. 启动模拟视觉后端 (port 9101)
#   5. 启动前端服务器 (port 8080)，指向模拟视觉后端
#   6. Ctrl+C 停止全部服务
#
# 用法:
#   bash scripts/start_mock_vision.sh
#   bash scripts/start_mock_vision.sh --port 9090
#   bash scripts/start_mock_vision.sh --no-index   (跳过索引重建，使用已有索引)
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
MOCK_VISION_PORT=9101
FRONTEND_PORT=8080
LOG_DIR="$REPO_ROOT/logs"
BIN_DIR="$REPO_ROOT/bin"
SKIP_INDEX=false

# PIDs (processes started by THIS script instance)
LLAMA_PID=""
FRONTEND_PID=""
MOCK_VISION_PID=""

# --- Kill processes occupying our ports ---
# Uses netstat (Windows) or ss/lsof (Unix) via port number
kill_port() {
    local port=$1
    local pids=""

    # Try netstat (Windows / most shells)
    if command -v netstat &>/dev/null; then
        pids=$(netstat -ano 2>/dev/null | grep ":${port} " | grep "LISTENING" | awk '{print $NF}' | sort -u | grep -v '^0$' || true)
    fi
    # Try ss (Linux)
    if [ -z "$pids" ] && command -v ss &>/dev/null; then
        pids=$(ss -tlnp 2>/dev/null | grep ":${port} " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u || true)
    fi
    # Try lsof (macOS)
    if [ -z "$pids" ] && command -v lsof &>/dev/null; then
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
    fi

    if [ -n "$pids" ]; then
        for pid in $pids; do
            echo "[cleanup] 释放端口 $port 上的旧进程 (PID: $pid)"
            kill "$pid" 2>/dev/null || true
            if command -v taskkill &>/dev/null; then
                taskkill //F //PID "$pid" >/dev/null 2>&1 || true
            fi
        done
        sleep 1
    fi
}

kill_old_processes() {
    echo "[startup] 清理占用端口的旧进程..."
    kill_port "$FRONTEND_PORT"
    kill_port "$MOCK_VISION_PORT"
    # Only kill llama port if we plan to start our own
    # (If user has llama already running, the check_llama later will detect it)
    # kill_port "$LLAMA_PORT"  # Intentionally skipped — we'll reuse or start fresh
    echo "  done."
}

# --- Cleanup on exit / Ctrl+C / SIGTERM ---
cleanup() {
    echo ""
    echo "[INFO] 正在停止本次启动的服务..."
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "[INFO] 关闭前端服务器 (PID: $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null || true
        wait "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [ -n "${MOCK_VISION_PID:-}" ] && kill -0 "$MOCK_VISION_PID" 2>/dev/null; then
        echo "[INFO] 关闭模拟视觉后端 (PID: $MOCK_VISION_PID)..."
        kill "$MOCK_VISION_PID" 2>/dev/null || true
        wait "$MOCK_VISION_PID" 2>/dev/null || true
    fi
    # Only kill llama if THIS script started it
    if [ -n "${WE_STARTED_LLAMA:-}" ] && [ -n "${LLAMA_PID:-}" ] && kill -0 "$LLAMA_PID" 2>/dev/null; then
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
FRONTEND_PORT_ARG=""
NEXT_IS_PORT=0
REMAINING_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        echo "AIGText 模拟视觉后端启动脚本"
        echo "用法: bash scripts/start_mock_vision.sh [选项]"
        echo ""
        echo "  --port PORT       前端端口 (默认: 8080)"
        echo "  --no-index        跳过向量索引重建（使用已有索引）"
        echo "  --help, -h        显示此帮助"
        exit 0
    elif [ "$arg" = "--no-index" ]; then
        SKIP_INDEX=true
    elif [ "$arg" = "--port" ]; then
        NEXT_IS_PORT=1
    elif [ "$NEXT_IS_PORT" = "1" ]; then
        FRONTEND_PORT="$arg"
        NEXT_IS_PORT=0
    else
        REMAINING_ARGS+=("$arg")
    fi
done

echo "========================================"
echo "  AIGText — 模拟视觉后端模式"
echo "========================================"
echo ""

# =============================================================
# 0. 清理旧进程
# =============================================================
kill_old_processes

# =============================================================
# 1. 生成 IoT 文档
# =============================================================
echo "[docs] 生成 IoT 知识文档..."
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
"$PYTHON" src/rag/generate_iot_docs.py
echo ""

# =============================================================
# 2. 重建向量索引
# =============================================================
if [ "$SKIP_INDEX" = "true" ]; then
    echo "[index] 跳过索引重建 (--no-index)"
else
    echo "[index] 重建向量索引..."
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON" -m src.rag.index
    echo ""
fi

# =============================================================
# 3. 启动 LLM（如果资源就绪）
# =============================================================
HAS_LLM=false

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

if check_llama; then
    echo "[llama] 已在运行 (端口 $LLAMA_PORT)"
    HAS_LLM=true
elif [ -f "$LLAMA_BIN" ] && [ -n "$MODEL" ]; then
    echo "[llama] 启动 llama-server..."
    WE_STARTED_LLAMA=true
    "$LLAMA_BIN" \
        -m "$MODEL" \
        -ngl -1 \
        -c 4096 \
        --port "$LLAMA_PORT" \
        --host 127.0.0.1 \
        > "$LOG_DIR/llama.log" 2>&1 &
    LLAMA_PID=$!

    echo "[llama] 等待模型加载 (PID: $LLAMA_PID)..."
    LLAMA_READY=false
    for i in $(seq 1 180); do
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "[WARN] llama-server 进程意外退出"
            LLAMA_PID=""
            break
        fi
        if check_llama; then
            echo "[llama] 就绪 ✓"
            HAS_LLM=true
            LLAMA_READY=true
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            echo "  ... 等待中 (${i}s)"
        fi
        sleep 1
    done

    if [ "$LLAMA_READY" != "true" ]; then
        echo "[WARN] LLM 启动超时或失败"
        if kill -0 "$LLAMA_PID" 2>/dev/null; then
            kill "$LLAMA_PID" 2>/dev/null || true
            wait "$LLAMA_PID" 2>/dev/null || true
        fi
        LLAMA_PID=""
    fi
else
    echo "[llama] 跳过（LLM 资源不可用）"
fi

# =============================================================
# 4. 启动模拟视觉后端
# =============================================================
echo "[mock-vision] 启动模拟视觉后端 (端口 $MOCK_VISION_PORT)..."
"$PYTHON" -m src.mock_vision_server \
    --port "$MOCK_VISION_PORT" \
    --taxonomy "data/iot_knowledge/iot_taxonomy.json" \
    > "$LOG_DIR/mock_vision.log" 2>&1 &
MOCK_VISION_PID=$!

# 等待模拟视觉后端就绪
echo "[mock-vision] 等待就绪 (PID: $MOCK_VISION_PID)..."
VISION_READY=false
for i in $(seq 1 30); do
    if ! kill -0 "$MOCK_VISION_PID" 2>/dev/null; then
        echo "[ERROR] 模拟视觉后端意外退出，查看日志:"
        tail -20 "$LOG_DIR/mock_vision.log" 2>/dev/null || true
        MOCK_VISION_PID=""
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${MOCK_VISION_PORT}/health" >/dev/null 2>&1; then
        echo "[mock-vision] 就绪 ✓"
        VISION_READY=true
        break
    fi
    sleep 0.5
done

if [ "$VISION_READY" != "true" ]; then
    echo "[ERROR] 模拟视觉后端启动超时"
    exit 1
fi

# =============================================================
# 5. 启动前端服务器
# =============================================================
BACKEND_URL="http://127.0.0.1:${LLAMA_PORT}"

echo ""
echo "[frontend] 启动 (端口 $FRONTEND_PORT)..."
echo "  聊天页面: http://localhost:$FRONTEND_PORT/chat.html"
echo "  健康检查: http://localhost:$FRONTEND_PORT/api/health"
echo "  模拟视觉后端: http://127.0.0.1:$MOCK_VISION_PORT"
echo "  Ctrl+C 停止全部服务"
echo ""

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VISION_BACKEND_URL="http://127.0.0.1:${MOCK_VISION_PORT}"

"$PYTHON" "$REPO_ROOT/src/front_server.py" \
    --port "$FRONTEND_PORT" \
    --host 0.0.0.0 \
    --backend "$BACKEND_URL" \
    --static "$REPO_ROOT/frontend" &
FRONTEND_PID=$!

wait "$FRONTEND_PID" || true
