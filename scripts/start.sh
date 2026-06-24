#!/bin/bash
# =============================================================
# AIGText — 完整启动脚本（LLM 后端 + Web 前端）
# 
# 启动流程:
#   1. 拉起 llama-server (127.0.0.1:18080, 仅本地)
#   2. 等待模型加载完成
#   3. 启动前端 Python 服务器 (0.0.0.0:8080, 局域网可访问)
#   4. Ctrl+C 停止全部服务
#
# 用法:
#   bash scripts/start.sh
#   bash scripts/start.sh --port 9090        # 自定义前端端口
# =============================================================

set -e

# Python 解释器绝对路径
PYTHON="D:/anaconda_envs/pytorch_env/python.exe"

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT="$(pwd)"

# 配置
MODEL_PATH="$PROJECT_ROOT/models/lang/Dolphin3.0-Llama3.1-8B-Q6_K.gguf"
LLAMA_PORT=18080
FRONTEND_PORT=8080
LLAMA_BIN_DIR="$PROJECT_ROOT/bin"
SERVER_EXE="$LLAMA_BIN_DIR/llama-server.exe"
LOG_DIR="$PROJECT_ROOT/logs"
SERVER_PID=""

# 尝试查找正确的模型文件
if [ ! -f "$MODEL_PATH" ]; then
    FOUND=$(ls "$PROJECT_ROOT/models/lang/"*.gguf 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        MODEL_PATH="$FOUND"
    fi
fi

# 解析 --port 参数
for arg in "$@"; do
    case "$arg" in
        --port)
            shift_and_set=1
            ;;
        *)
            if [ "${shift_and_set:-0}" = "1" ]; then
                FRONTEND_PORT=$arg
                shift_and_set=0
            fi
            ;;
    esac
done

# -------------------------------------------------------------
# 清理函数
# -------------------------------------------------------------
cleanup() {
    echo ""
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[INFO] 关闭 llama-server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    echo "[INFO] 所有服务已停止。"
}
trap cleanup EXIT INT TERM

# -------------------------------------------------------------
# 检查 server 是否已在运行
# -------------------------------------------------------------
check_server() {
    curl -s "http://127.0.0.1:${LLAMA_PORT}/health" 2>/dev/null | grep -q '"status":"ok"' 2>/dev/null
}

# -------------------------------------------------------------
# 启动 llama-server (仅绑定 127.0.0.1)
# -------------------------------------------------------------
start_llama() {
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/server.log"

    echo "========================================"
    echo "  AIGText — 启动全部服务"
    echo "========================================"
    echo "  语言模型: $(basename "$MODEL_PATH")"
    echo "  LLM 端口: $LLAMA_PORT (仅本地)"
    echo "  前端端口: $FRONTEND_PORT (局域网可访问)"
    echo "----------------------------------------"

    export PATH="$LLAMA_BIN_DIR:$PATH"

    echo "  [1/3] 启动 llama-server..."
    "$SERVER_EXE" \
        -m "$MODEL_PATH" \
        -ngl -1 \
        -c 4096 \
        --port "$LLAMA_PORT" \
        --host 127.0.0.1 \
        > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!

    echo "  [2/3] 等待模型加载 (PID: $SERVER_PID)..."
    for i in $(seq 1 120); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo ""
            echo "[ERROR] llama-server 启动失败！最后日志:"
            echo "----------------------------------------"
            tail -20 "$LOG_FILE"
            exit 1
        fi
        if check_server; then
            echo "  [3/3] llama-server 已就绪 ✓"
            return 0
        fi
        sleep 1
    done

    echo ""
    echo "[ERROR] 等待超时 (120s)，server 未就绪"
    tail -20 "$LOG_FILE"
    exit 1
}

# -------------------------------------------------------------
# 启动前端服务器 (绑定 0.0.0.0，对外开放)
# -------------------------------------------------------------
start_frontend() {
    echo ""
    echo "----------------------------------------"
    echo "  启动前端服务器..."
    echo "----------------------------------------"
    echo ""

    export PYTHONPATH="$PROJECT_ROOT"
    "$PYTHON" -m src.front_server \
        --port "$FRONTEND_PORT" \
        --host 0.0.0.0 \
        --backend "http://127.0.0.1:${LLAMA_PORT}" \
        --static "$PROJECT_ROOT/frontend"
}

# -------------------------------------------------------------
# 主流程
# -------------------------------------------------------------

if check_server; then
    echo "========================================"
    echo "  AIGText — llama-server 已在运行"
    echo "========================================"
    echo "  端口: $LLAMA_PORT (复用已有)"
    echo "----------------------------------------"
else
    start_llama
fi

start_frontend
