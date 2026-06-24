#!/bin/bash
# =============================================================
# AIGText — CLI 聊天启动器
# 自动拉起 llama-server (GPU 加速) + Python 聊天客户端
# 用法:
#   bash scripts/chat.sh                  # 默认配置
#   bash scripts/chat.sh --port 9090      # 自定义端口
#   bash scripts/chat.sh --system "你是..." # 自定义提示词
# =============================================================

# Python 解释器绝对路径
PYTHON="D:/anaconda_envs/pytorch_env/python.exe"

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT="$(pwd)"

# 配置
MODEL_PATH="$PROJECT_ROOT/models/lang/Dolphin3.1-8B-Q6_K.gguf"
DEFAULT_PORT=18080
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
PORT=$DEFAULT_PORT
for arg in "$@"; do
    case "$arg" in
        --port) shift_and_set_port=1 ;;
        *)
            if [ "${shift_and_set_port:-0}" = "1" ]; then
                PORT=$arg
                shift_and_set_port=0
            fi
            ;;
    esac
done

# 清理函数
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "[INFO] 关闭 llama-server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# 检查 server 是否已在运行（模型加载完成）
check_server() {
    local status
    status=$(curl -s "http://127.0.0.1:${PORT}/health" 2>/dev/null) || return 1
    echo "$status" | grep -q '"status":"ok"' 2>/dev/null
}

# 启动 server
start_server() {
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/server.log"

    echo "========================================"
    echo "  AIGText — 启动聊天系统"
    echo "========================================"
    echo "  模型: $(basename "$MODEL_PATH")"
    echo "  端口: $PORT"
    echo "  GPU:  全部层级 (RTX 4060 Laptop)"
    echo "----------------------------------------"

    # 添加 llama-bin 到 PATH 以便找到 CUDA DLL
    export PATH="$LLAMA_BIN_DIR:$PATH"

    echo "  [1/3] 启动 llama-server..."
    "$SERVER_EXE" \
        -m "$MODEL_PATH" \
        -ngl -1 \
        -c 4096 \
        --port "$PORT" \
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
            echo "  [3/3] Server 已就绪 ✓"
            return 0
        fi
        sleep 1
    done

    echo ""
    echo "[ERROR] 等待超时 (120s)，server 未就绪"
    echo "最后日志:"
    tail -20 "$LOG_FILE"
    exit 1
}

# 检查 server 是否已在运行
if check_server; then
    echo "========================================"
    echo "  AIGText — 使用已有 server"
    echo "========================================"
    echo "  端口: $PORT (已运行)"
    echo "----------------------------------------"
else
    start_server
fi

echo ""
echo "  正在启动聊天客户端..."
echo "========================================"
echo ""

# 启动 Python 聊天客户端（使用 -m 模块方式运行，支持包内相对导入）
export PYTHONPATH="$PROJECT_ROOT"
"$PYTHON" -m src.lang.chat --port "$PORT" "$@"
