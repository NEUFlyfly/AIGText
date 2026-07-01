#!/usr/bin/env bash
# =============================================================
# AIGText — 启动视觉后端 (Host B)
#
# 用法:
#   终端1 (文本后端): bash scripts/start_lang_server.sh
#   终端2 (视觉后端): bash scripts/run_vision_backend.sh
#
# 部署到其他主机时，仅需修改端口号，然后在 Host A 侧
# 通过 --vision-url 指定远程地址即可恢复通信。
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd -P)"
VISION_ROOT="$REPO_ROOT/vision_backend"

# --- Python 解释器 ---
PYTHON="D:/anaconda_envs/pytorch_env/python.exe"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python"
fi

# --- 默认配置 ---
VISION_HOST="0.0.0.0"
VISION_PORT=9101
TAXONOMY_PATH="$VISION_ROOT/data/iot_knowledge/iot_taxonomy.json"
REFERENCE_DIR="$VISION_ROOT/data/iot_knowledge/reference_images"

# --- 解析参数 ---
NEXT_IS_PORT=0
NEXT_IS_HOST=0
NEXT_IS_TAXONOMY=0
NEXT_IS_REF_DIR=0

for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        echo "AIGText 视觉后端启动脚本"
        echo "用法: bash scripts/run_vision_backend.sh [选项]"
        echo ""
        echo "  --port PORT        监听端口 (默认: 9101)"
        echo "  --host HOST        绑定地址 (默认: 0.0.0.0)"
        echo "  --taxonomy PATH    分类体系 JSON 路径"
        echo "  --reference-dir DIR  参考图片目录"
        echo "  --help, -h         显示此帮助"
        exit 0
    elif [ "$arg" = "--port" ]; then
        NEXT_IS_PORT=1
    elif [ "$NEXT_IS_PORT" = "1" ]; then
        VISION_PORT="$arg"
        NEXT_IS_PORT=0
    elif [ "$arg" = "--host" ]; then
        NEXT_IS_HOST=1
    elif [ "$NEXT_IS_HOST" = "1" ]; then
        VISION_HOST="$arg"
        NEXT_IS_HOST=0
    elif [ "$arg" = "--taxonomy" ]; then
        NEXT_IS_TAXONOMY=1
    elif [ "$NEXT_IS_TAXONOMY" = "1" ]; then
        TAXONOMY_PATH="$arg"
        NEXT_IS_TAXONOMY=0
    elif [ "$arg" = "--reference-dir" ]; then
        NEXT_IS_REF_DIR=1
    elif [ "$NEXT_IS_REF_DIR" = "1" ]; then
        REFERENCE_DIR="$arg"
        NEXT_IS_REF_DIR=0
    fi
done

# --- 检查必要文件 ---
if [ ! -d "$VISION_ROOT" ]; then
    echo "[ERROR] vision_backend 目录不存在: $VISION_ROOT"
    exit 1
fi

if [ ! -f "$TAXONOMY_PATH" ]; then
    echo "[ERROR] 分类体系文件不存在: $TAXONOMY_PATH"
    echo "[INFO] 请确认 vision_backend/data/iot_knowledge/iot_taxonomy.json 存在"
    exit 1
fi

# --- 启动信息 ---
echo "========================================"
echo "  AIGText 视觉后端 (Host B)"
echo "========================================"
echo "  监听地址 : $VISION_HOST:$VISION_PORT"
echo "  分类体系 : $TAXONOMY_PATH"
echo "  参考图片 : $REFERENCE_DIR"
echo ""
echo "  Host A 连接方式:"
echo "    bash scripts/start_lang_server.sh --vision-url http://127.0.0.1:$VISION_PORT"
echo ""
echo "  健康检查: curl http://127.0.0.1:$VISION_PORT/health"
echo "========================================"
echo ""

# --- 切换到 vision_backend 目录，设置路径 ---
cd "$VISION_ROOT"
export PYTHONPATH="$VISION_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- 启动服务器 ---
exec "$PYTHON" -m src.vision.vision_server \
    --host "$VISION_HOST" \
    --port "$VISION_PORT" \
    --taxonomy "$TAXONOMY_PATH" \
    --reference_dir "$REFERENCE_DIR"
