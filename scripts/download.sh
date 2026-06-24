#!/bin/bash
# =============================================================
# AIGText — 模型下载启动脚本
# 用法:
#   bash scripts/download.sh              # 使用默认配置
#   bash scripts/download.sh --mirror URL # 自定义镜像站
# =============================================================

# Python 解释器绝对路径
PYTHON="D:/anaconda_envs/pytorch_env/python.exe"

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 运行下载脚本
echo "[1/1] 开始下载模型..."
"$PYTHON" src/download_model.py "$@"

exit_code=$?
echo ""
if [ $exit_code -eq 0 ]; then
    echo "下载完成！模型文件位于: models/"
else
    echo "下载失败 (退出码: $exit_code)，请检查错误信息"
    exit $exit_code
fi
