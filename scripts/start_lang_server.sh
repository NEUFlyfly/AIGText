# 启动电脑A 服务（语言模型 + RAG）
# 前提: llama-server 已在 18080 端口运行
# 用法: bash scripts/start_lang_server.sh
#
# 1. 检查 data/vectorstore/ 是否有索引，没有则自动建库
# 2. 设置 PYTHONPATH，启动 lang_server (端口 18082)
#    python -m src.lang.lang_server
