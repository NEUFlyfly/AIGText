# 语言模型 HTTP 服务 (电脑A)
# 职责:
#   - 接收前端传来的分类结果(物联网设备类别名)
#   - 调用 RAG Pipeline 检索知识库中该设备的介绍资料
#   - 调用本地 LLM (llama-server) 流式生成设备介绍
#   - 以 SSE (Server-Sent Events) 格式逐 token 推送回前端
#
# 端点:
#   GET  /api/lang/ask?category=温湿度传感器
#        → SSE 流: data: {"token": "这是..."} ... data: [DONE]
#   GET  /health
#        → {"status": "ok"}
#
# 依赖:
#   - src.lang.model_client  (LlamaCppChatClient, 连接 llama-server:18080)
#   - src.rag.pipeline       (RAG 检索 + prompt 构造)
#   - config.settings        (全局配置)
