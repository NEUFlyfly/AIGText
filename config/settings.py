# 全局配置文件
# 职责:
#   - 集中管理所有服务的端口、URL、模型路径
#   - 电脑A 和电脑B 部署在不同机器上时，各自修改对应配置
#
# 配置项:
#   LANG_SERVER_HOST / PORT     → 电脑A 语言服务监听地址
#   VISION_SERVER_HOST / PORT   → 电脑B 视觉服务监听地址
#   LLAMA_SERVER_URL            → llama-server 地址 (电脑A 本地)
#   LLAMA_TEMPERATURE           → LLM 温度参数
#   LLAMA_MAX_TOKENS            → LLM 最大输出 token
#   EMBEDDING_MODEL_NAME        → embedding 模型名称/路径
#   MODELS_LANG_DIR             → 语言模型权重目录 (models/lang/)
#   MODELS_VISION_DIR           → 视觉模型权重目录 (models/vision/)
#   VISION_MODEL_PATH           → 视觉模型文件路径
#   VISION_MODEL_LABELS         → 类别标签映射文件路径
#   CHROMA_PERSIST_DIR          → ChromaDB 持久化目录
#   CHUNK_SIZE / CHUNK_OVERLAP  → 文档切分参数
#   RETRIEVAL_TOP_K             → 检索返回数量
#   DOCUMENTS_DIR               → 知识库源文档目录
