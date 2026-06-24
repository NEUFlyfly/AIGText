# 建立 RAG 索引
# 用法: bash scripts/index_docs.sh
#
# 扫描 data/documents/ → 切分 → embedding → 写入 ChromaDB
# 内部调用: python -m src.rag.index
