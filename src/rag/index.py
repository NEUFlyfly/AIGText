"""
RAG 离线建库脚本

职责:
  - 扫描 data/documents/ 下的所有 .txt / .md 文件
  - 加载 → 切分 → embedding → 写入 ChromaDB

用法:
  python -m src.rag.index
"""

import os
import sys
import time

from .loader import load_documents
from .splitter import split_text
from .embedder import Embedder
from .store import VectorStore


def build_index(
    documents_dir: str = "data/documents",
    persist_dir: str = "data/vectorstore",
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> int:
    """构建/重建文档索引。

    Args:
        documents_dir: 文档目录
        persist_dir: 向量库持久化目录
        chunk_size: 每个 chunk 的字符数
        chunk_overlap: chunk 重叠字符数

    Returns:
        生成的 chunk 总数，失败返回 -1
    """
    sep = "=" * 60
    print(sep)
    print("  RAG 索引构建工具")
    print(sep)

    # 1. 加载文档
    print("\n[1/4] 加载文档...")
    docs = load_documents(documents_dir)
    if not docs:
        print(f"  警告: 在 '{documents_dir}/' 下未找到 .txt/.md 文件")
        return -1
    for d in docs:
        print(f"  - {d['source']} ({len(d['content'])} 字符)")
    print(f"  共加载 {len(docs)} 个文档")

    # 2. 切分
    print("\n[2/4] 切分文本...")
    all_chunks = []
    for doc in docs:
        chunks = split_text(doc["content"], doc["source"], chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  - {doc['source']}: {len(chunks)} chunks")
    print(f"  共生成 {len(all_chunks)} 个 chunk")

    if not all_chunks:
        print("  错误: 没有生成任何 chunk")
        return -1

    # 3. Embedding
    print("\n[3/4] 生成向量 (首次运行将下载模型约 100MB)...")
    t0 = time.time()

    try:
        embedder = Embedder()
    except ImportError as e:
        print(f"  错误: {e}")
        return -1

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed_texts(texts)
    elapsed = time.time() - t0
    print(f"  已生成 {len(embeddings)} 个向量 (耗时 {elapsed:.1f}s)")

    # 4. 写入
    print("\n[4/4] 写入向量库...")

    try:
        store = VectorStore(persist_dir=persist_dir)
    except ImportError as e:
        print(f"  错误: {e}")
        return -1

    store.clear()
    store.upsert(all_chunks, embeddings)
    print(f"  已写入 ChromaDB: {os.path.abspath(persist_dir)}/")

    print(f"\n{sep}")
    print(f"  索引构建完成！共 {len(all_chunks)} 个 chunk")
    print(f"  向量库路径: {os.path.abspath(persist_dir)}/")
    print(sep)

    return len(all_chunks)


def main():
    """命令行入口。"""
    # 确保从项目根目录运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)

    result = build_index()
    if result < 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
