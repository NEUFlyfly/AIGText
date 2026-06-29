"""
RAG 离线建库脚本

职责:
  - 扫描 data/documents/ 下的所有 .txt / .md 文件
  - 加载 → 切分 → embedding → 写入 ChromaDB

用法:
  python -m src.rag.index
"""

import argparse
import hashlib
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

from .loader import load_documents
from .iot_loader import load_iot_documents
from .splitter import split_text
from .embedder import Embedder
from .store import VectorStore


Chunk: TypeAlias = dict[str, str | int | float]


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class TextStoreWriter(Protocol):
    def clear(self) -> None:
        ...

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...


@dataclass(frozen=True)
class TextIndexCliArgs:
    fixtures: bool
    documents_dir: str | None
    iot_documents_dir: str | None
    persist_dir: str


def build_index(
    documents_dir: str = "data/documents",
    iot_documents_dir: str = "data/iot_knowledge",
    persist_dir: str = "data/vectorstore",
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    fixture_mode: bool = False,
    store: TextStoreWriter | None = None,
    embedder: TextEmbedder | None = None,
) -> int:
    """构建/重建文档索引。

    Args:
        documents_dir: 传统文档目录
        iot_documents_dir: IoT 知识目录
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
    docs.extend(load_iot_documents(iot_documents_dir))
    if not docs:
        print(f"  警告: 在 '{documents_dir}/' 或 '{iot_documents_dir}/' 下未找到 .txt/.md 文件")
        return -1
    for d in docs:
        print(f"  - {d['source']} ({len(d['content'])} 字符)")
    print(f"  共加载 {len(docs)} 个文档")

    # 2. 切分
    print("\n[2/4] 切分文本...")
    all_chunks: list[Chunk] = []
    for doc in docs:
        metadata = {
            "doc_id": doc["doc_id"],
            "coarse_category": doc["coarse_category"],
            "sub_category": doc["sub_category"],
            "asset_type": doc.get("asset_type", "text"),
        }
        chunks = split_text(
            doc["content"],
            doc["source"],
            chunk_size,
            chunk_overlap,
            metadata=metadata,
        )
        all_chunks.extend(_index_chunks(chunks))
        print(f"  - {doc['source']}: {len(chunks)} chunks")
    print(f"  共生成 {len(all_chunks)} 个 chunk")

    if not all_chunks:
        print("  错误: 没有生成任何 chunk")
        return -1

    # 3. Embedding
    if fixture_mode:
        print("\n[3/4] 生成离线 fixture 向量...")
    else:
        print("\n[3/4] 生成向量 (首次运行将下载模型约 100MB)...")
    t0 = time.time()

    try:
        active_embedder = embedder or (
            _DeterministicFixtureTextEmbedder() if fixture_mode else Embedder()
        )
    except ImportError as e:
        print(f"  错误: {e}")
        return -1

    texts = [c["text"] for c in all_chunks]
    embeddings = active_embedder.embed_texts([str(text) for text in texts])
    elapsed = time.time() - t0
    print(f"  已生成 {len(embeddings)} 个向量 (耗时 {elapsed:.1f}s)")

    # 4. 写入
    print("\n[4/4] 写入向量库...")

    try:
        active_store = store or (
            _InMemoryTextStore() if fixture_mode else VectorStore(persist_dir=persist_dir)
        )
    except ImportError as e:
        print(f"  错误: {e}")
        return -1

    active_store.clear()
    active_store.upsert(all_chunks, embeddings)
    if fixture_mode:
        print("  已写入离线 fixture 内存向量库")
    else:
        print(f"  已写入 ChromaDB: {os.path.abspath(persist_dir)}/")

    print(f"\n{sep}")
    print(f"  索引构建完成！共 {len(all_chunks)} 个 chunk")
    if fixture_mode:
        print("  fixture 模式未写入持久化向量库")
    else:
        print(f"  向量库路径: {os.path.abspath(persist_dir)}/")
    print(sep)

    return len(all_chunks)


class _DeterministicFixtureTextEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_stable_unit_vector(text) for text in texts]


class _InMemoryTextStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []

    def clear(self) -> None:
        self.chunks.clear()
        self.embeddings.clear()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunk and embedding counts must match")
        self.chunks = list(chunks)
        self.embeddings = list(embeddings)


def _stable_unit_vector(value: str) -> list[float]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    vector = [float(digest[index] + 1) for index in range(8)]
    norm = math.sqrt(sum(component * component for component in vector))
    return [component / norm for component in vector]


def _index_chunks(chunks: list[dict[str, str | int]]) -> list[Chunk]:
    return [dict(chunk) for chunk in chunks]


def _parse_args(argv: Sequence[str] | None = None) -> TextIndexCliArgs:
    parser = argparse.ArgumentParser(description="Build the text RAG vector index.")
    _ = parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Build deterministic fixture text chunks with fake offline embeddings.",
    )
    _ = parser.add_argument("--documents-dir", default=None, help="Legacy text documents directory.")
    _ = parser.add_argument("--iot-dir", default=None, help="IoT knowledge directory.")
    _ = parser.add_argument("--persist-dir", default="data/vectorstore")
    raw_args = cast(dict[str, object], vars(parser.parse_args(argv)))
    documents_dir = raw_args.get("documents_dir")
    iot_documents_dir = raw_args.get("iot_dir")
    persist_dir = raw_args.get("persist_dir")
    return TextIndexCliArgs(
        fixtures=raw_args.get("fixtures") is True,
        documents_dir=documents_dir if isinstance(documents_dir, str) else None,
        iot_documents_dir=iot_documents_dir if isinstance(iot_documents_dir, str) else None,
        persist_dir=persist_dir if isinstance(persist_dir, str) else "data/vectorstore",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """命令行入口。"""
    # 确保从项目根目录运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)

    args = _parse_args(argv)
    documents_dir = args.documents_dir or (
        "tests/fixtures/documents" if args.fixtures else "data/documents"
    )
    iot_documents_dir = args.iot_documents_dir or (
        "tests/fixtures/iot_knowledge" if args.fixtures else "data/iot_knowledge"
    )

    result = build_index(
        documents_dir=documents_dir,
        iot_documents_dir=iot_documents_dir,
        persist_dir=args.persist_dir,
        fixture_mode=args.fixtures,
    )
    if result < 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
