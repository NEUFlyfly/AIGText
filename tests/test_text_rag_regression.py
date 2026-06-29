from src.rag.pipeline import RAGPipeline
from src.rag.retriever import MetadataFilter, Retriever, SearchResult


Chunk = dict[str, str | int | float]


class FakeTextEmbedder:
    def embed_query(self, query: str) -> list[float]:
        assert query == "设备用途"
        return [1.0, 0.0, 0.0]


class FakeTextStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = [
            {
                "doc_id": "legacy_router_note",
                "coarse_category": "legacy",
                "sub_category": "legacy",
                "asset_type": "text",
                "source": "router.md",
                "chunk_id": 0,
                "text": "路由器用于连接局域网设备并提供网络访问。",
            },
            {
                "doc_id": "fixture_temp_sensor",
                "coarse_category": "智能传感器",
                "sub_category": "温湿度传感器",
                "asset_type": "text",
                "source": "tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/document.md",
                "chunk_id": 0,
                "text": "温湿度传感器用于采集室内温湿度。",
            },
        ]
        self.seen_filters: list[MetadataFilter | None] = []

    @property
    def count(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        assert query_vector == [1.0, 0.0, 0.0]
        self.seen_filters.append(where)
        return [_search_result(chunk) for chunk in self.chunks[:top_k]]


def test_text_only_retrieval_still_supported() -> None:
    store = FakeTextStore()
    pipeline = _text_only_pipeline(store)

    chunks = pipeline.retrieve("设备用途")

    assert chunks
    assert len(chunks) == 2
    assert store.seen_filters == [None]
    assert {chunk["doc_id"] for chunk in chunks} == {
        "legacy_router_note",
        "fixture_temp_sensor",
    }
    assert all("visual_candidates" not in chunk for chunk in chunks)


def _search_result(chunk: Chunk) -> SearchResult:
    return {
        "text": str(chunk["text"]),
        "source": str(chunk["source"]),
        "doc_id": str(chunk["doc_id"]),
        "coarse_category": str(chunk["coarse_category"]),
        "sub_category": str(chunk["sub_category"]),
        "asset_type": str(chunk["asset_type"]),
        "chunk_id": int(chunk["chunk_id"]),
        "score": 0.95,
    }


def _text_only_pipeline(store: FakeTextStore) -> RAGPipeline:
    pipeline = object.__new__(RAGPipeline)
    setattr(pipeline, "_store", store)
    setattr(pipeline, "_top_k", 5)
    setattr(pipeline, "_min_score", 0.0)
    setattr(pipeline, "_embedder", FakeTextEmbedder())
    setattr(pipeline, "_retriever", Retriever(FakeTextEmbedder(), store))
    return pipeline
