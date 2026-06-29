from src.rag.pipeline import RAGPipeline
from src.rag.retriever import MetadataFilter, Retriever, SearchResult, build_metadata_filter


Chunk = dict[str, str | int | float]


class StubEmbedder:
    def embed_query(self, query: str) -> list[float]:
        _ = query
        return [1.0, 0.0, 0.0]


class MemoryStore:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

    @property
    def count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        where: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        _ = query_vector
        matched = [chunk for chunk in self._chunks if _matches_where(chunk, where)]
        return [
            {
                "text": str(chunk["text"]),
                "source": str(chunk["source"]),
                "doc_id": str(chunk["doc_id"]),
                "coarse_category": str(chunk["coarse_category"]),
                "sub_category": str(chunk["sub_category"]),
                "asset_type": str(chunk["asset_type"]),
                "chunk_id": int(chunk["chunk_id"]),
                "score": 1.0,
            }
            for chunk in matched[:top_k]
        ]


def _matches_where(chunk: Chunk, where: MetadataFilter | None) -> bool:
    if where is None:
        return True

    and_clauses = where.get("$and")
    if isinstance(and_clauses, list):
        return all(
            isinstance(clause, dict) and _matches_where(chunk, clause)
            for clause in and_clauses
        )

    for key, expected in where.items():
        if key == "$and":
            continue
        actual = chunk.get(key)
        if isinstance(expected, dict):
            allowed = expected.get("$in")
            if isinstance(allowed, list) and actual not in allowed:
                return False
        elif actual != expected:
            return False
    return True


def _seed_store() -> MemoryStore:
    chunks: list[Chunk] = [
        {
            "doc_id": "fixture_temp_sensor",
            "coarse_category": "智能传感器",
            "sub_category": "温湿度传感器",
            "asset_type": "text",
            "source": "tests/fixtures/iot_knowledge/智能传感器/温湿度传感器/document.md",
            "chunk_id": 0,
            "text": "温湿度传感器用于采集温度和湿度。",
        },
        {
            "doc_id": "fixture_pir_sensor",
            "coarse_category": "智能传感器",
            "sub_category": "人体红外传感器",
            "asset_type": "text",
            "source": "tests/fixtures/iot_knowledge/智能传感器/人体红外传感器/document.md",
            "chunk_id": 0,
            "text": "人体红外传感器用于检测人员移动。",
        },
        {
            "doc_id": "fixture_ptz_camera",
            "coarse_category": "智能摄像头",
            "sub_category": "室内云台摄像头",
            "asset_type": "text",
            "source": "tests/fixtures/iot_knowledge/智能摄像头/室内云台摄像头/document.md",
            "chunk_id": 0,
            "text": "室内云台摄像头支持远程查看和巡航。",
        },
    ]
    return MemoryStore(chunks)


def test_build_metadata_filter_combines_supported_fields() -> None:
    metadata_filter = build_metadata_filter(
        doc_ids=["fixture_temp_sensor"],
        coarse_category="智能传感器",
        sub_category="温湿度传感器",
    )

    assert metadata_filter == {
        "$and": [
            {"doc_id": {"$in": ["fixture_temp_sensor"]}},
            {"coarse_category": "智能传感器"},
            {"sub_category": "温湿度传感器"},
        ]
    }


def test_retrieve_filters_by_doc_ids() -> None:
    store = _seed_store()
    retriever = Retriever(StubEmbedder(), store)

    results = retriever.retrieve(
        "温湿度",
        top_k=5,
        min_score=0.0,
        doc_ids=["fixture_temp_sensor"],
    )

    assert results
    assert {result["doc_id"] for result in results} == {"fixture_temp_sensor"}
    assert all(result["chunk_id"] == 0 for result in results)


def test_retrieve_filters_by_category_fields() -> None:
    store = _seed_store()
    retriever = Retriever(StubEmbedder(), store)

    results = retriever.retrieve(
        "传感器",
        top_k=5,
        min_score=0.0,
        coarse_category="智能传感器",
    )

    assert results
    assert {result["doc_id"] for result in results} == {
        "fixture_temp_sensor",
        "fixture_pir_sensor",
    }

    sub_category_results = retriever.retrieve(
        "人体检测",
        top_k=5,
        min_score=0.0,
        sub_category="人体红外传感器",
    )
    assert {result["doc_id"] for result in sub_category_results} == {"fixture_pir_sensor"}


def test_retrieve_without_filter_preserves_text_only_behavior() -> None:
    store = _seed_store()
    retriever = Retriever(StubEmbedder(), store)

    results = retriever.retrieve("设备", top_k=3, min_score=0.0)

    assert len(results) == 3
    assert {result["doc_id"] for result in results} == {
        "fixture_temp_sensor",
        "fixture_pir_sensor",
        "fixture_ptz_camera",
    }


def test_pipeline_retrieve_accepts_optional_filters() -> None:
    store = _seed_store()
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._store = store
    pipeline._top_k = 5
    pipeline._min_score = 0.0
    pipeline._embedder = StubEmbedder()
    pipeline._retriever = Retriever(StubEmbedder(), store)

    results = pipeline.retrieve("温湿度", doc_ids=["fixture_temp_sensor"])

    assert results
    assert {result["doc_id"] for result in results} == {"fixture_temp_sensor"}
