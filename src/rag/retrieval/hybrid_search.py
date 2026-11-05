"""
Dense (向量) + Sparse (BM25) 混合检索，RRF 融合。
需要集合中有 sparse_embedding 字段 (Milvus 2.6 内置 BM25)。
"""
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


async def hybrid_search(
    milvus: MilvusClient,
    collection_name: str,
    dense_embedding: list[float],
    query_text: str,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    filter_expr = _build_filter(filters) if filters else ""

    dense_req = AnnSearchRequest(
        data=[dense_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=filter_expr,
    )
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=top_k,
        expr=filter_expr,
    )

    results = milvus.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=["text", "doc_name", "doc_type", "category", "chunk_index", "parent_text"],
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "text": hit["entity"]["text"],
            "parent_text": hit["entity"].get("parent_text", ""),
            "doc_name": hit["entity"]["doc_name"],
            "doc_type": hit["entity"]["doc_type"],
            "score": hit["distance"],
            "chunk_index": hit["entity"]["chunk_index"],
        })
    return hits


def _build_filter(filters: dict) -> str:
    parts = []
    for key, value in filters.items():
        if isinstance(value, list):
            values_str = ", ".join(f'"{v}"' for v in value)
            parts.append(f'{key} in [{values_str}]')
        else:
            parts.append(f'{key} == "{value}"')
    return " and ".join(parts)