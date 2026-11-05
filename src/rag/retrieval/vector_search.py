from pymilvus import MilvusClient


async def vector_search(
    milvus: MilvusClient,
    collection_name: str,
    embedding: list[float],
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    filter_expr = _build_filter(filters) if filters else ""
    results = milvus.search(
        collection_name=collection_name,
        data=[embedding],
        limit=top_k,
        output_fields=["text", "doc_name", "doc_type", "category", "chunk_index"],
        filter=filter_expr,
        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
    )
    hits = []
    for hit in results[0]:
        hits.append({
            "text": hit["entity"]["text"],
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