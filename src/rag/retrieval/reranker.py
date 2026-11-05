import dashscope
from loguru import logger
from src.core.config import get_settings

settings = get_settings()


async def rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """使用 DashScope qwen3-rerank 模型重排序"""
    if not results:
        return []

    documents = [r["text"] for r in results]
    try:
        response = dashscope.TextReRank.call(
            api_key=settings.DASHSCOPE_API_KEY,
            model="qwen3-rerank",
            query=query,
            documents=documents,
            top_n=top_k,
            return_documents=False,
        )
        if response.status_code != 200:
            logger.warning(f"Rerank API 失败: {response.message}")
            return results[:top_k]

        reranked = []
        for item in response.output.results:
            result = results[item.index].copy()
            result["rerank_score"] = item.relevance_score
            reranked.append(result)
        return reranked

    except Exception as e:
        logger.warning(f"Rerank 异常: {e}")
        return results[:top_k]