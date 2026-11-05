from dataclasses import dataclass
from pymilvus import MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from src.rag.config import RAGConfig
from src.rag.retrieval.vector_search import vector_search
from src.rag.retrieval.hybrid_search import hybrid_search
from src.rag.retrieval.reranker import rerank
from src.rag.retrieval.hyde import generate_hypothetical_doc
from src.rag.retrieval.query_transform import rewrite_query
from src.rag.generation.generator import generate_answer


@dataclass
class RAGDeps:
    milvus: MilvusClient
    embedding_model: DashScopeEmbeddings
    llm: ChatOpenAI
    config: RAGConfig


class RAGEngine:
    """
    RAG 引擎 - 基础设施层。
    Agent 通过 collection_name 指定查询哪个知识库。
    优化策略通过 RAGConfig 开关控制。
    """

    def __init__(self, deps: RAGDeps):
        self.deps = deps
        self.config = deps.config

    async def retrieve(
        self,
        query: str,
        collection_name: str | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        target = collection_name or self.config.collection_name
        cfg = self.config.retrieval

        processed_query = await rewrite_query(query, self.deps.llm)

        if cfg.use_hyde:
            hypo_doc = await generate_hypothetical_doc(processed_query, self.deps.llm)
            embedding = await self.deps.embedding_model.aembed_query(hypo_doc)
        else:
            embedding = await self.deps.embedding_model.aembed_query(processed_query)

        if cfg.use_hybrid:
            results = await hybrid_search(
                self.deps.milvus, target, embedding, processed_query, cfg.top_k, filters
            )
        else:
            results = await vector_search(
                self.deps.milvus, target, embedding, cfg.top_k, filters
            )

        if cfg.use_rerank and results:
            results = await rerank(processed_query, results, cfg.rerank_top_k)

        return results

    async def generate(self, query: str, contexts: list[dict]) -> str:
        return await generate_answer(
            query, contexts, self.deps.llm, self.config.generation
        )

    async def query(
        self,
        query: str,
        collection_name: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        contexts = await self.retrieve(query, collection_name, filters)
        answer = await self.generate(query, contexts)
        return {"answer": answer, "contexts": contexts}