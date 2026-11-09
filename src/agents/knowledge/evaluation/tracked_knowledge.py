"""
TrackedKnowledge — TruLens 追踪包装类

将现有的 5 条检索通路统一包装为 retrieve → generate → query 三段式，
通过 @instrument 装饰器注入 OpenTelemetry 追踪。

不修改任何现有检索函数，只做适配和转发。
"""
from __future__ import annotations
import json
from trulens.core.otel.instrument import instrument, SpanAttributes
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

SpanType = SpanAttributes.SpanType


class TrackedKnowledge:
    """
    知识 Agent 的 TruLens 追踪包装。
    每个实例绑定一条检索通路（channel），通过 app_version 区分。
    """

    def __init__(
        self,
        channel: str,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        milvus_client: MilvusClient,
        neo4j_driver: AsyncDriver,
        db_session: AsyncSession | None = None,
        role: str = "patient",
    ):
        self.channel = channel
        self.llm = llm
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.neo4j_driver = neo4j_driver
        self.db_session = db_session
        self.role = role

    @instrument(
        span_type=SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    async def retrieve(self, query: str) -> list[str]:
        """
        检索步骤 — 根据 channel 分发到对应的检索函数。
        返回 list[str]，每个元素是一个检索片段的文本。
        """
        if self.channel == "doc_rag":
            from src.agents.knowledge.doc_rag import search_docs_raw, format_doc_context
            hits = await search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                llm=self.llm,
                use_hyde=True,
            )
            return [h["text"] for h in hits] if hits else []

        elif self.channel == "graph_rag":
            from src.agents.knowledge.graph_rag import search_graph_raw
            records = await search_graph_raw(query, self.neo4j_driver, self.llm)
            return [json.dumps(r, ensure_ascii=False) for r in records] if records else []

        elif self.channel == "nl2sql":
            from src.agents.knowledge.nl2sql import search_sql_raw
            if not self.db_session:
                return ["数据库连接不可用"]
            data, sql = await search_sql_raw(query, self.llm, self.db_session)
            if isinstance(data, str):
                return [data]
            return [json.dumps(d, ensure_ascii=False) for d in data[:10]]

        elif self.channel == "fusion":
            from src.agents.knowledge.doc_rag import search_docs_raw
            from src.agents.knowledge.graph_rag import search_graph_raw
            import asyncio

            doc_task = search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                llm=self.llm,
                use_hyde=True,
            )
            graph_task = search_graph_raw(query, self.neo4j_driver, self.llm)
            doc_hits, graph_records = await asyncio.gather(
                doc_task, graph_task, return_exceptions=True
            )

            contexts = []
            if isinstance(doc_hits, list):
                contexts.extend([h["text"] for h in doc_hits])
            if isinstance(graph_records, list):
                contexts.extend([json.dumps(r, ensure_ascii=False) for r in graph_records])
            return contexts

        elif self.channel == "prescription_review":
            from src.agents.knowledge.doc_rag import search_docs_raw
            from src.agents.knowledge.graph_rag import search_graph_raw
            import asyncio

            doc_task = search_docs_raw(
                question=query,
                embedding_model=self.embedding_model,
                milvus_client=self.milvus_client,
                doc_type="drug_instruction",
            )
            graph_task = search_graph_raw(query, self.neo4j_driver, self.llm)
            doc_hits, graph_records = await asyncio.gather(
                doc_task, graph_task, return_exceptions=True
            )

            contexts = []
            if isinstance(doc_hits, list):
                contexts.extend([h["text"] for h in doc_hits])
            if isinstance(graph_records, list):
                contexts.extend([json.dumps(r, ensure_ascii=False) for r in graph_records])
            return contexts

        return []

    @instrument(span_type=SpanType.GENERATION)
    async def generate(self, query: str, contexts: list[str]) -> str:
        """生成步骤 — 基于检索结果调用 LLM 生成回答"""
        if not contexts:
            return "未找到相关信息。"

        context_str = "\n---\n".join(contexts[:10])
        prompt = (
            f"你是医流（MedFlow）的知识问答助手。根据以下检索结果回答用户问题。\n"
            f"如果检索结果中没有答案，请明确告知。\n\n"
            f"用户角色：{self.role}\n"
            f"检索结果：\n{context_str}\n\n"
            f"用户问题：{query}"
        )
        response = await self.llm.ainvoke([SystemMessage(content=prompt)])
        return response.content

    @instrument()
    async def query(self, query: str) -> str:
        """完整 RAG 流程入口"""
        contexts = await self.retrieve(query)
        answer = await self.generate(query, contexts)
        return answer