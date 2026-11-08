"""
用 @instrument 装饰器标记 RAG 关键步骤，
TruLens 基于 OpenTelemetry 自动追踪每步的输入/输出/延迟。
"""
from trulens.core.otel.instrument import instrument, SpanAttributes
from src.rag.engine import RAGEngine

SpanType = SpanAttributes.SpanType


class TrackedRAG:
    def __init__(self, engine: RAGEngine):
        self.engine = engine

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    async def retrieve(self, query: str) -> list[str]:
        """检索步骤 - TruLens 追踪返回值作为 context"""
        results = await self.engine.retrieve(query)
        return [r["text"] for r in results]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    async def generate(self, query: str, contexts: list[str]) -> str:
        """生成步骤 - TruLens 追踪返回值作为 answer"""
        context_dicts = [{"text": c} for c in contexts]
        return await self.engine.generate(query, context_dicts)

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    async def query(self, query: str) -> str:
        """完整 RAG 流程入口"""
        contexts = await self.retrieve(query)
        answer = await self.generate(query, contexts)
        return answer
