"""
NL2SQL + ChatBI API 路由
前后分离场景：返回 ECharts option JSON，前端直接 setOption 渲染。
"""
from __future__ import annotations
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from src.infra.llm import get_chat_llm
from loguru import logger

from src.core.config import get_settings
from src.infra.database import get_db
from src.nl2sql.engine import run_query, ConversationContext
from src.nl2sql.chart_advisor import recommend_chart
from src.nl2sql.echarts_builder import to_echarts_option

settings = get_settings()
router = APIRouter(prefix="/api/v1/bi", tags=["ChatBI"])

# 内存中维护多轮对话上下文（生产环境可改为 Redis）
_sessions: dict[str, ConversationContext] = {}


def _get_llm():
    return get_chat_llm(temperature=0.1)


def _get_context(session_id: str) -> ConversationContext:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationContext()
    return _sessions[session_id]


# ── Request / Response Schema ─────────────────────────────────────────

class BiQueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    role: str = "operator"
    department_id: int | None = None


class BiQueryResponse(BaseModel):
    success: bool
    question: str
    sql: str
    summary: str
    chart_type: str
    echarts_option: dict
    data: list[dict]
    columns: list[str]
    row_count: int
    error: str = ""


class BiHistoryResponse(BaseModel):
    history: list[dict]


# ── 核心接口 ──────────────────────────────────────────────────────────

@router.post("/query", response_model=BiQueryResponse)
async def bi_query(req: BiQueryRequest, db: AsyncSession = Depends(get_db)):
    """
    ChatBI 查询接口。
    输入自然语言问题，返回 SQL + 数据 + ECharts option JSON。
    前端拿到 echarts_option 后直接 chart.setOption(option) 渲染。
    """
    llm = _get_llm()
    context = _get_context(req.session_id)

    result = await run_query(
        question=req.question,
        llm=llm,
        db=db,
        role=req.role,
        department_id=req.department_id,
        context=context,
    )

    if not result.success:
        return BiQueryResponse(
            success=False,
            question=req.question,
            sql=result.sql,
            summary="",
            chart_type="table",
            echarts_option={},
            data=[],
            columns=[],
            row_count=0,
            error=result.error,
        )

    chart_config = await recommend_chart(req.question, result.data, result.columns, llm)
    echarts_option = to_echarts_option(result.data, chart_config)
    chart_type = chart_config.get("chart_type", "table")

    return BiQueryResponse(
        success=True,
        question=req.question,
        sql=result.sql,
        summary=result.summary,
        chart_type=chart_type,
        echarts_option=echarts_option,
        data=result.data,
        columns=result.columns,
        row_count=result.row_count,
    )


@router.get("/history/{session_id}", response_model=BiHistoryResponse)
async def bi_history(session_id: str):
    """获取某个会话的查询历史"""
    context = _get_context(session_id)
    history = [
        {
            "question": r.question,
            "sql": r.sql,
            "summary": r.summary,
            "row_count": r.row_count,
            "success": r.success,
            "error": r.error,
        }
        for r in context.history
    ]
    return BiHistoryResponse(history=history)


@router.delete("/history/{session_id}")
async def bi_clear_history(session_id: str):
    """清空会话历史"""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"message": "会话已清空"}
