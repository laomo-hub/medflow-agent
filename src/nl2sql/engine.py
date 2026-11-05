"""
NL2SQL 核心引擎
负责：自然语言 → SQL 生成 → 执行 → 结果返回
"""
from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass, field
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.nl2sql.prompts import (
    SCHEMA_PROMPT, NL2SQL_SYSTEM_PROMPT, FOLLOWUP_PROMPT, SUMMARY_PROMPT,
)
from src.nl2sql.security import validate_sql, apply_role_filter


MAX_RETRIES = 2
SQL_TIMEOUT = 10


@dataclass
class QueryResult:
    """一次 NL2SQL 查询的完整结果"""
    question: str
    sql: str
    data: list[dict] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    summary: str = ""
    error: str = ""
    success: bool = True


@dataclass
class ConversationContext:
    """多轮对话上下文"""
    history: list[QueryResult] = field(default_factory=list)

    @property
    def last_result(self) -> QueryResult | None:
        return self.history[-1] if self.history else None

    def add(self, result: QueryResult):
        self.history.append(result)
        if len(self.history) > 10:
            self.history = self.history[-10:]


async def generate_sql(
    question: str,
    llm: BaseChatModel,
    role: str = "operator",
    department_id: int | None = None,
    context: ConversationContext | None = None,
    error_hint: str = "",
) -> str:
    """LLM 生成 SQL"""
    if context and context.last_result and context.last_result.success:
        prompt = FOLLOWUP_PROMPT.format(
            previous_sql=context.last_result.sql,
            previous_summary=context.last_result.summary[:500],
            question=question,
            schema=SCHEMA_PROMPT,
        )
    else:
        prompt = NL2SQL_SYSTEM_PROMPT.format(
            schema=SCHEMA_PROMPT,
            role=role,
            department_id=department_id or "N/A",
        )

    messages = [SystemMessage(content=prompt)]
    if error_hint:
        messages.append(HumanMessage(content=f"上一次生成的 SQL 执行报错：{error_hint}\n请修正后重新生成。\n\n{question}"))
    else:
        messages.append(HumanMessage(content=question))

    response = await llm.ainvoke(messages)
    sql = response.content.strip()
    if "```" in sql:
        sql = sql.split("```")[1].lstrip("sql").strip()
    return sql


async def execute_sql(sql: str, db: AsyncSession) -> tuple[list[dict], list[str]]:
    """执行 SQL 并返回 (rows, columns)"""
    result = await asyncio.wait_for(
        db.execute(text(sql)),
        timeout=SQL_TIMEOUT,
    )
    columns = list(result.keys())
    rows = [dict(row) for row in result.mappings().all()]
    return rows, columns


async def generate_summary(
    question: str, data: list[dict], llm: BaseChatModel,
) -> str:
    """LLM 生成数据摘要"""
    result_str = json.dumps(data[:20], ensure_ascii=False, default=str)
    prompt = SUMMARY_PROMPT.format(question=question, result=result_str)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content


async def run_query(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
    role: str = "operator",
    department_id: int | None = None,
    context: ConversationContext | None = None,
) -> QueryResult:
    """
    完整的 NL2SQL 流程：生成 SQL → 校验 → 执行 → 摘要
    支持重试和多轮对话上下文。
    """
    error_hint = ""

    for attempt in range(MAX_RETRIES + 1):
        raw_sql = await generate_sql(
            question, llm, role, department_id, context, error_hint,
        )
        logger.info(f"NL2SQL (attempt {attempt + 1}): {raw_sql}")

        valid, validated = validate_sql(raw_sql)
        if not valid:
            result = QueryResult(
                question=question, sql=raw_sql,
                error=f"安全校验失败: {validated}", success=False,
            )
            if context:
                context.add(result)
            return result
        # 应用角色过滤: 管理员、运营、医生、患者；
        validated = apply_role_filter(validated, role, department_id)

        try:
            data, columns = await execute_sql(validated, db)
            summary = await generate_summary(question, data, llm)

            result = QueryResult(
                question=question, sql=validated,
                data=data, columns=columns,
                row_count=len(data), summary=summary,
            )
            if context:
                context.add(result)
            return result

        except asyncio.TimeoutError:
            result = QueryResult(
                question=question, sql=validated,
                error=f"查询超时（{SQL_TIMEOUT}秒）", success=False,
            )
            if context:
                context.add(result)
            return result

        except Exception as e:
            error_hint = str(e)
            logger.warning(f"SQL 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_RETRIES:
                result = QueryResult(
                    question=question, sql=validated,
                    error=f"执行失败: {error_hint}", success=False,
                )
                if context:
                    context.add(result)
                return result

    result = QueryResult(question=question, sql="", error="未知错误", success=False)
    if context:
        context.add(result)
    return result
