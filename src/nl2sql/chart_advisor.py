"""
图表推荐 + Plotly 图表生成
LLM 根据查询结果推荐图表类型，然后用 Plotly 渲染。
"""
from __future__ import annotations
import json
import re
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.nl2sql.prompts import CHART_ADVISOR_PROMPT


def _extract_json(content: str) -> dict | None:
    """从 LLM 输出中稳健地提取 JSON 对象"""
    # 去掉 <think>...</think> 推理块
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # 优先匹配 ```json ... ``` 或 ``` ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
    else:
        # 退化：抓第一个 { 到最后一个 }
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = content[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}; 原始片段: {candidate[:200]}")
        return None


async def recommend_chart(
    question: str,
    data: list[dict],
    columns: list[str],
    llm: BaseChatModel,
) -> dict:
    """LLM 推荐图表类型和配置"""
    if not data:
        return {"chart_type": "table", "title": "无数据", "description": "查询结果为空"}

    df = pd.DataFrame(data)
    preview = df.head(5).to_string(index=False)

    prompt = CHART_ADVISOR_PROMPT.format(
        question=question,
        preview=preview,
        columns=columns,
        row_count=len(data),
    )

    response = await llm.ainvoke([SystemMessage(content=prompt)])
    content = response.content.strip()
    logger.debug(f"[chart_advisor] LLM raw response: {content[:500]}")

    config = _extract_json(content)
    if config is None:
        logger.warning(f"[chart_advisor] 解析失败，降级为 table; raw: {content[:300]}")
        return {"chart_type": "table", "title": "查询结果", "description": ""}

    logger.info(f"[chart_advisor] 推荐配置: {config}")
    return config


def _pick_columns(df: pd.DataFrame, x_col, y_col, color_col) -> tuple[str | None, str | None, str | None]:
    """当 LLM 给的列名缺失或为 None 时，用启发式回退"""
    cols = list(df.columns)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric = [c for c in cols if c not in numeric_cols]

    if not x_col or x_col not in cols:
        x_col = non_numeric[0] if non_numeric else cols[0]
    if not y_col or y_col not in cols:
        # y 优先选数值列，且不同于 x
        y_candidates = [c for c in numeric_cols if c != x_col]
        y_col = y_candidates[0] if y_candidates else (cols[-1] if len(cols) > 1 else None)
    if color_col and color_col not in cols:
        color_col = None
    return x_col, y_col, color_col


def render_chart(data: list[dict], config: dict) -> go.Figure | None:
    """根据推荐配置生成 Plotly 图表"""
    if not data:
        logger.info("[render_chart] 无数据，跳过渲染")
        return None
    if config.get("chart_type") == "table":
        logger.info("[render_chart] chart_type=table，跳过渲染")
        return None

    df = pd.DataFrame(data)
    chart_type = config.get("chart_type", "bar")
    title = config.get("title", "")
    x_col, y_col, color_col = _pick_columns(
        df,
        config.get("x_column"),
        config.get("y_column"),
        config.get("color_column"),
    )
    logger.debug(f"[render_chart] type={chart_type} x={x_col} y={y_col} color={color_col} cols={list(df.columns)}")

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "heatmap":
            if x_col and y_col and color_col:
                pivot = df.pivot_table(index=y_col, columns=x_col, values=color_col, aggfunc="sum")
                fig = px.imshow(pivot, title=title, aspect="auto")
            else:
                numeric_cols = df.select_dtypes(include="number")
                fig = px.imshow(numeric_cols.corr(), title=title or "相关性热力图", aspect="auto")
        else:
            logger.warning(f"[render_chart] 未知图表类型: {chart_type}")
            return None

        fig.update_layout(
            template="plotly_white",
            font=dict(family="Microsoft YaHei, sans-serif"),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    except Exception as e:
        logger.warning(f"[render_chart] 图表渲染失败: {e}; type={chart_type} x={x_col} y={y_col}")
        return None
