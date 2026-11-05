"""
ECharts 图表配置生成器
将查询结果 + LLM 推荐的图表类型转换为 ECharts option JSON，供前端直接渲染。
"""
from __future__ import annotations
import pandas as pd


def to_echarts_option(data: list[dict], config: dict) -> dict:
    """
    根据 LLM 推荐的 config 和查询数据，生成 ECharts option。
    前端拿到后直接 chart.setOption(option) 即可。
    """
    if not data or config.get("chart_type") == "table":
        return _table_option(data, config)

    df = pd.DataFrame(data)
    chart_type = config.get("chart_type", "bar")
    title = config.get("title", "")
    x_col = config.get("x_column")
    y_col = config.get("y_column")
    color_col = config.get("color_column")

    if x_col and x_col not in df.columns:
        x_col = df.columns[0]
    if y_col and y_col not in df.columns:
        y_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]

    builders = {
        "bar": _bar_option,
        "line": _line_option,
        "pie": _pie_option,
        "scatter": _scatter_option,
        "heatmap": _heatmap_option,
    }

    builder = builders.get(chart_type, _table_option)
    return builder(df, title, x_col, y_col, color_col)


def _base_option(title: str) -> dict:
    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis"},
        "toolbox": {
            "feature": {
                "saveAsImage": {},
                "dataZoom": {},
                "restore": {},
            }
        },
    }


def _bar_option(df: pd.DataFrame, title: str, x_col: str, y_col: str, color_col: str | None) -> dict:
    option = _base_option(title)
    option["tooltip"]["trigger"] = "axis"

    if color_col and color_col in df.columns:
        groups = df.groupby(color_col)
        categories = df[x_col].unique().tolist()
        series = []
        for name, group in groups:
            values = []
            for cat in categories:
                row = group[group[x_col] == cat]
                values.append(float(row[y_col].iloc[0]) if len(row) > 0 else 0)
            series.append({"name": str(name), "type": "bar", "data": values})
        option["legend"] = {"top": "bottom"}
        option["xAxis"] = {"type": "category", "data": [str(c) for c in categories]}
        option["yAxis"] = {"type": "value"}
        option["series"] = series
    else:
        x_data = df[x_col].astype(str).tolist()
        y_data = df[y_col].tolist()
        option["xAxis"] = {"type": "category", "data": x_data, "axisLabel": {"rotate": 30}}
        option["yAxis"] = {"type": "value"}
        option["series"] = [{"type": "bar", "data": [_to_number(v) for v in y_data]}]

    return option


def _line_option(df: pd.DataFrame, title: str, x_col: str, y_col: str, color_col: str | None) -> dict:
    option = _base_option(title)
    option["tooltip"]["trigger"] = "axis"

    if color_col and color_col in df.columns:
        groups = df.groupby(color_col)
        categories = df[x_col].unique().tolist()
        series = []
        for name, group in groups:
            values = []
            for cat in categories:
                row = group[group[x_col] == cat]
                values.append(float(row[y_col].iloc[0]) if len(row) > 0 else 0)
            series.append({"name": str(name), "type": "line", "data": values, "smooth": True})
        option["legend"] = {"top": "bottom"}
        option["xAxis"] = {"type": "category", "data": [str(c) for c in categories]}
        option["yAxis"] = {"type": "value"}
        option["series"] = series
    else:
        x_data = df[x_col].astype(str).tolist()
        y_data = df[y_col].tolist()
        option["xAxis"] = {"type": "category", "data": x_data}
        option["yAxis"] = {"type": "value"}
        option["series"] = [{"type": "line", "data": [_to_number(v) for v in y_data], "smooth": True}]

    return option


def _pie_option(df: pd.DataFrame, title: str, x_col: str, y_col: str, color_col: str | None) -> dict:
    option = _base_option(title)
    option["tooltip"] = {"trigger": "item", "formatter": "{b}: {c} ({d}%)"}
    option["legend"] = {"orient": "vertical", "left": "left", "top": "middle"}

    pie_data = [
        {"name": str(row[x_col]), "value": _to_number(row[y_col])}
        for _, row in df.iterrows()
    ]
    option["series"] = [{
        "type": "pie",
        "radius": ["40%", "70%"],
        "center": ["60%", "50%"],
        "data": pie_data,
        "emphasis": {"itemStyle": {"shadowBlur": 10}},
    }]
    return option


def _scatter_option(df: pd.DataFrame, title: str, x_col: str, y_col: str, color_col: str | None) -> dict:
    option = _base_option(title)
    option["tooltip"]["trigger"] = "item"
    option["xAxis"] = {"type": "value", "name": x_col}
    option["yAxis"] = {"type": "value", "name": y_col}

    if color_col and color_col in df.columns:
        groups = df.groupby(color_col)
        series = []
        for name, group in groups:
            points = group[[x_col, y_col]].values.tolist()
            series.append({"name": str(name), "type": "scatter", "data": points})
        option["legend"] = {"top": "bottom"}
        option["series"] = series
    else:
        points = df[[x_col, y_col]].values.tolist()
        option["series"] = [{"type": "scatter", "data": points}]

    return option


def _heatmap_option(df: pd.DataFrame, title: str, x_col: str, y_col: str, color_col: str | None) -> dict:
    option = _base_option(title)
    option["tooltip"]["trigger"] = "item"

    if x_col and y_col and color_col and color_col in df.columns:
        x_categories = df[x_col].unique().tolist()
        y_categories = df[y_col].unique().tolist()
        heat_data = []
        for _, row in df.iterrows():
            xi = x_categories.index(row[x_col])
            yi = y_categories.index(row[y_col])
            heat_data.append([xi, yi, _to_number(row[color_col])])

        option["xAxis"] = {"type": "category", "data": [str(c) for c in x_categories]}
        option["yAxis"] = {"type": "category", "data": [str(c) for c in y_categories]}
        option["visualMap"] = {"min": 0, "max": max(d[2] for d in heat_data) if heat_data else 1, "calculable": True}
        option["series"] = [{"type": "heatmap", "data": heat_data}]
    else:
        return _table_option(df.to_dict("records"), {"title": title})

    return option


def _table_option(data, config: dict) -> dict:
    """表格类型：返回原始数据，前端用 table 组件渲染"""
    return {
        "chart_type": "table",
        "title": config.get("title", "查询结果"),
        "data": data if isinstance(data, list) else [],
    }


def _to_number(val) -> float | int:
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return 0
