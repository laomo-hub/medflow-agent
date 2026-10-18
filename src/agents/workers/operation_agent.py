# src/agents/workers/operation_agent.py

from langchain.agents import create_agent
from src.core.config import get_llm

OPERATION_SYSTEM_PROMPT = """你是医流（MedFlow）的运营数据助手。

你的职责：
1. 根据运营人员的自然语言问题，查询统计数据库中的运营数据
2. 生成数据报表和趋势分析
3. 辅助医院管理层做运营决策

重要安全规则：
- 只允许查询聚合统计数据，严禁返回患者个人信息
- 所有查询结果需脱敏处理
- 如果问题涉及患者隐私，拒绝回答并说明原因

回复格式：
- 数据摘要：xxx
- 关键指标：xxx
- 趋势分析：xxx
- 决策建议：xxx

注意：数据查询结果仅供内部运营参考。"""


def create_operation_agent():
    llm = get_llm(temperature=0.2)

    # 当前阶段：模拟数据返回
    # 未来可添加：NL2SQL 工具（查询 PostgreSQL）、数据可视化工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=OPERATION_SYSTEM_PROMPT,
        name="operation_agent",
    )


_operation_agent = None

def get_operation_agent():
    global _operation_agent
    if _operation_agent is None:
        _operation_agent = create_operation_agent()
    return _operation_agent