# src/agents/workers/drug_agent.py

from langchain.agents import create_agent
from src.core.config import get_llm

DRUG_SYSTEM_PROMPT = """你是医流（MedFlow）的药物咨询助手。

你的职责：
1. 根据患者病情推荐合适的药物
2. 检测药物之间的相互作用（药物交互检测）
3. 审查处方，找出潜在风险（剂量、禁忌症、过敏史冲突）
4. 用通俗语言解释用药注意事项

回复格式：
- 推荐药物：xxx（用途：xxx，用法：xxx）
- 药物交互风险：xxx
- 注意事项：xxx
- 禁忌提示：xxx

注意：药物建议仅供参考，请遵医嘱用药。"""


def create_drug_agent():
    llm = get_llm(temperature=0.1)

    # 当前阶段：模拟推荐
    # 未来可添加：Neo4j 知识图谱查询工具（NL2Cypher）、药物数据库检索工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=DRUG_SYSTEM_PROMPT,
        name="drug_agent",
    )


_drug_agent = None

def get_drug_agent():
    global _drug_agent
    if _drug_agent is None:
        _drug_agent = create_drug_agent()
    return _drug_agent