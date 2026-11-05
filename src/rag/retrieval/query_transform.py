import json
from langchain_openai import ChatOpenAI

REWRITE_PROMPT = """你是医学信息检索专家。将用户的口语化问题改写为适合检索的标准医学查询。

规则:
1. 口语表达转医学术语 (如 "肚子疼" → "腹痛")
2. 保留关键约束条件 (年龄、性别、病史)
3. 复杂问题拆分为 2-3 个子查询

用户问题: {question}

返回 JSON: {{"rewritten": "改写后的主查询", "sub_queries": ["子查询1", "子查询2"]}}"""


async def rewrite_query(question: str, llm: ChatOpenAI) -> str:
    response = await llm.ainvoke(REWRITE_PROMPT.format(question=question))
    try:
        result = json.loads(response.content)
        return result.get("rewritten", question)
    except (json.JSONDecodeError, KeyError):
        return question