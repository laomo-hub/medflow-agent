from langchain_openai import ChatOpenAI

HYDE_PROMPT = """你是一位医学专家。请根据以下问题，撰写一段 100-200 字的假设性回答。
请尽量包含相关的医学术语和关键信息。

问题: {question}

假设性回答:"""


async def generate_hypothetical_doc(question: str, llm: ChatOpenAI) -> str:
    response = await llm.ainvoke(HYDE_PROMPT.format(question=question))
    return response.content