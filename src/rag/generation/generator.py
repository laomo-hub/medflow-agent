from langchain_openai import ChatOpenAI
from src.rag.config import GenerationConfig

DOC_QA_PROMPT = """你是专业的医学知识助手。请根据以下参考资料回答用户问题。

要求:
1. 仅基于参考资料回答，不要编造信息
2. 资料不足时明确说明"根据现有资料无法确定"
3. 在回答中标注信息来源 [文档名]
4. 使用专业但易懂的语言

参考资料:
{context}

用户问题: {question}

回答:"""


async def generate_answer(
    query: str,
    contexts: list[dict],
    llm: ChatOpenAI,
    config: GenerationConfig,
) -> str:
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        source = ctx.get("doc_name", "未知来源")
        text = ctx.get("parent_text") or ctx["text"]
        context_parts.append(f"[{i}] 来源: {source}\n{text}")

    context_str = "\n---\n".join(context_parts)
    prompt = DOC_QA_PROMPT.format(context=context_str, question=query)

    response = await llm.ainvoke(
        prompt, temperature=config.temperature, max_tokens=config.max_tokens,
    )
    return response.content