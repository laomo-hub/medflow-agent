"""Chat model factory for ordinary workers.

Supervisor and summary models keep their independent construction and unspecified
temperature; inquiry models preserve their explicit temperatures.
"""

from langchain_deepseek import ChatDeepSeek

from src.core.config import get_settings

settings = get_settings()


def get_chat_llm(temperature: float = 0.3) -> ChatDeepSeek:
    """返回关闭思考模式的 ChatDeepSeek 实例。"""
    return ChatDeepSeek(
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},  # 核心：关闭思考模式
    )
