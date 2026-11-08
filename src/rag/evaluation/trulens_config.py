from typing import Any, ClassVar

from pydantic import Field

from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM
from src.core.config import get_settings

settings = get_settings()


class _EvaluationLiteLLM(LiteLLM):
    """Keep the thinking setting when TruLens filters completion arguments."""

    completion_args: dict[str, Any] = Field(default_factory=dict)

    _LITELLM_ROUTING_PARAMS: ClassVar[frozenset[str]] = (
        LiteLLM._LITELLM_ROUTING_PARAMS | {"extra_body"}
    )


def get_trulens_session() -> TruSession:
    """
    初始化 TruLens 会话。
    评估结果存入 PostgreSQL，与业务数据隔离。
    """
    session = TruSession(
        database_url=f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
    )
    return session


def get_llm_provider() -> LiteLLM:
    """
    评估用 LLM Provider。
    通过 LiteLLM 对接 DeepSeek (OpenAI 兼容 API)。
    """
    return _EvaluationLiteLLM(
        model_engine=f"openai/{settings.CHAT_MODEL}",
        completion_kwargs={
            "api_key": settings.DEEPSEEK_API_KEY,
            "api_base": "https://api.deepseek.com",
            "extra_body": {"thinking": {"type": "disabled"}},
        },
    )


def launch_dashboard(session: TruSession | None = None, port: int = 8501):
    """启动 TruLens Dashboard（Streamlit 界面）。"""
    from trulens.dashboard import run_dashboard
    s = session or get_trulens_session()
    run_dashboard(session=s, port=port)