"""
TruLens 会话配置 + LLM Provider + Dashboard 启动
"""
from trulens.core import TruSession
from trulens.providers.litellm import LiteLLM
from src.core.config import get_settings
from src.rag.evaluation.trulens_config import get_llm_provider as get_evaluation_provider

settings = get_settings()


def get_trulens_session() -> TruSession:
    """评估结果存入独立的 PostgreSQL 数据库"""
    return TruSession(
        database_url=f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/trulens_eval"
    )


def get_llm_provider() -> LiteLLM:
    """
    评估用 LLM Provider（LLM-as-Judge）。
    注意：参数名是 model_engine，不是 model_name。
    前缀 openai/ 表示走 OpenAI 兼容协议。
    """
    return get_evaluation_provider()


def launch_dashboard(port: int = 8501):
    """启动 TruLens Streamlit Dashboard"""
    from trulens.dashboard import run_dashboard
    session = get_trulens_session()
    run_dashboard(session=session, port=port)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    launch_dashboard(port)