"""
知识 Agent 评估指标定义
包含：RAG Triad 三指标 + 医学领域自定义指标
"""
import numpy as np
from trulens.core.metric import Metric
from trulens.core.metric.selector import Selector
from trulens.providers.litellm import LiteLLM


def build_rag_triad_metrics(provider: LiteLLM) -> list[Metric]:
    """RAG Triad 三大核心指标"""

    m_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="答案相关性",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    m_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="上下文相关性",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )

    m_groundedness = Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="有据性",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
        },
    )

    return [m_answer_relevance, m_context_relevance, m_groundedness]


def build_medical_metrics(provider: LiteLLM) -> list[Metric]:
    """医学领域自定义指标"""

    m_safety = Metric(
        implementation=provider.harmfulness_with_cot_reasons,
        name="安全性",
        selectors={
            "text": Selector.select_record_output(),
        },
    )

    return [m_safety]


def build_all_metrics(provider: LiteLLM) -> list[Metric]:
    """全部指标（RAG Triad + 医学领域）"""
    return build_rag_triad_metrics(provider) + build_medical_metrics(provider)