from __future__ import annotations

from .llm import LLMClient
from .schemas import InterviewContext, InterviewPlan, StageType

_SYSTEM = "你是资深面试官，为候选人设计针对性的结构化面试计划。只输出符合 schema 的 JSON。"

_PROMPT_TEMPLATE = """根据候选人画像与岗位要求，设计一场结构化面试计划。

# 候选人
{candidate}

# 岗位
{job}

要求：
- 严格按这 6 个阶段顺序产出：{stages}
- 每个阶段给 1-3 个针对性问题，问题必须结合候选人简历项目和岗位硬性要求
- project 阶段的问题要点名候选人的具体项目深挖
- candidate_q 阶段让候选人向面试官提问，closing 阶段收尾
"""


def make_plan(context: InterviewContext, *, llm: LLMClient) -> InterviewPlan:
    prompt = _PROMPT_TEMPLATE.format(
        candidate=context.candidate.model_dump_json(),
        job=context.job.model_dump_json(),
        stages=", ".join(s.value for s in StageType.order()),
    )
    data = llm.structured(prompt=prompt, schema=InterviewPlan.model_json_schema(), system=_SYSTEM)
    return InterviewPlan.model_validate(data)
