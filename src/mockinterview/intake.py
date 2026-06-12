from __future__ import annotations

from .llm import LLMClient
from .schemas import InterviewContext

_SYSTEM = "你是资深技术招聘专家，负责把简历、岗位描述、公司信息抽取成结构化数据。只输出符合 schema 的 JSON。"

_PROMPT_TEMPLATE = """请基于以下材料抽取候选人画像与岗位要求。

# 简历
{resume}

# 岗位描述
{jd}

# 公司信息
{company}

要求：
- candidate.projects 提取可深挖的真实项目
- job.must_have 提取岗位硬性要求
- job.company_focus 用一句话概括公司业务侧重
"""


def build_context(*, resume_text: str, jd_text: str, company_text: str,
                  llm: LLMClient) -> InterviewContext:
    prompt = _PROMPT_TEMPLATE.format(resume=resume_text, jd=jd_text, company=company_text)
    data = llm.structured(prompt=prompt, schema=InterviewContext.model_json_schema(), system=_SYSTEM)
    return InterviewContext.model_validate(data)
