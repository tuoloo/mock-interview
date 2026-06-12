from mockinterview.planner import make_plan
from mockinterview.schemas import (
    InterviewContext, CandidateProfile, JobRequirement, InterviewPlan, StageType,
)


def _ctx():
    return InterviewContext(
        candidate=CandidateProfile(summary="前端", skills=["React"],
                                   projects=["Code Review 工具"], highlights=[]),
        job=JobRequirement(title="AI 应用工程师", must_have=["Python"],
                           tech_stack=["FastAPI"], company_focus="SaaS"),
    )


def test_make_plan(fake_llm):
    fake_llm.queue_structured({
        "stages": [
            {"stage": "intro", "objective": "了解背景",
             "questions": [{"text": "自我介绍一下", "probes": []}]},
            {"stage": "tech", "objective": "考察 Python",
             "questions": [{"text": "讲讲装饰器", "probes": ["用过吗"]}]},
        ]
    })
    plan = make_plan(_ctx(), llm=fake_llm)
    assert isinstance(plan, InterviewPlan)
    assert plan.stages[1].stage is StageType.TECH
    # 上下文要进 prompt
    assert "Code Review" in fake_llm.calls[0]["prompt"]
