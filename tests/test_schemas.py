from mockinterview.schemas import (
    StageType, CandidateProfile, JobRequirement, InterviewContext,
    PlannedQuestion, StagePlan, InterviewPlan,
    AnswerEvaluation, QATurn, InterviewReport,
)


def test_stage_order():
    assert StageType.order() == [
        StageType.INTRO, StageType.TECH, StageType.PROJECT,
        StageType.BEHAVIORAL, StageType.CANDIDATE_Q, StageType.CLOSING,
    ]


def test_context_roundtrip():
    ctx = InterviewContext(
        candidate=CandidateProfile(
            summary="前端工程师", skills=["React", "TypeScript"],
            projects=["AI Code Review 工具"], highlights=["3年前端经验"],
        ),
        job=JobRequirement(
            title="AI 应用工程师", must_have=["Python", "LangGraph"],
            tech_stack=["FastAPI"], company_focus="企业 SaaS",
        ),
    )
    dumped = ctx.model_dump_json()
    assert "LangGraph" in dumped
    assert InterviewContext.model_validate_json(dumped) == ctx


def test_plan_structure():
    plan = InterviewPlan(stages=[
        StagePlan(stage=StageType.TECH, objective="考察 Python 基础",
                  questions=[PlannedQuestion(text="讲讲装饰器", probes=["实际用过吗"])]),
    ])
    assert plan.stages[0].stage is StageType.TECH


def test_evaluation_bounds():
    ev = AnswerEvaluation(score=7, relevance=8, depth=6,
                          watery=False, rationale="回答完整", suggested_followup=None)
    assert 0 <= ev.score <= 10
