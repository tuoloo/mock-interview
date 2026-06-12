from mockinterview.intake import build_context
from mockinterview.schemas import InterviewContext


def test_build_context_uses_llm(fake_llm):
    fake_llm.queue_structured({
        "candidate": {"summary": "前端工程师", "skills": ["React"],
                      "projects": ["Code Review 工具"], "highlights": ["3年经验"]},
        "job": {"title": "AI 应用工程师", "must_have": ["Python"],
                "tech_stack": ["FastAPI"], "company_focus": "SaaS"},
    })
    ctx = build_context(
        resume_text="我是前端，做过 Code Review 工具",
        jd_text="招 AI 应用工程师，要求 Python",
        company_text="企业 SaaS 公司",
        llm=fake_llm,
    )
    assert isinstance(ctx, InterviewContext)
    assert ctx.job.title == "AI 应用工程师"
    # 三段原料都应进入 prompt
    prompt = fake_llm.calls[0]["prompt"]
    assert "Code Review" in prompt and "Python" in prompt and "SaaS" in prompt
