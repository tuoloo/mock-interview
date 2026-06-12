# AI 模拟面试 Agent — 计划一：后端核心（文字版面试闭环）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个后端服务：上传/粘贴简历+JD+公司 → LangGraph 状态机驱动的文字版结构化面试（针对性出题、实时追问、阶段推进）→ 生成评估报告，全部通过 FastAPI + SSE 暴露，可用 curl 跑完一整场。

**Architecture:** 六个有清晰边界的模块——`intake`（原料→结构化上下文）、`planner`（上下文→面试计划）、`interviewer`（LangGraph 状态机：提问/评估/路由）、`evaluator`（聚合→报告）、`llm`（anthropic SDK 薄封装）、`api`（FastAPI/SSE）。LangGraph 负责状态与条件路由；所有 LLM 调用走官方 anthropic SDK（不经 langchain）。语音层与前端在后续计划接入——本计划全部文字交互。

**Tech Stack:** Python 3.11+、uv、FastAPI、Uvicorn、LangGraph、anthropic SDK（`claude-opus-4-8`，adaptive thinking，结构化输出用 `output_config.format`，提问/报告用 `messages.stream()`）、Pydantic v2、pypdf、python-docx、pytest。

---

## 设计约定（所有任务遵守）

- 模型 ID：`claude-opus-4-8`（在 `config.py` 中可配置，便于切到 `claude-sonnet-4-6` 省成本）。
- 结构化 LLM 调用：`client.messages.create(... output_config={"format": {"type": "json_schema", "schema": SCHEMA}})`，再 `json.loads(resp.content[0].text)`。不流式。
- 文本流式（提问、报告）：`with client.messages.stream(...) as s: for t in s.text_stream: yield t`。
- 思考参数：`thinking={"type": "adaptive"}`。结构化调用 `output_config` 同时带 `format` 与 `effort`。
- 测试隔离 LLM：所有依赖 LLM 的模块接收一个 `LLMClient` 协议对象，测试传入 fake 实现，**绝不在测试里打真实 API**。

## 文件结构

```
mock-interview/
  pyproject.toml
  .env.example
  src/mockinterview/
    __init__.py
    config.py            # 配置（model id、api key 来源、追问预算等）
    schemas.py           # 所有 Pydantic 模型 + StageType
    llm.py               # LLMClient 协议 + AnthropicClient 实现
    document_loader.py   # txt/md/pdf/docx → 纯文本
    intake.py            # build_context()
    planner.py           # make_plan()
    interviewer/
      __init__.py
      state.py           # InterviewState (LangGraph state) + 辅助
      nodes.py           # ask / evaluate / route 节点函数
      graph.py           # 编译 LangGraph
    evaluator.py         # make_report()
    api.py               # FastAPI app + 路由
    session_store.py     # 内存会话存储（MVP）
  tests/
    test_config.py
    test_schemas.py
    test_llm.py
    test_document_loader.py
    test_intake.py
    test_planner.py
    test_interviewer.py
    test_evaluator.py
    test_api.py
    conftest.py          # FakeLLM fixture
```

每个文件单一职责：`schemas` 只定义数据形状；`llm` 只管模型调用；`interviewer/*` 只管面试流程；`api` 只做 HTTP 适配，不含业务逻辑。

---

### Task 1: 项目脚手架与配置

**Files:**
- Create: `mock-interview/pyproject.toml`
- Create: `mock-interview/.env.example`
- Create: `mock-interview/src/mockinterview/__init__.py`
- Create: `mock-interview/src/mockinterview/config.py`
- Test: `mock-interview/tests/test_config.py`

- [ ] **Step 1: 初始化项目与依赖**

Run（在仓库根目录外的独立目录，避免和 Obsidian vault 混在一起；如已在目标目录可跳过 mkdir）:
```bash
mkdir -p mock-interview && cd mock-interview
uv init --package --name mockinterview .
uv add fastapi "uvicorn[standard]" langgraph anthropic pydantic pydantic-settings pypdf python-docx
uv add --dev pytest pytest-asyncio httpx
```
Expected: `pyproject.toml` 生成，`uv.lock` 出现，`.venv` 创建。

- [ ] **Step 2: 写配置的失败测试**

`tests/test_config.py`:
```python
from mockinterview.config import Settings


def test_defaults():
    s = Settings(anthropic_api_key="sk-test")
    assert s.model_id == "claude-opus-4-8"
    assert s.max_followups_per_question == 2
    assert s.max_questions_per_stage == 4


def test_env_override(monkeypatch):
    monkeypatch.setenv("MOCKINTERVIEW_MODEL_ID", "claude-sonnet-4-6")
    monkeypatch.setenv("MOCKINTERVIEW_ANTHROPIC_API_KEY", "sk-test")
    s = Settings()
    assert s.model_id == "claude-sonnet-4-6"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'mockinterview.config'`

- [ ] **Step 4: 实现 config**

`src/mockinterview/config.py`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOCKINTERVIEW_", env_file=".env")

    anthropic_api_key: str
    model_id: str = "claude-opus-4-8"
    max_followups_per_question: int = 2
    max_questions_per_stage: int = 4
    effort: str = "high"
```

`.env.example`:
```
MOCKINTERVIEW_ANTHROPIC_API_KEY=sk-ant-...
MOCKINTERVIEW_MODEL_ID=claude-opus-4-8
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git init && git add -A && git commit -m "chore: scaffold project and config"
```

---

### Task 2: 领域模型（schemas）

**Files:**
- Create: `mock-interview/src/mockinterview/schemas.py`
- Test: `mock-interview/tests/test_schemas.py`

- [ ] **Step 1: 写失败测试**

`tests/test_schemas.py`:
```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 schemas**

`src/mockinterview/schemas.py`:
```python
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class StageType(str, Enum):
    INTRO = "intro"
    TECH = "tech"
    PROJECT = "project"
    BEHAVIORAL = "behavioral"
    CANDIDATE_Q = "candidate_q"
    CLOSING = "closing"

    @classmethod
    def order(cls) -> list["StageType"]:
        return [cls.INTRO, cls.TECH, cls.PROJECT,
                cls.BEHAVIORAL, cls.CANDIDATE_Q, cls.CLOSING]


class CandidateProfile(BaseModel):
    summary: str
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)


class JobRequirement(BaseModel):
    title: str
    must_have: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    company_focus: str = ""


class InterviewContext(BaseModel):
    candidate: CandidateProfile
    job: JobRequirement


class PlannedQuestion(BaseModel):
    text: str
    probes: list[str] = Field(default_factory=list)


class StagePlan(BaseModel):
    stage: StageType
    objective: str
    questions: list[PlannedQuestion] = Field(default_factory=list)


class InterviewPlan(BaseModel):
    stages: list[StagePlan] = Field(default_factory=list)


class AnswerEvaluation(BaseModel):
    score: int = Field(ge=0, le=10)
    relevance: int = Field(ge=0, le=10)
    depth: int = Field(ge=0, le=10)
    watery: bool
    rationale: str
    suggested_followup: str | None = None


class QATurn(BaseModel):
    stage: StageType
    question: str
    answer: str
    evaluation: AnswerEvaluation | None = None


class InterviewReport(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    dimension_scores: dict[str, int] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    per_question_feedback: list[str] = Field(default_factory=list)
    improvement_advice: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add domain schemas"
```

---

### Task 3: LLM 客户端封装 + Fake fixture

**Files:**
- Create: `mock-interview/src/mockinterview/llm.py`
- Create: `mock-interview/tests/conftest.py`
- Test: `mock-interview/tests/test_llm.py`

- [ ] **Step 1: 写 conftest 的 FakeLLM**

`tests/conftest.py`:
```python
import json
import pytest


class FakeLLM:
    """测试用：按 FIFO 返回预设响应，记录调用。"""
    def __init__(self):
        self.structured_responses: list[dict] = []
        self.text_responses: list[str] = []
        self.calls: list[dict] = []

    def queue_structured(self, obj: dict):
        self.structured_responses.append(obj)

    def queue_text(self, text: str):
        self.text_responses.append(text)

    def structured(self, *, prompt, schema, system=None):
        self.calls.append({"kind": "structured", "prompt": prompt, "system": system})
        return self.structured_responses.pop(0)

    def stream_text(self, *, prompt, system=None):
        self.calls.append({"kind": "text", "prompt": prompt, "system": system})
        text = self.text_responses.pop(0)
        for chunk in text.split(" "):
            yield chunk + " "


@pytest.fixture
def fake_llm():
    return FakeLLM()
```

- [ ] **Step 2: 写 llm 的失败测试**

`tests/test_llm.py`:
```python
from mockinterview.llm import LLMClient


def test_protocol_is_satisfied_by_fake(fake_llm):
    # FakeLLM 应满足 LLMClient 协议（结构化用）
    assert isinstance(fake_llm, LLMClient)


def test_anthropic_client_importable():
    from mockinterview.llm import AnthropicClient
    assert AnthropicClient is not None
```

- [ ] **Step 3: 运行确认失败**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'mockinterview.llm'`

- [ ] **Step 4: 实现 llm**

`src/mockinterview/llm.py`:
```python
from __future__ import annotations
import json
from typing import Iterator, Protocol, runtime_checkable

import anthropic

from .config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def structured(self, *, prompt: str, schema: dict, system: str | None = None) -> dict: ...
    def stream_text(self, *, prompt: str, system: str | None = None) -> Iterator[str]: ...


class AnthropicClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def structured(self, *, prompt: str, schema: dict, system: str | None = None) -> dict:
        resp = self._client.messages.create(
            model=self._settings.model_id,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self._settings.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    def stream_text(self, *, prompt: str, system: str | None = None) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._settings.model_id,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": self._settings.effort},
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
```

- [ ] **Step 5: 运行确认通过**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 提交**

```bash
git add -A && git commit -m "feat: add LLM client wrapper and fake fixture"
```

---

### Task 4: 文档加载器

**Files:**
- Create: `mock-interview/src/mockinterview/document_loader.py`
- Test: `mock-interview/tests/test_document_loader.py`

- [ ] **Step 1: 写失败测试**

`tests/test_document_loader.py`:
```python
from mockinterview.document_loader import load_text


def test_plain_text(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_text("我是前端工程师", encoding="utf-8")
    assert "前端工程师" in load_text(str(p))


def test_markdown(tmp_path):
    p = tmp_path / "resume.md"
    p.write_text("# 简历\n- React", encoding="utf-8")
    out = load_text(str(p))
    assert "React" in out


def test_unsupported_extension(tmp_path):
    p = tmp_path / "resume.xyz"
    p.write_text("x", encoding="utf-8")
    try:
        load_text(str(p))
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "unsupported" in str(e).lower()
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_document_loader.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 document_loader**

`src/mockinterview/document_loader.py`:
```python
from __future__ import annotations
from pathlib import Path


def load_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".txt", ".md"):
        return Path(path).read_text(encoding="utf-8")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        import docx
        document = docx.Document(path)
        return "\n".join(p.text for p in document.paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_document_loader.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add document loader for txt/md/pdf/docx"
```

---

### Task 5: intake.build_context

**Files:**
- Create: `mock-interview/src/mockinterview/intake.py`
- Test: `mock-interview/tests/test_intake.py`

- [ ] **Step 1: 写失败测试**

`tests/test_intake.py`:
```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_intake.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 intake**

`src/mockinterview/intake.py`:
```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_intake.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add intake.build_context"
```

---

### Task 6: planner.make_plan

**Files:**
- Create: `mock-interview/src/mockinterview/planner.py`
- Test: `mock-interview/tests/test_planner.py`

- [ ] **Step 1: 写失败测试**

`tests/test_planner.py`:
```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_planner.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 planner**

`src/mockinterview/planner.py`:
```python
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
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_planner.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add planner.make_plan"
```

---

### Task 7: 面试状态机（LangGraph）

本任务是项目核心。状态机回合驱动：每次 `step()` 处理一条候选人回答，产出下一个问题（或结束信号）。用一个显式的「待回答问题」槽位 + 追问预算实现追问/推进/结束的条件路由。

**Files:**
- Create: `mock-interview/src/mockinterview/interviewer/__init__.py`
- Create: `mock-interview/src/mockinterview/interviewer/state.py`
- Create: `mock-interview/src/mockinterview/interviewer/nodes.py`
- Create: `mock-interview/src/mockinterview/interviewer/graph.py`
- Test: `mock-interview/tests/test_interviewer.py`

- [ ] **Step 1: 写引擎的失败测试（最小用例）**

`tests/test_interviewer.py`（先放一个最小用例，后续步骤补全）:
```python
from mockinterview.interviewer import InterviewEngine


def test_engine_importable():
    assert InterviewEngine is not None
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_interviewer.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'mockinterview.interviewer'`

- [ ] **Step 3: 实现 state、nodes、graph，并补全测试**

`src/mockinterview/interviewer/__init__.py`:
```python
from .graph import InterviewEngine

__all__ = ["InterviewEngine"]
```

`src/mockinterview/interviewer/state.py`:
```python
from __future__ import annotations
from typing import TypedDict

from ..schemas import InterviewPlan, QATurn, AnswerEvaluation, StageType


class InterviewState(TypedDict, total=False):
    plan: InterviewPlan
    stage_index: int          # 当前阶段在 StageType.order() 的下标
    question_index: int       # 当前阶段内已问的「主问题」下标
    followups_used: int       # 当前问题已追问次数
    questions_in_stage: int   # 当前阶段已问问题总数（含追问）
    pending_question: str     # 已抛出、等待回答的问题
    last_answer: str          # 最近一次候选人回答
    last_evaluation: AnswerEvaluation
    history: list[QATurn]
    finished: bool


def new_state(plan: InterviewPlan) -> InterviewState:
    return {
        "plan": plan,
        "stage_index": 0,
        "question_index": 0,
        "followups_used": 0,
        "questions_in_stage": 0,
        "pending_question": "",
        "history": [],
        "finished": False,
    }


def current_stage(state: InterviewState) -> StageType:
    return StageType.order()[state["stage_index"]]
```

`src/mockinterview/interviewer/nodes.py`:
```python
from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient
from ..schemas import AnswerEvaluation, QATurn, StageType
from .state import InterviewState, current_stage

_ASK_SYSTEM = "你是面试官。根据面试计划与对话历史，提出下一个问题。只输出问题本身，简洁自然，不要解释。"
_EVAL_SYSTEM = "你是面试官，客观评估候选人回答。只输出符合 schema 的 JSON。"


def _stage_plan(state: InterviewState):
    stage = current_stage(state)
    for sp in state["plan"].stages:
        if sp.stage is stage:
            return sp
    return None


def make_ask_prompt(state: InterviewState, *, followup: bool) -> str:
    sp = _stage_plan(state)
    objective = sp.objective if sp else ""
    planned = ""
    if sp and state["question_index"] < len(sp.questions):
        planned = sp.questions[state["question_index"]].text
    history = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in state["history"][-4:])
    mode = "针对候选人上一回答追问、挤出细节" if followup else "提出本阶段的下一个主问题"
    return (
        f"当前阶段：{current_stage(state).value}\n阶段目标：{objective}\n"
        f"参考问题：{planned}\n近期对话：\n{history}\n\n任务：{mode}。"
    )


def ask(state: InterviewState, *, llm: LLMClient, followup: bool) -> str:
    prompt = make_ask_prompt(state, followup=followup)
    return "".join(llm.stream_text(prompt=prompt, system=_ASK_SYSTEM)).strip()


def evaluate(state: InterviewState, *, llm: LLMClient) -> AnswerEvaluation:
    prompt = (
        f"问题：{state['pending_question']}\n候选人回答：{state['last_answer']}\n\n"
        "评估该回答：score/relevance/depth(0-10)，watery 表示是否在打太极/答非所问，"
        "rationale 简述理由，若值得追问给出 suggested_followup，否则为 null。"
    )
    data = llm.structured(prompt=prompt, schema=AnswerEvaluation.model_json_schema(), system=_EVAL_SYSTEM)
    return AnswerEvaluation.model_validate(data)


def route(state: InterviewState, settings: Settings) -> str:
    """返回 'followup' | 'next_question' | 'next_stage' | 'finish'。"""
    ev = state["last_evaluation"]
    sp = _stage_plan(state)
    n_planned = len(sp.questions) if sp else 0

    should_dig = (ev.watery or ev.score < 6) and ev.suggested_followup is not None
    if should_dig and state["followups_used"] < settings.max_followups_per_question:
        return "followup"

    last_stage = state["stage_index"] >= len(StageType.order()) - 1
    more_questions = (state["question_index"] + 1 < n_planned
                      and state["questions_in_stage"] < settings.max_questions_per_stage)
    if more_questions:
        return "next_question"
    if last_stage:
        return "finish"
    return "next_stage"
```

`src/mockinterview/interviewer/graph.py`:
```python
from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient
from ..schemas import InterviewPlan, QATurn, StageType
from . import nodes
from .state import InterviewState, new_state, current_stage


class InterviewEngine:
    """回合制面试引擎：start() 给首问；submit_answer() 吃一条回答、给下一动作。

    说明：LangGraph 的图编译在 _build_graph 中提供（用于可观测/可视化），
    回合推进逻辑由 nodes.route 的条件路由实现，保持与图一致。
    """

    def __init__(self, plan: InterviewPlan, *, llm: LLMClient, settings: Settings):
        self._llm = llm
        self._settings = settings
        self.state: InterviewState = new_state(plan)

    def start(self) -> str:
        q = nodes.ask(self.state, llm=self._llm, followup=False)
        self.state["pending_question"] = q
        self.state["questions_in_stage"] = 1
        return q

    def submit_answer(self, answer: str) -> dict:
        """返回 {'done': bool, 'question': str|None, 'evaluation': AnswerEvaluation}。"""
        st = self.state
        st["last_answer"] = answer
        ev = nodes.evaluate(st, llm=self._llm)
        st["last_evaluation"] = ev
        st["history"].append(QATurn(
            stage=current_stage(st), question=st["pending_question"],
            answer=answer, evaluation=ev,
        ))

        decision = nodes.route(st, self._settings)
        if decision == "finish":
            st["finished"] = True
            st["pending_question"] = ""
            return {"done": True, "question": None, "evaluation": ev}

        if decision == "followup":
            st["followups_used"] += 1
            st["questions_in_stage"] += 1
            q = nodes.ask(st, llm=self._llm, followup=True)
        elif decision == "next_question":
            st["question_index"] += 1
            st["followups_used"] = 0
            st["questions_in_stage"] += 1
            q = nodes.ask(st, llm=self._llm, followup=False)
        else:  # next_stage
            st["stage_index"] += 1
            st["question_index"] = 0
            st["followups_used"] = 0
            st["questions_in_stage"] = 1
            q = nodes.ask(st, llm=self._llm, followup=False)

        st["pending_question"] = q
        return {"done": False, "question": q, "evaluation": ev}
```

把 `tests/test_interviewer.py` 改为真实测试：
```python
from mockinterview.config import Settings
from mockinterview.interviewer import InterviewEngine
from mockinterview.interviewer import nodes
from mockinterview.schemas import (
    InterviewPlan, StagePlan, PlannedQuestion, StageType,
)


def _plan():
    return InterviewPlan(stages=[
        StagePlan(stage=s, objective=f"{s.value} 目标",
                  questions=[PlannedQuestion(text=f"{s.value} 问题1"),
                             PlannedQuestion(text=f"{s.value} 问题2")])
        for s in StageType.order()
    ])


def _settings():
    return Settings(anthropic_api_key="sk-test",
                    max_followups_per_question=1, max_questions_per_stage=4)


def _good_eval():
    return {"score": 9, "relevance": 9, "depth": 8, "watery": False,
            "rationale": "很好", "suggested_followup": None}


def _watery_eval():
    return {"score": 3, "relevance": 4, "depth": 2, "watery": True,
            "rationale": "打太极", "suggested_followup": "能举个具体例子吗"}


def test_start_returns_first_question(fake_llm):
    fake_llm.queue_text("自我介绍一下")
    eng = InterviewEngine(_plan(), llm=fake_llm, settings=_settings())
    assert eng.start() == "自我介绍一下"
    assert eng.state["pending_question"] == "自我介绍一下"


def test_watery_answer_triggers_followup(fake_llm):
    fake_llm.queue_text("首问")           # start
    fake_llm.queue_structured(_watery_eval())  # evaluate
    fake_llm.queue_text("追问：具体点")    # followup ask
    eng = InterviewEngine(_plan(), llm=fake_llm, settings=_settings())
    eng.start()
    out = eng.submit_answer("呃我觉得还行")
    assert out["done"] is False
    assert out["question"] == "追问：具体点"
    assert eng.state["followups_used"] == 1


def test_followup_budget_capped(fake_llm):
    # max_followups_per_question=1：第二次仍水也不再追问，转下一题
    fake_llm.queue_text("首问")
    fake_llm.queue_structured(_watery_eval())
    fake_llm.queue_text("追问1")
    fake_llm.queue_structured(_watery_eval())
    fake_llm.queue_text("下一主问题")
    eng = InterviewEngine(_plan(), llm=fake_llm, settings=_settings())
    eng.start()
    eng.submit_answer("水")          # → followup
    out = eng.submit_answer("还是水")  # 预算用尽 → next_question
    assert out["question"] == "下一主问题"
    assert eng.state["followups_used"] == 0
    assert eng.state["question_index"] == 1


def test_good_answers_advance_through_stages_to_finish(fake_llm):
    eng = InterviewEngine(_plan(), llm=fake_llm, settings=_settings())
    fake_llm.queue_text("首问")
    eng.start()
    done = False
    guard = 0
    while not done and guard < 50:
        guard += 1
        fake_llm.queue_structured(_good_eval())
        fake_llm.queue_text(f"问题{guard}")
        out = eng.submit_answer("很完整的回答")
        done = out["done"]
    assert done is True
    assert eng.state["finished"] is True
    # 应至少覆盖到最后一个阶段
    assert eng.state["stage_index"] == len(StageType.order()) - 1


def test_route_unit_next_stage_when_questions_exhausted(fake_llm):
    from mockinterview.schemas import AnswerEvaluation
    eng = InterviewEngine(_plan(), llm=fake_llm, settings=_settings())
    eng.state["stage_index"] = 1               # tech 阶段（非最后阶段）
    eng.state["question_index"] = 1            # 已是本阶段最后一题
    eng.state["questions_in_stage"] = 2
    eng.state["last_evaluation"] = AnswerEvaluation(**_good_eval())
    assert nodes.route(eng.state, _settings()) == "next_stage"
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_interviewer.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 补充 LangGraph 图（可观测）并保证不破坏测试**

在 `graph.py` 末尾追加一个用于可视化/可观测的图构建函数（不改变 `InterviewEngine` 行为）：
```python
def build_graph():
    """构建 LangGraph 状态图，用于可视化与未来扩展。
    节点：ask → (await answer) → evaluate → route → {ask|END}。
    回合制下由 InterviewEngine 驱动；此图用于 langgraph 可视化与文档。"""
    from langgraph.graph import StateGraph, END
    from .state import InterviewState as _S

    g = StateGraph(_S)
    g.add_node("ask", lambda s: s)
    g.add_node("evaluate", lambda s: s)
    g.set_entry_point("ask")
    g.add_edge("ask", "evaluate")
    g.add_conditional_edges(
        "evaluate",
        lambda s: "finish" if s.get("finished") else "ask",
        {"ask": "ask", "finish": END},
    )
    return g.compile()
```
添加测试 `tests/test_interviewer.py`：
```python
def test_graph_compiles():
    from mockinterview.interviewer.graph import build_graph
    assert build_graph() is not None
```

- [ ] **Step 6: 运行全部面试测试确认通过**

Run: `uv run pytest tests/test_interviewer.py -v`
Expected: PASS（6 passed）

- [ ] **Step 7: 提交**

```bash
git add -A && git commit -m "feat: add LangGraph-backed interview state machine"
```

---

### Task 8: evaluator.make_report

**Files:**
- Create: `mock-interview/src/mockinterview/evaluator.py`
- Test: `mock-interview/tests/test_evaluator.py`

- [ ] **Step 1: 写失败测试**

`tests/test_evaluator.py`:
```python
from mockinterview.evaluator import make_report
from mockinterview.schemas import (
    InterviewReport, QATurn, AnswerEvaluation, StageType,
)


def _history():
    return [
        QATurn(stage=StageType.TECH, question="讲讲装饰器", answer="……",
               evaluation=AnswerEvaluation(score=8, relevance=8, depth=7,
                                           watery=False, rationale="不错")),
        QATurn(stage=StageType.PROJECT, question="项目难点", answer="……",
               evaluation=AnswerEvaluation(score=5, relevance=6, depth=4,
                                           watery=True, rationale="偏浅")),
    ]


def test_make_report(fake_llm):
    fake_llm.queue_structured({
        "overall_score": 72,
        "dimension_scores": {"技术深度": 70, "表达": 75},
        "strengths": ["基础扎实"],
        "weaknesses": ["项目深度不足"],
        "per_question_feedback": ["装饰器回答到位", "项目难点偏浅"],
        "improvement_advice": ["多准备项目细节"],
    })
    report = make_report(_history(), llm=fake_llm)
    assert isinstance(report, InterviewReport)
    assert report.overall_score == 72
    # 历史问答应进入 prompt
    assert "装饰器" in fake_llm.calls[0]["prompt"]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 evaluator**

`src/mockinterview/evaluator.py`:
```python
from __future__ import annotations

from .llm import LLMClient
from .schemas import InterviewReport, QATurn

_SYSTEM = "你是资深面试官，基于完整面试记录给出客观评估报告。只输出符合 schema 的 JSON。"


def _format_history(history: list[QATurn]) -> str:
    lines = []
    for i, t in enumerate(history, 1):
        ev = t.evaluation
        score = ev.score if ev else "NA"
        lines.append(f"[{i}][{t.stage.value}] Q: {t.question}\nA: {t.answer}\n评分: {score}")
    return "\n\n".join(lines)


def make_report(history: list[QATurn], *, llm: LLMClient) -> InterviewReport:
    prompt = (
        "以下是完整面试记录，请生成评估报告。\n\n"
        f"{_format_history(history)}\n\n"
        "要求：overall_score(0-100)；dimension_scores 给出技术深度/表达/项目匹配/行为面等维度；"
        "strengths/weaknesses/per_question_feedback/improvement_advice 具体可执行。"
    )
    data = llm.structured(prompt=prompt, schema=InterviewReport.model_json_schema(), system=_SYSTEM)
    return InterviewReport.model_validate(data)
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: PASS（1 passed）

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: add evaluator.make_report"
```

---

### Task 9: 会话存储 + FastAPI/SSE 端到端

**Files:**
- Create: `mock-interview/src/mockinterview/session_store.py`
- Create: `mock-interview/src/mockinterview/api.py`
- Test: `mock-interview/tests/test_api.py`

- [ ] **Step 1: 写会话存储 + API 的失败测试**

`tests/test_api.py`:
```python
import json
import pytest
from fastapi.testclient import TestClient

from mockinterview import api as api_module
from mockinterview.config import Settings


@pytest.fixture
def client(fake_llm, monkeypatch):
    # 注入 fake LLM 与测试配置
    monkeypatch.setattr(api_module, "_build_llm",
                        lambda settings: fake_llm)
    monkeypatch.setattr(api_module, "_load_settings",
                        lambda: Settings(anthropic_api_key="sk-test",
                                         max_followups_per_question=0,
                                         max_questions_per_stage=1))
    app = api_module.create_app()
    app.state.fake_llm = fake_llm
    return TestClient(app)


def test_create_session_returns_first_question(client):
    fake = client.app.state.fake_llm
    # intake、planner 各一次结构化；start 一次文本
    fake.queue_structured({
        "candidate": {"summary": "前端", "skills": [], "projects": [], "highlights": []},
        "job": {"title": "AI 应用工程师", "must_have": [], "tech_stack": [], "company_focus": ""},
    })
    fake.queue_structured({"stages": [
        {"stage": s, "objective": "o", "questions": [{"text": f"{s} q", "probes": []}]}
        for s in ["intro", "tech", "project", "behavioral", "candidate_q", "closing"]
    ]})
    fake.queue_text("请做个自我介绍")

    resp = client.post("/sessions", json={
        "resume_text": "我是前端", "jd_text": "招 AI 应用工程师", "company_text": "SaaS",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert body["question"] == "请做个自我介绍"


def test_answer_then_finish_then_report(client):
    fake = client.app.state.fake_llm
    fake.queue_structured({
        "candidate": {"summary": "前端", "skills": [], "projects": [], "highlights": []},
        "job": {"title": "X", "must_have": [], "tech_stack": [], "company_focus": ""},
    })
    fake.queue_structured({"stages": [
        {"stage": s, "objective": "o", "questions": [{"text": f"{s} q", "probes": []}]}
        for s in ["intro", "tech", "project", "behavioral", "candidate_q", "closing"]
    ]})
    fake.queue_text("Q-intro")
    sid = client.post("/sessions", json={
        "resume_text": "r", "jd_text": "j", "company_text": "c"}).json()["session_id"]

    # 用 good eval 一路推进到结束（每阶段 1 题，0 追问 → 每答一题进下一阶段）
    done = False
    guard = 0
    while not done and guard < 20:
        guard += 1
        fake.queue_structured({"score": 9, "relevance": 9, "depth": 9,
                               "watery": False, "rationale": "好", "suggested_followup": None})
        fake.queue_text(f"Q{guard}")
        out = client.post(f"/sessions/{sid}/answer", json={"answer": "完整回答"}).json()
        done = out["done"]
    assert done is True

    # 取报告
    fake.queue_structured({
        "overall_score": 80, "dimension_scores": {"技术深度": 80},
        "strengths": ["s"], "weaknesses": ["w"],
        "per_question_feedback": ["f"], "improvement_advice": ["a"],
    })
    report = client.get(f"/sessions/{sid}/report").json()
    assert report["overall_score"] == 80


def test_stream_question_endpoint(client):
    fake = client.app.state.fake_llm
    fake.queue_structured({
        "candidate": {"summary": "x", "skills": [], "projects": [], "highlights": []},
        "job": {"title": "X", "must_have": [], "tech_stack": [], "company_focus": ""},
    })
    fake.queue_structured({"stages": [
        {"stage": s, "objective": "o", "questions": [{"text": "q", "probes": []}]}
        for s in ["intro", "tech", "project", "behavioral", "candidate_q", "closing"]
    ]})
    fake.queue_text("流式问题内容")
    sid = client.post("/sessions", json={
        "resume_text": "r", "jd_text": "j", "company_text": "c"}).json()["session_id"]
    with client.stream("GET", f"/sessions/{sid}/current_question/stream") as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())
    assert "流式问题内容" in body
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'mockinterview.session_store'`（或 api）

- [ ] **Step 3: 实现 session_store**

`src/mockinterview/session_store.py`:
```python
from __future__ import annotations
import uuid

from .interviewer import InterviewEngine


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, InterviewEngine] = {}

    def create(self, engine: InterviewEngine) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = engine
        return sid

    def get(self, sid: str) -> InterviewEngine:
        if sid not in self._sessions:
            raise KeyError(sid)
        return self._sessions[sid]
```

- [ ] **Step 4: 实现 api**

`src/mockinterview/api.py`:
```python
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import Settings
from .interviewer import InterviewEngine
from .intake import build_context
from .planner import make_plan
from .evaluator import make_report
from .llm import AnthropicClient, LLMClient
from .session_store import SessionStore


class CreateSessionRequest(BaseModel):
    resume_text: str
    jd_text: str
    company_text: str


class AnswerRequest(BaseModel):
    answer: str


def _load_settings() -> Settings:
    return Settings()  # 从环境变量加载


def _build_llm(settings: Settings) -> LLMClient:
    return AnthropicClient(settings)


def create_app() -> FastAPI:
    app = FastAPI(title="AI Mock Interview")
    settings = _load_settings()
    llm = _build_llm(settings)
    store = SessionStore()

    @app.post("/sessions")
    def create_session(req: CreateSessionRequest):
        ctx = build_context(resume_text=req.resume_text, jd_text=req.jd_text,
                            company_text=req.company_text, llm=llm)
        plan = make_plan(ctx, llm=llm)
        engine = InterviewEngine(plan, llm=llm, settings=settings)
        question = engine.start()
        sid = store.create(engine)
        return {"session_id": sid, "question": question}

    @app.post("/sessions/{sid}/answer")
    def answer(sid: str, req: AnswerRequest):
        try:
            engine = store.get(sid)
        except KeyError:
            raise HTTPException(404, "session not found")
        out = engine.submit_answer(req.answer)
        return {"done": out["done"], "question": out["question"],
                "evaluation": out["evaluation"].model_dump()}

    @app.get("/sessions/{sid}/current_question/stream")
    def stream_current_question(sid: str):
        try:
            engine = store.get(sid)
        except KeyError:
            raise HTTPException(404, "session not found")
        text = engine.state.get("pending_question", "")

        def gen():
            for ch in text:
                yield f"data: {ch}\n\n"
            yield "event: done\ndata: \n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/sessions/{sid}/report")
    def report(sid: str):
        try:
            engine = store.get(sid)
        except KeyError:
            raise HTTPException(404, "session not found")
        rep = make_report(engine.state["history"], llm=llm)
        return rep.model_dump()

    return app


app = create_app()
```

注：测试通过 monkeypatch 替换 `_build_llm` / `_load_settings`，因此 `create_app()` 内部对这两个函数的调用会拿到 fake。务必在 `create_app` 内部以 `_build_llm(settings)` / `_load_settings()` 形式调用（已如上）。

- [ ] **Step 5: 运行确认通过**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS（3 passed）

- [ ] **Step 6: 跑全量测试**

Run: `uv run pytest -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add -A && git commit -m "feat: add session store and FastAPI/SSE endpoints"
```

---

### Task 10: 手动冒烟（真实 LLM，可选但推荐）

**Files:**
- Create: `mock-interview/README.md`

- [ ] **Step 1: 写 README 运行说明**

`mock-interview/README.md` 至少包含：
```markdown
# AI 模拟面试 Agent（后端核心）

## 运行
1. cp .env.example .env 并填入 MOCKINTERVIEW_ANTHROPIC_API_KEY
2. uv run uvicorn mockinterview.api:app --reload

## 试一场（curl）
\`\`\`bash
SID=$(curl -s localhost:8000/sessions -H 'content-type: application/json' \
  -d '{"resume_text":"我是前端，做过AI Code Review工具","jd_text":"招AI应用工程师，要求Python/LangGraph","company_text":"企业SaaS"}' | python -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
curl -s localhost:8000/sessions/$SID/answer -H 'content-type: application/json' -d '{"answer":"我有三年前端经验……"}'
curl -s localhost:8000/sessions/$SID/report
\`\`\`
```

- [ ] **Step 2: 真实跑一次（需 API key）**

Run: `uv run uvicorn mockinterview.api:app --port 8000`，另开终端执行 README 里的 curl。
Expected: 创建会话返回首个问题；多次 answer 后 `done:true`；report 返回 0-100 评分与各维度。
若无 key 可跳过此步，单测已覆盖逻辑闭环。

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "docs: add README and smoke-test instructions"
```

---

## 后续计划（不在本计划范围）

- **计划二（语音层）**：`voice/stt_faster_whisper.py` + `voice/tts_edge.py` 实现 `STT`/`TTS` 接口；API 增加音频上传转写、问题转语音端点。
- **计划三（前端）**：Next.js + TS 工作台，录音、SSE 流式渲染、阶段进度、报告可视化。
- **可观测**：接入 Langfuse trace `llm.py` 的每次调用。
