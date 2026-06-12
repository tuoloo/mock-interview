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
