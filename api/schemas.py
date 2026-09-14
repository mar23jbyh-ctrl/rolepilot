from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    jd_text: str = ""
    resume_text: str = ""
    session_name: str = ""


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=12000)
    answer_request_id: UUID
    expected_question_version: int = Field(ge=1)


class RenameSessionRequest(BaseModel):
    name: str = ""


class UpdateJobTitleRequest(BaseModel):
    """P1/A1：用户修正系统识别到的岗位名。"""

    job_title: str = ""
    regenerate: bool = True


class MessageOut(BaseModel):
    role: str
    content: str


class SessionOut(BaseModel):
    session_id: str
    name: str = ""
    status: str = "interviewing"
    done: bool = False
    question: str = ""
    question_version: int = 0
    replayed: bool = False
    messages: list[MessageOut] = Field(default_factory=list)
    report: dict[str, Any] | None = None
    usage_total: int | None = 0
    usage_summary: dict[str, Any] = Field(default_factory=dict)
    # ---- M23：岗位与题单可见性（原先用户看不到系统识别的岗位与题量）----
    job_title: str = ""
    domain: str = ""
    industry: str = ""
    plan_question_count: int = 0
    effective_sample_count: int = 0


class SessionSummaryOut(BaseModel):
    id: str
    name: str = ""
    status: str = ""
    total_tokens: int | None = 0
    created_at: str = ""
    updated_at: str = ""


class UploadOut(BaseModel):
    text: str
    method: str = "text"
    file_name: str = ""
