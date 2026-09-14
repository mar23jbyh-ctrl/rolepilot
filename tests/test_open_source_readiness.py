"""Regression cases for public startup, stable references and score reproduction."""
import importlib
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import DEFAULT_GRADE_CUTOFFS, settings


@pytest.mark.parametrize("value", ["nan,6,4", "inf,6,4", "11,6,4", "8,6,-1", "4,6,8",
                                  "8,8,4", "8,6,4,2", "8,broken,6,4", "8,,6,4"])
def test_invalid_grade_cutoffs_use_safe_defaults(monkeypatch, value):
    monkeypatch.setattr(settings, "grade_thresholds", value)
    assert settings.grade_cutoffs == DEFAULT_GRADE_CUTOFFS


def test_missing_llm_key_fails_before_external_call(monkeypatch):
    module = importlib.import_module("app.llm.client")
    monkeypatch.setattr(module.client, "api_key", "")
    calls = []
    monkeypatch.setattr(module.client.chat.completions, "create", lambda **kw: calls.append(kw))
    with pytest.raises(ValueError, match="llm_credentials_missing"):
        module._create_with_retry(model="synthetic", messages=[])
    assert calls == []


def test_rename_hides_storage_error_and_preserves_service_status(monkeypatch):
    from api.routers.sessions import rename_session
    from api.schemas import RenameSessionRequest
    from app.errors import ServiceError

    class FailingService:
        def rename_session(self, *args):
            raise self.error

    service = FailingService()
    for error, status, code in [(RuntimeError("synthetic-private-path-and-secret"), 503, "operation_failed"),
                                (ServiceError("session_busy"), 409, "session_busy")]:
        service.error = error
        with pytest.raises(HTTPException) as failure:
            rename_session("synthetic", RenameSessionRequest(name="面试"), service)
        assert (failure.value.status_code, failure.value.detail) == (status, code)


def test_built_spa_never_masks_unknown_api_routes():
    from api.main import app
    with TestClient(app) as client:
        response = client.get("/api/unknown-synthetic-route")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_ranked_project_prompt_and_guard_preserve_resume_indices(monkeypatch):
    module = importlib.import_module("app.nodes.plan")
    from app.nodes.ask import _project_note
    projects = ["2020：官网页面", "2026：Python 异步任务平台", "2021：表格整理"]
    state = {"resume_profile": {"projects": projects}, "jd_text": "Python 开发",
             "jd_profile": {"requirements": [{"text": "Python", "skills": ["Python"]}]},
             "role_profile": {"job_title": "开发工程师"}}
    seen = {}

    def generate(messages, budget):
        seen["prompt"] = messages[1].content
        budget.used = budget.limit
        return [{"content": "请讲讲 Python 异步任务平台的设计", "project_ref": "1", "category": "project"}], {}

    def audit(questions, profile, **kwargs):
        seen["projects"] = kwargs["resume_projects"]
        return {"kept": questions, "dropped": [], "audit": [], "status": "ok", "summary": {}, "drop_rate": 0}

    monkeypatch.setattr(module, "_generate_questions", generate)
    monkeypatch.setattr(module, "audit_questions", audit)
    result = module.plan(state)
    assert seen["prompt"].index("[1] " + projects[1]) < seen["prompt"].index("[2] " + projects[2])
    assert seen["projects"] == projects
    assert result["question_plan"][0]["project_ref"] == "1"
    assert projects[1] in _project_note(state, result["question_plan"][0])


def test_resume_conflict_caps_levels_and_reproducible_scores(monkeypatch):
    module = importlib.import_module("app.nodes.assess")
    answer = "我设计了任务队列，采用重试与死信机制"
    payload = {"scoreable": True, "dimension_levels": {"d1": 5}, "evidence": {"d1": ["任务队列"]},
               "missing_points": [], "hallucination_or_conflict": False,
               "next_action": "next_question", "confidence": "high"}
    monkeypatch.setattr(module, "_chat_for_assessment", lambda messages: (json.dumps(payload), payload, {}))
    monkeypatch.setattr(module, "validate_resume_claims", lambda **kw: {"conflict": True, "claims": []})
    state = {"assessment_protocol_version": "practice-v1", "current_question": "讲讲任务队列",
             "current_answer": answer, "question_plan": [{"category": "project", "focus_key": "f1"}],
             "role_profile": {"rubric": [{"key": "d1", "weight": 1.0}]}}
    assessment = module.assess(state)["assessments"][0]
    assert assessment["hallucination_or_conflict"] is True
    assert assessment["dimension_levels"] == {"d1": 2}
    assert assessment["dimensions"] == {"d1": 4.0}
    assert assessment["score"] == 4.0


def test_resume_validation_uses_complete_summary_instead_of_characters():
    from app.validation import _resume_project_corpus
    assert _resume_project_corpus({"summary": "负责业务数据平台", "projects": ["任务队列"]}) == [
        "任务队列", "负责业务数据平台"]


@pytest.mark.parametrize("payload", [
    {"conflict": "false", "claims": [{"in_resume": False}]},
    {"conflict": True, "claims": [{"in_resume": "false"}]},
    {"conflict": True, "claims": [{"name": "未给归属"}]},
])
def test_invalid_resume_claim_flags_do_not_create_conflicts(monkeypatch, payload):
    from app import validation
    monkeypatch.setattr(validation, "chat_with_usage", lambda *args, **kw: (json.dumps(payload), {}))
    result = validation.validate_resume_claims("我设计了一个业务任务队列", {}, "project")
    assert result["conflict"] is False
