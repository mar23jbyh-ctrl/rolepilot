from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from uuid import UUID

from app.service import InterviewService
from app.errors import ServiceError

from api.deps import service_dep
from api.schemas import (
    AnswerRequest,
    RenameSessionRequest,
    SessionOut,
    SessionSummaryOut,
    StartSessionRequest,
    UpdateJobTitleRequest,
)
from api.serializers import session_payload

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _public_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ServiceError):
        return HTTPException(status_code=exc.status, detail=exc.code)
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail="invalid_request")
    return HTTPException(status_code=503, detail="operation_failed")


def _answer_response(result: dict, service: InterviewService, session_id: str):
    if "state" not in result:
        return JSONResponse(status_code=202 if result.get("processing") else 409,
                            content=result, headers={"Retry-After": "1"})
    record = service.store.get(session_id) or {}
    payload = session_payload(result["state"], session_id, record.get("name", ""))
    payload["replayed"] = bool(result.get("replayed"))
    return SessionOut(**payload)


@router.post("", response_model=SessionOut)
def start_session(
    body: StartSessionRequest,
    service: InterviewService = Depends(service_dep),
) -> SessionOut:
    try:
        result = service.start_session(
            jd_text=body.jd_text,
            resume_text=body.resume_text,
            session_name=body.session_name,
        )
    except ValueError as exc:
        status = 409 if "已存在" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _public_error(exc) from exc
    return SessionOut(
        **session_payload(result["state"], result["session_id"], body.session_name)
    )


@router.post("/{session_id}/answer", response_model=SessionOut)
def submit_answer(
    session_id: str,
    body: AnswerRequest,
    service: InterviewService = Depends(service_dep),
) -> SessionOut:
    try:
        result = service.submit_answer(session_id, body.answer, str(body.answer_request_id),
                                       body.expected_question_version)
    except Exception as exc:  # noqa: BLE001
        raise _public_error(exc) from exc
    return _answer_response(result, service, session_id)


@router.get("/{session_id}/answer-requests/{request_id}", response_model=SessionOut)
def answer_request_status(session_id: str, request_id: UUID,
                          service: InterviewService = Depends(service_dep)):
    try:
        result = service.answer_request_status(session_id, str(request_id))
        return _answer_response(result, service, session_id)
    except Exception as exc:
        raise _public_error(exc) from exc


@router.patch("/{session_id}/job-title", response_model=SessionOut)
def update_job_title(
    session_id: str,
    body: UpdateJobTitleRequest,
    service: InterviewService = Depends(service_dep),
) -> SessionOut:
    """P1/A1：修正岗位名；未答题时可选连带重新出题。"""

    try:
        result = service.update_job_title(
            session_id, body.job_title, regenerate=body.regenerate
        )
    except Exception as exc:  # noqa: BLE001
        raise _public_error(exc) from exc
    record = service.store.get(session_id) or {}
    return SessionOut(**session_payload(result["state"], session_id, record.get("name", "")))


@router.post("/{session_id}/stop", response_model=SessionOut)
def stop_session(
    session_id: str,
    service: InterviewService = Depends(service_dep),
) -> SessionOut:
    try:
        result = service.stop_session(session_id)
    except Exception as exc:  # noqa: BLE001
        raise _public_error(exc) from exc
    record = service.store.get(session_id) or {}
    return SessionOut(**session_payload(result["state"], session_id, record.get("name", "")))


@router.get("", response_model=list[SessionSummaryOut])
def list_sessions(
    service: InterviewService = Depends(service_dep),
) -> list[SessionSummaryOut]:
    records = service.store.list_sessions()
    return [
        SessionSummaryOut(
            id=item["id"],
            name=item.get("name") or "",
            status=item.get("status") or "",
            total_tokens=_list_tokens(item),
            created_at=item.get("created_at") or "",
            updated_at=item.get("updated_at") or "",
        )
        for item in records
    ]


def _list_tokens(item):
    import json
    from app.telemetry.summary import summarize_usage, state_usage_records
    try:
        state = json.loads(item.get('state_json') or '{}')
    except (ValueError, TypeError):
        return None
    if state.get('usage_schema_version') == 'usage-v1':
        return summarize_usage(state_usage_records(state))['total']
    return int(item.get('total_tokens') or 0)


@router.get("/{session_id}", response_model=SessionOut)
def get_session(
    session_id: str,
    service: InterviewService = Depends(service_dep),
) -> SessionOut:
    try:
        result = service.restore(session_id)
    except Exception as exc:  # noqa: BLE001
        raise _public_error(exc) from exc
    record = service.store.get(session_id) or {}
    return SessionOut(**session_payload(result["state"], session_id, record.get("name", "")))


@router.delete("/{session_id}")
def delete_session(
    session_id: str,
    service: InterviewService = Depends(service_dep),
) -> dict[str, bool]:
    try:
        deleted = service.delete_session(session_id)
    except Exception as exc:  # noqa: BLE001
        raise _public_error(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


@router.patch("/{session_id}", response_model=SessionSummaryOut)
def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    service: InterviewService = Depends(service_dep),
) -> SessionSummaryOut:
    try:
        service.rename_session(session_id, body.name)
    except ValueError as exc:
        status = 409 if "已存在" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _public_error(exc) from exc
    record = service.store.get(session_id) or {}
    return SessionSummaryOut(
        id=session_id,
        name=record.get("name") or "",
        status=record.get("status") or "",
        total_tokens=int(record.get("total_tokens") or 0),
        created_at=record.get("created_at") or "",
        updated_at=record.get("updated_at") or "",
    )
