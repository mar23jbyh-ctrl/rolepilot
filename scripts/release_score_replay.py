"""Opt-in replay of failed motivation assessments from selected local sessions.

Default is prepared: no app imports, database access, files written or real calls.
Provide --target OWNER_ID:SESSION_ID for each session, then add --execute to
permit real model calls. Only the requested state_json is read over read-only
DB URIs. Reports may contain private Q/A and remain Git-ignored. This is a
targeted protocol replay, not a full E2E run or scoring-quality certification.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "docs" / "release" / "evidence"
TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")
SOURCE_FILES = (
    "scripts/release_score_replay.py", "app/nodes/assess.py", "app/llm/client.py",
    "app/prompts/templates.py", "app/validation.py", "app/role.py",
    "app/utils/numeric.py", "app/utils/json_utils.py", "app/models/schemas.py", "app/config.py",
)


class ReplayError(Exception):
    """Fixed non-sensitive stage code only; never provider exception text."""


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Permit real scoring calls and evidence writes")
    parser.add_argument("--target", action="append", default=[], help="OWNER_ID:SESSION_ID; 32 lowercase hex characters each")
    parser.add_argument("--output", default=str(EVIDENCE_ROOT / "motivation-replay.json"))
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[a-f0-9]{32}:[a-f0-9]{32}", target) for target in args.target):
        parser.error("target must be OWNER_ID:SESSION_ID with 32 lowercase hex characters each")
    args.target = list(dict.fromkeys(args.target))
    if len(args.target) > 10 or args.execute and not args.target:
        parser.error("execute requires 1 to 10 explicit targets")
    path = Path(args.output)
    args.output = (ROOT / path if not path.is_absolute() else path).resolve()
    if not args.output.is_relative_to(EVIDENCE_ROOT.resolve()) or args.output.suffix.lower() != ".json":
        parser.error("output must be a JSON file inside docs/release/evidence")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_session_state(path, session_id):
    # Do not instantiate SessionStore: its constructor performs migrations/cleanup.
    if not path.is_file():
        raise ReplayError("target_database_missing")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        row = connection.execute(
            "SELECT state_json FROM sessions WHERE id = ? LIMIT 1", (session_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ReplayError("target_session_missing")
    try:
        state = json.loads(row[0])
    except (TypeError, ValueError):
        raise ReplayError("target_state_invalid_json") from None
    if not isinstance(state, dict):
        raise ReplayError("target_state_not_object")
    return state


def integer(value, code):
    if isinstance(value, bool):
        raise ReplayError(code)
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ReplayError(code) from None
    if isinstance(value, float) and parsed != value:
        raise ReplayError(code)
    return parsed


def is_motivation(meta):
    return str(meta.get("question_kind") or "") == "motivation" or (
        not str(meta.get("focus_key") or "").strip()
        and str(meta.get("question_type") or "") == "behavioral"
        and not any(str(meta.get(field) or "").strip() for field in ("jd_ref", "project_ref", "research_ref"))
    )


def replay_inputs(state):
    plan, assessments, turns = (state.get(key) or [] for key in ("question_plan", "assessments", "turn_records"))
    role_profile = state.get("role_profile")
    if not isinstance(role_profile, dict) or not isinstance(role_profile.get("rubric"), list):
        raise ReplayError("original_professional_role_profile_missing")
    professional_keys = {str(row.get("key")) for row in role_profile["rubric"] if isinstance(row, dict) and row.get("key")}
    if not professional_keys or professional_keys <= {"m1", "m2"}:
        raise ReplayError("original_rubric_not_professional")
    inputs = []
    for position, before in enumerate(assessments):
        if not isinstance(before, dict) or not before.get("score_error"):
            continue
        index = integer(before.get("plan_question_index"), "failed_question_plan_index_missing")
        if not 0 <= index < len(plan) or not isinstance(plan[index], dict):
            raise ReplayError("failed_question_plan_index_invalid")
        if not is_motivation(plan[index]):
            continue
        question_id = integer(before.get("question_id"), "failed_question_id_missing")
        matching = [turn for turn in turns if isinstance(turn, dict) and str(turn.get("question_id")) == str(question_id)]
        if len(matching) != 1:
            raise ReplayError("original_turn_missing_or_ambiguous")
        original_turn = matching[0]
        question, answer = original_turn.get("question"), original_turn.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise ReplayError("original_question_answer_missing")
        if before.get("score") is not None:
            raise ReplayError("before_error_has_non_null_score")
        preceding = copy.deepcopy(assessments[:position])
        if question_id < 1 or len(preceding) > question_id - 1:
            raise ReplayError("historical_counter_cannot_preserve_question_id")
        follow_count = 0
        if before.get("is_follow_up"):
            follow_count = 1
            for previous in reversed(preceding):
                if not previous.get("is_follow_up") or previous.get("plan_question_index") != index:
                    break
                follow_count += 1
        previous_ids = {str(row.get("question_id")) for row in preceding if isinstance(row, dict)}
        previous_turns = copy.deepcopy([turn for turn in turns if isinstance(turn, dict) and str(turn.get("question_id")) in previous_ids])
        history = []
        for turn in previous_turns:
            history.extend([{"role": "assistant", "content": str(turn.get("question") or "")},
                            {"role": "user", "content": str(turn.get("answer") or "")}])
        replay = {
            "question_plan": copy.deepcopy(plan), "current_question_index": index,
            "current_question": question, "current_answer": answer,
            "global_question_counter": question_id - 1, "follow_up_count": follow_count,
            "total_follow_ups": sum(bool(row.get("is_follow_up")) for row in preceding if isinstance(row, dict)),
            "max_follow_ups": state.get("max_follow_ups", 3),
            "max_total_follow_ups": state.get("max_total_follow_ups", 12),
            "difficulty": before.get("difficulty") or plan[index].get("difficulty") or "medium",
            "role_profile": copy.deepcopy(role_profile),
            "resume_profile": copy.deepcopy(state.get("resume_profile") or {}),
            "assessments": preceding, "turn_records": previous_turns, "conversation_history": history,
            "usage_records": [], "end_requested": False, "explanation_pending": False,
        }
        inputs.append({"question_id": question_id, "plan_question_index": index,
                       "before": copy.deepcopy(before), "input": replay})
    if not 1 <= len(inputs) <= 20:
        raise ReplayError("expected_one_to_twenty_failed_motivation_questions_per_session")
    return inputs


def usage_total(records):
    return {key: sum(int((row or {}).get(key, 0) or 0) for row in records) for key in TOKEN_KEYS}


def model_name(value):
    text = str(value or "")
    return text if re.fullmatch(r"[A-Za-z0-9._:/-]{1,120}", text) and "://" not in text else "configured_or_unavailable"


def safe_assessment(value):
    # Evidence projection only; original function result is never modified.
    return {key: copy.deepcopy(value.get(key)) for key in (
        "question_id", "plan_question_index", "question_kind", "score", "score_error",
        "score_error_status", "score_fallback", "dimensions", "dimensions_not_covered",
        "dimension_levels", "candidate_intent", "is_follow_up", "next_action", "resume_conflict",
    ) if key in value}


def execute(args):
    if args.output.exists():
        raise ReplayError("output_exists_choose_another_output")
    evidence = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Selected original-Q/A motivation scoring replay",
                "input_material": "User-selected local sessions; may contain private data", "full_e2e_rerun": False, "old_database_updates": False,
                "quality_limit": "Selected failed assessments; does not independently establish scoring accuracy or full workflow correctness.",
                "database_read": "Read-only URI mode=ro; SELECT state_json FROM sessions WHERE id=? LIMIT 1",
                "cases": [], "sdk_attempts": [], "logical_llm_calls": [], "databases": [],
                "source_sha256_before": {}, "source_sha256_after": {}, "billing_verified": False,
                "provider_billed_cost": None}
    prepared = []
    for number, target in enumerate(args.target, 1):
        owner, sid = target.split(":")
        label = f"selected_session_{number}"
        path = ROOT / "data" / f"sessions_{owner}.db"
        item = {"label": label, "owner_id": owner, "session_id": sid, "database_path": str(path),
                "sha256_before": sha256(path) if path.is_file() else None}
        evidence["databases"].append(item)
        wal_path = Path(str(path) + "-wal")
        item["wal_sha256_before"] = sha256(wal_path) if wal_path.is_file() else None
        state = read_session_state(path, sid)
        for original in replay_inputs(state):
            prepared.append({"label": label, "session_id": sid, **original})
    evidence["before_failed_questions"] = len(prepared)
    evidence["source_sha256_before"] = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open("x", encoding="utf-8")
    secrets = set()
    start = time.perf_counter()
    fatal_code = None
    restorations = []

    def scrub(value):
        if isinstance(value, dict):
            return {str(key): "[REDACTED]" if any(word in str(key).lower() for word in (
                "authorization", "password", "api_key", "access_token", "refresh_token", "cookie"
            )) or str(key).lower() == "token" else scrub(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [scrub(item) for item in value]
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[REDACTED]")
            return re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", value)
        return value

    try:
        sys.path.insert(0, str(ROOT))
        from app.llm import client as lc
        from app.nodes import assess
        from app.config import settings
        from app import validation
        from dotenv import dotenv_values
        secrets.update(str(value) for key, value in dotenv_values(ROOT / ".env").items()
                       if value and any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET")))
        for secret in (getattr(lc, "api_key", ""), settings.api_key):
            if secret:
                secrets.add(str(secret))
        evidence["assessment_function"] = {"file": "app/nodes/assess.py", "function": "assess", "line": inspect.getsourcelines(assess.assess)[1]}
        # Targeted replays do not certify provider pricing or cache discounts.
        evidence["estimated_cost"] = None
        current = {}
        original_create = lc.client.chat.completions.create

        @functools.wraps(original_create)
        def create(*call_args, **kwargs):
            attempt = {**current, "attempt_number": len(evidence["sdk_attempts"]) + 1,
                       "requested_model": model_name(kwargs.get("model"))}
            started = time.perf_counter()
            try:
                response = original_create(*call_args, **kwargs)
                attempt.update(success=True, actual_model=model_name(getattr(response, "model", None)),
                               usage=lc._usage_dict(response), provider_usage_present=getattr(response, "usage", None) is not None)
                return response
            except Exception as exc:
                attempt.update(success=False, error_type=type(exc).__name__, usage=None)
                raise
            finally:
                attempt["seconds"] = round(time.perf_counter() - started, 6)
                evidence["sdk_attempts"].append(attempt)
        restorations.append((lc.client.chat.completions, "create", original_create))
        lc.client.chat.completions.create = create
        original_chat = lc.chat_with_usage

        @functools.wraps(original_chat)
        def chat(*call_args, **kwargs):
            started = time.perf_counter()
            row = {**current, "logical_call_number": len(evidence["logical_llm_calls"]) + 1,
                   "requested_model": model_name(settings.model)}
            try:
                result = original_chat(*call_args, **kwargs)
                row.update(success=True, usage=copy.deepcopy(result[1]))
                return result
            except Exception as exc:
                row.update(success=False, error_type=type(exc).__name__, usage=None)
                raise
            finally:
                row["seconds"] = round(time.perf_counter() - started, 6)
                evidence["logical_llm_calls"].append(row)
        for module in (lc, assess, validation):
            if getattr(module, "chat_with_usage", None) is original_chat:
                restorations.append((module, "chat_with_usage", original_chat))
                module.chat_with_usage = chat

        for number, original in enumerate(prepared, 1):
            current.clear()
            current.update(label=original["label"], session_id=original["session_id"], original_question_id=original["question_id"])
            replay = original["input"]
            row = {**current, "plan_question_index": original["plan_question_index"],
                   "question": replay["current_question"], "answer": replay["current_answer"],
                   "before": safe_assessment(original["before"]),
                   "input_control": {key: replay[key] for key in (
                       "current_question_index", "global_question_counter", "follow_up_count", "total_follow_ups", "difficulty")},
                   "original_role_profile_unchanged": True,
                   "original_professional_rubric_keys": sorted(str(item["key"]) for item in replay["role_profile"]["rubric"] if isinstance(item, dict) and item.get("key"))}
            print(f"[Replay] {number}/{len(prepared)} {original['label']} question_id={original['question_id']}: scoring", flush=True)
            started = time.perf_counter()
            sdk_start = len(evidence["sdk_attempts"])
            profile_before = copy.deepcopy(replay["role_profile"])
            try:
                result = assess.assess(replay)  # Real original function; no graph, service or DB writer.
                records = result.get("assessments") or []
                if len(records) != 1 or not isinstance(records[0], dict):
                    raise ReplayError("assessment_result_missing")
                after = records[0]
                output_usage = [{key: record.get(key) for key in ("node", *TOKEN_KEYS)}
                                for record in (result.get("usage_records") or []) if isinstance(record, dict)]
                attempts = evidence["sdk_attempts"][sdk_start:]
                sdk_usage = usage_total([attempt["usage"] for attempt in attempts if attempt.get("success")])
                output_total = usage_total(output_usage)
                score = after.get("score")
                row.update(after=safe_assessment(after), output_usage_records=output_usage,
                           sdk_usage=sdk_usage, output_usage=output_total,
                           usage_matches=sdk_usage == output_total,
                           original_role_profile_unchanged=replay["role_profile"] == profile_before,
                           sdk_attempt_count=len(attempts),
                           passed=isinstance(score, (int, float)) and not isinstance(score, bool)
                           and math.isfinite(score) and 0 <= score <= 10 and not after.get("score_error")
                           and after.get("question_kind") == "motivation"
                           and set(after.get("dimensions") or {}) == {"m1", "m2"}
                           and after.get("question_id") == original["question_id"]
                           and sdk_usage == output_total and replay["role_profile"] == profile_before
                           and any(attempt.get("success") and attempt.get("provider_usage_present") for attempt in attempts))
            except Exception as exc:
                row.update(passed=False, error_type=type(exc).__name__,
                           failure=str(exc) if isinstance(exc, ReplayError) else "assessment_call_failed")
            row["seconds"] = round(time.perf_counter() - started, 6)
            evidence["cases"].append(row)
            print(f"[Replay] {number}/{len(prepared)}: passed={row['passed']}, {row['seconds']:.2f}s", flush=True)
        evidence["sdk_usage_total"] = usage_total([r["usage"] for r in evidence["sdk_attempts"] if r.get("success")])
        evidence["output_usage_total"] = usage_total([r.get("output_usage", {}) for r in evidence["cases"]])
        evidence["total_usage_matches"] = evidence["sdk_usage_total"] == evidence["output_usage_total"]
        evidence["unknown_billing_failed_sdk_attempts"] = sum(not r.get("success") for r in evidence["sdk_attempts"])
        evidence["actual_models"] = sorted({r["actual_model"] for r in evidence["sdk_attempts"] if r.get("actual_model")})
        evidence["sdk_attempt_scope"] = "Each Python completions.create invocation; SDK-internal transport retries are not independently counted"
    except Exception as exc:
        fatal_code = str(exc) if isinstance(exc, ReplayError) else "replay_setup_failed"
        evidence.update(failure=fatal_code, error_type=type(exc).__name__)
    finally:
        for module, name, original in reversed(restorations):
            setattr(module, name, original)
        evidence["seconds"] = round(time.perf_counter() - start, 6)
        for item in evidence["databases"]:
            try:
                item["sha256_after"] = sha256(Path(item["database_path"]))
                item["unchanged"] = item["sha256_before"] == item["sha256_after"]
                wal_path = Path(item["database_path"] + "-wal")
                item["wal_sha256_after"] = sha256(wal_path) if wal_path.is_file() else None
                item["wal_unchanged"] = item["wal_sha256_before"] == item["wal_sha256_after"]
            except OSError:
                item.update(sha256_after=None, unchanged=False, wal_unchanged=False)
        for name in SOURCE_FILES:
            try:
                evidence["source_sha256_after"][name] = sha256(ROOT / name)
            except OSError:
                evidence["source_sha256_after"][name] = None
        evidence["source_unchanged_during_replay"] = evidence["source_sha256_before"] == evidence["source_sha256_after"]
        evidence["databases_unchanged"] = all(item["unchanged"] and item["wal_unchanged"] for item in evidence["databases"])
        evidence["database_hash_scope"] = "Database and WAL content; transient SQLite shared-memory lock state is not hashed"
        evidence["after_passed_questions"] = sum(bool(row.get("passed")) for row in evidence["cases"])
        evidence["after_failed_questions"] = sum(bool((row.get("after") or {}).get("score_error")) for row in evidence["cases"])
        evidence["after_call_error_or_failed_acceptance"] = sum(not row.get("passed") for row in evidence["cases"])
        evidence["all_passed"] = not fatal_code and len(evidence["cases"]) == len(prepared) and evidence["after_passed_questions"] == len(prepared) \
            and evidence.get("total_usage_matches") is True and evidence["databases_unchanged"] \
            and evidence["source_unchanged_during_replay"]
        try:
            json.dump(scrub(evidence), output, ensure_ascii=False, indent=2, allow_nan=False)
        finally:
            output.close()
    if not evidence["all_passed"]:
        raise ReplayError(fatal_code or "targeted_replay_acceptance_failed")
    print(f"[Replay] before_failed={len(prepared)}; after_passed={evidence['after_passed_questions']}; evidence={args.output}", flush=True)
    return 0


def main(argv=None):
    args = arguments(argv)
    if not args.execute:
        print(f"Prepared only: {len(args.target)} selected sessions; no DB access or real calls.")
        print("Provide --target OWNER_ID:SESSION_ID, review, then add --execute. Evidence: " + str(args.output))
        return 0
    try:
        return execute(args)
    except ReplayError:
        raise
    except Exception:
        raise ReplayError("targeted_replay_preparation_or_evidence_failed") from None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as exc:
        raise SystemExit("Scoring replay: " + str(exc)) from None
