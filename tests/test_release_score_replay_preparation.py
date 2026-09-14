"""Offline D12 harness checks: synthetic temporary states only, no model calls."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def replay():
    spec = importlib.util.spec_from_file_location("offline_score_replay", ROOT / "scripts" / "release_score_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_production_imports_at_top_level():
    tree = ast.parse((ROOT / "scripts" / "release_score_replay.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("app", "api", "openai", "dotenv"))
        elif isinstance(node, ast.Import):
            assert not any(alias.name.startswith(("app", "api", "openai", "dotenv")) for alias in node.names)


def test_prepared_mode_never_reads_databases_or_executes(replay, monkeypatch, capsys):
    def prohibited(*args, **kwargs):
        raise AssertionError("Prepared mode must not execute")
    monkeypatch.setattr(replay, "execute", prohibited)
    monkeypatch.setattr(replay, "read_session_state", prohibited)
    assert replay.main([]) == 0
    assert replay.arguments([]).execute is False
    assert "no DB access or real calls" in capsys.readouterr().out


def test_only_requested_session_is_read_and_db_hash_unchanged(replay, tmp_path):
    path = tmp_path / "synthetic.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, state_json TEXT)")
        connection.executemany("INSERT INTO sessions VALUES (?, ?)", [
            ("wanted", json.dumps({"fixture": "wanted"})), ("other", json.dumps({"fixture": "not_read"})),
        ])
    before = replay.sha256(path)
    assert replay.read_session_state(path, "wanted") == {"fixture": "wanted"}
    assert replay.sha256(path) == before
    with pytest.raises(replay.ReplayError, match="target_session_missing"):
        replay.read_session_state(path, "absent")


def test_read_only_uri_and_parameterized_select(replay, tmp_path, monkeypatch):
    path = tmp_path / "exists.db"
    path.touch()
    calls = []
    class Connection:
        def execute(self, sql, values):
            calls.append((sql, values))
            return SimpleNamespace(fetchone=lambda: ("{}",))
        def close(self):
            pass
    def connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro")
        assert kwargs["uri"] is True
        return Connection()
    monkeypatch.setattr(replay.sqlite3, "connect", connect)
    assert replay.read_session_state(path, "fixed-sid") == {}
    assert calls == [("SELECT state_json FROM sessions WHERE id = ? LIMIT 1", ("fixed-sid",))]


def synthetic_state():
    return {
        "role_profile": {"job_title": "Synthetic specialist", "rubric": [{"key": "d1", "weight": 1}]},
        "resume_profile": {"projects": ["ALL INVENTED TEST DATA project"]},
        "question_plan": [{"question_kind": "motivation", "content": "plan question", "difficulty": "easy"}],
        "assessments": [
            {"question_id": 1, "plan_question_index": 0, "score": None, "score_error": True, "is_follow_up": False},
            {"question_id": 2, "plan_question_index": 0, "score": None, "score_error": True, "is_follow_up": True},
        ],
        "turn_records": [
            {"question_id": 1, "question": "Original synthetic Q1", "answer": "Original synthetic A1"},
            {"question_id": 2, "question": "Original synthetic Q2", "answer": "Original synthetic A2"},
        ],
        "usage_records": [{"input_tokens": 999, "total_tokens": 999}], "end_requested": True,
    }


def test_reconstructs_original_qa_and_controls_without_replacing_rubric(replay):
    state = synthetic_state()
    before = json.dumps(state, sort_keys=True)
    cases = replay.replay_inputs(state)
    assert len(cases) == 2
    for position, case in enumerate(cases):
        control = case["input"]
        assert control["current_question"] == f"Original synthetic Q{position + 1}"
        assert control["current_answer"] == f"Original synthetic A{position + 1}"
        assert control["global_question_counter"] == position
        assert control["follow_up_count"] == position
        assert control["role_profile"] == state["role_profile"]
        assert control["role_profile"]["rubric"][0]["key"] == "d1"
        assert control["usage_records"] == []
        assert control["end_requested"] is False
    assert json.dumps(state, sort_keys=True) == before


def test_missing_original_turn_rejected_without_guessing(replay):
    state = synthetic_state()
    state["turn_records"].pop()
    with pytest.raises(replay.ReplayError, match="original_turn_missing_or_ambiguous"):
        replay.replay_inputs(state)


def test_failed_targets_are_selected_and_execution_requires_explicit_sessions(replay):
    state = synthetic_state()
    state["assessments"][1]["score_error"] = False
    assert len(replay.replay_inputs(state)) == 1
    state["assessments"][0]["score_error"] = False
    with pytest.raises(replay.ReplayError, match="expected_one_to_twenty"):
        replay.replay_inputs(state)
    with pytest.raises(SystemExit):
        replay.arguments(["--execute"])
    with pytest.raises(SystemExit):
        replay.arguments(["--target", "../unsafe:session"])
    target = "a" * 32 + ":" + "b" * 32
    assert replay.arguments(["--target", target, "--execute"]).target == [target]


def test_safe_projection_drops_resume_prompt_and_headers(replay):
    projected = replay.safe_assessment({"score": 8, "dimensions": {"m1": 8, "m2": 8},
                                        "prompt": "not logged", "resume": "not logged", "headers": "not logged"})
    assert projected == {"score": 8, "dimensions": {"m1": 8, "m2": 8}}
