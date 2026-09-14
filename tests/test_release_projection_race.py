"""Non-network tests for restore projection compare-and-swap."""
from app.session.store import SessionStore


def _store(tmp_path):
    store = SessionStore(tmp_path / "synthetic.db")
    store.create("synthetic", "public JD summary", "synthetic resume", {"question_version": 1})
    store.update("synthetic", {"question_version": 1, "current_question": "q1"})
    return store


def test_idle_projection_can_repair_unchanged_snapshot(tmp_path):
    store = _store(tmp_path)
    old = store.get("synthetic")["state_json"]
    assert store.project_if_idle("synthetic", {"question_version": 2, "current_question": "q2"}, old)
    assert store.load_state("synthetic")["question_version"] == 2


def test_restore_does_not_overwrite_active_answer(tmp_path):
    store = _store(tmp_path)
    old = store.get("synthetic")["state_json"]
    store.claim_answer("synthetic", "synthetic-request", 1, "synthetic answer")
    assert not store.project_if_idle("synthetic", {"question_version": 99}, old)
    assert store.load_state("synthetic")["question_version"] == 1
    assert store.get("synthetic")["active_operation"] == "answer:synthetic-request"


def test_restore_does_not_roll_back_newly_committed_snapshot(tmp_path):
    store = _store(tmp_path)
    old = store.get("synthetic")["state_json"]
    store.update("synthetic", {"question_version": 3, "current_question": "q3"})
    assert not store.project_if_idle("synthetic", {"question_version": 2}, old)
    assert store.load_state("synthetic")["question_version"] == 3
