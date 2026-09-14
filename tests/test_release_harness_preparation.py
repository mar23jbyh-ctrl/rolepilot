"""Offline fixture/harness checks; never import the production app or call providers."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("offline_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def driver():
    return load_script("release_run")


def test_imports_are_lazy_and_syntax_valid():
    for name in ("release_run", "release_server"):
        tree = ast.parse((ROOT / "scripts" / (name + ".py")).read_text(encoding="utf-8"))
        top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in top_level_imports:
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            assert not any(name.startswith(("app", "api", "requests", "openai", "dotenv", "uvicorn")) for name in names)
        load_script(name)


def test_sources_resumes_and_word_budget(driver):
    cases = driver.fixture_cases()
    assert set(cases) == set(driver.CASES)
    for case in cases.values():
        assert len(case["jd"].split()) < 180
        assert case["source"]["retrieved_on"] == "2026-09-12"
        assert case["source"]["url"] in case["jd"]
        assert "ALL INVENTED TEST DATA" in case["resume"]
        assert case["facts"]["data_label"] == "ALL INVENTED TEST DATA"
        assert len(case["facts"]["projects"]) >= 2
        assert "fictional" in case["facts"]["competition"]["invented_result"]


def test_spin_source_correction(driver):
    case = driver.fixture_cases()["technical_writing"]
    source = case["source"]
    assert source["experience_years_minimum"] == 3
    assert source["requirements"] == [
        "3+ years of B2B SaaS technical writing",
        "Bachelor-level degree in technical communication, English, computer science or engineering",
        "Jira and Confluence", "Writing/editing and cloud/cybersecurity knowledge", "Writing portfolio",
    ]
    assert all(word not in case["jd"].casefold() for word in ("markdown", "docusaurus", "openapi", "git"))
    assert case["facts"]["invented_professional_history"]["duration_years"] == 3


def test_default_llm_and_budget(driver):
    args = driver.arguments(["--run-id", "offline"])
    assert args.answer_mode == "llm"
    assert args.cases == list(driver.CASES)
    assert args.max_rounds == 60
    assert args.max_total_answers == 180
    assert args.execute is False


@pytest.mark.parametrize("case", ("observability", "product_data", "technical_writing"))
def test_individual_case_selection(driver, case):
    args = driver.arguments(["--run-id", "same-run", "--case", case, "--answer-mode", "llm"])
    assert args.cases == [case]


@pytest.mark.parametrize("options", (
    ["--max-rounds", "61"], ["--max-total-answers", "181"],
    ["--base-url", "https://example.com"], ["--run-id", "../unsafe"],
))
def test_invalid_execution_targets_rejected(driver, options):
    with pytest.raises(SystemExit):
        driver.arguments(["--run-id", "offline"] + options)


def test_no_execute_does_not_initialize_http(driver, monkeypatch, capsys):
    def prohibited(*args, **kwargs):
        raise AssertionError("Execution must remain opt-in")
    monkeypatch.setattr(driver, "HTTP", prohibited)
    monkeypatch.setattr(driver, "Candidate", prohibited)
    monkeypatch.setattr(driver, "fixture_cases", prohibited)
    assert driver.main(["--run-id", "offline"]) == 0
    assert "Prepared only" in capsys.readouterr().out


def test_bootstrap_response_not_archived_or_printed(driver, tmp_path, capsys):
    evidence = driver.Evidence(tmp_path)
    credential = "invented-test-credential-do-not-export"
    response = SimpleNamespace(status_code=201, json=lambda: {"token": credential, "token_type": "Bearer"})
    class Session:
        def __init__(self):
            self.headers = {}
        def request(self, *args, **kwargs):
            return response
        def close(self):
            pass
    # Bypass HTTP.__init__ solely to install an offline fake transport.
    http = driver.HTTP.__new__(driver.HTTP)
    http.session = Session()
    http.args = driver.arguments(["--run-id", "offline"])
    http.evidence, http.case_id = evidence, "observability"
    http.requests = SimpleNamespace(RequestException=RuntimeError)
    http.authenticate()
    assert http.session.headers["Authorization"] == "Bearer " + credential
    http.close()
    evidence.close()
    assert credential not in (tmp_path / "driver.jsonl").read_text(encoding="utf-8")
    assert credential not in capsys.readouterr().out
    assert evidence.records[0]["response"] == {"excluded": "auth_bootstrap"}


def test_existing_evidence_not_overwritten(driver, tmp_path):
    path = tmp_path / "driver.jsonl"
    path.write_text("existing offline evidence", encoding="utf-8")
    with pytest.raises(driver.RunFailure):
        driver.Evidence(tmp_path)
    assert path.read_text(encoding="utf-8") == "existing offline evidence"


def test_templates_are_topic_specific_and_no_model(driver, tmp_path):
    evidence = driver.Evidence(tmp_path)
    candidate = driver.Candidate("templates", evidence)
    cases = driver.fixture_cases()
    for case_id, question, topic in (
        ("observability", "Why does Prometheus metric cardinality matter?", "metrics"),
        ("product_data", "How do settlement reconciliation exceptions work?", "reconciliation"),
        ("technical_writing", "How do you use Jira and Confluence?", "writer_process"),
    ):
        answer = candidate.answer(case_id, cases[case_id], question, 1, "synthetic", 1, [])
        assert answer
        assert evidence.candidate_records[-1]["topic"] == topic
        assert evidence.candidate_records[-1]["model_calls"] == 0
    candidate.answer("product_data", cases["product_data"], "quux frobnicator", 2, "synthetic", 2, [])
    assert evidence.candidate_records[-1]["matched"] is False
    assert driver.candidate_totals(evidence)["logical_calls"] == 0
    evidence.close()


def test_gui_empty_case_not_in_case_statistics(driver, tmp_path):
    events = [
        {"kind": "node_start", "case_id": "", "thread_id": "gui-only", "function": "analyze"},
        {"kind": "node_start", "case_id": "observability", "thread_id": "case-only", "function": "ask"},
        {"kind": "graph_end", "case_id": "observability", "thread_id": "case-only", "next_nodes": [], "usage_records": []},
    ]
    (tmp_path / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in events), encoding="utf-8")
    summary = driver.observer_summary(tmp_path, [{"case_id": "observability", "session_id": "case-only"}])
    assert summary["cases"]["observability"]["node_order"] == ["ask"]
