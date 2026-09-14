"""Preflight checks operate on metadata and never print secret match values."""
import importlib.util
import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def preflight():
    path = Path(__file__).resolve().parents[1] / "scripts/delivery_preflight.py"
    spec = importlib.util.spec_from_file_location("test_preflight_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_credential_findings_contain_only_location_and_rule(preflight):
    secret = "sk-" + "A" * 40
    result = preflight.scan_text("safe\n" + secret, "synthetic.txt")
    assert result == [{"path": "synthetic.txt", "rule": "provider_key", "line": 2}]
    assert secret not in str(result)


def test_explicit_synthetic_fixture_key_is_not_real_credential(preflight):
    assert preflight.scan_text("sk-synthetic-" + "A" * 40, "synthetic.txt") == []


@pytest.mark.parametrize("path, private", [
    (".env", True), (".env.production", True), (".env.example", False),
    ("data/sessions.sqlite3", True), ("data/sessions.db-wal", True), ("README.md", False),
])
def test_private_artifact_paths(preflight, path, private):
    assert preflight.private_path(path) is private


def test_dependency_closure_checks_constraints_requested_extras_and_platform(preflight, tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("parent[feature]==1\n", encoding="utf-8")
    (tmp_path / "requirements-dev.txt").write_text("", encoding="utf-8")
    versions = {"parent": "1", "child": "2", "optional": "3"}
    monkeypatch.setattr(preflight.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(preflight.metadata, "requires", lambda name: [
        "child>=2", "optional==3; extra == 'feature'", "absent; python_version < '2'"
    ] if name == "parent" else [])
    result = preflight.dependency_check(tmp_path)
    assert result["issue_count"] == 0 and result["unique_package_count"] == 3
    monkeypatch.setitem(versions, "child", "1")
    assert preflight.dependency_check(tmp_path)["issue_count"] == 1


@pytest.mark.parametrize("existing", [False, True])
def test_invalid_output_rejected_before_any_scan(preflight, tmp_path, monkeypatch, existing):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    target = tmp_path / "docs/release/evidence/existing" if existing else tmp_path / "outside"
    if existing:
        target.mkdir(parents=True)
    monkeypatch.setattr(preflight, "dependency_check", lambda: pytest.fail("Must reject target before scan"))
    with pytest.raises(SystemExit):
        preflight.main(["--output", str(target)])
    assert not (target / "preflight.json").exists()


@pytest.mark.parametrize("case", ["official", "tampered", "unknown"])
def test_large_historical_models_require_exact_pinned_content(preflight, tmp_path, monkeypatch, case):
    payload = b"\0" + b"X" * (6 * 1024 * 1024)
    name = "assets/ocr/tessdata/chi_sim.traineddata"
    relative = name if case != "unknown" else "assets/ocr/unknown.binary"
    expected = hashlib.sha256(payload if case != "tampered" else b"different").hexdigest()
    monkeypatch.setattr(preflight, "OFFICIAL_MODELS", {name: expected})
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    reads = []
    def fake_git(*args, input=None):
        if args[0] == "ls-files": return b""
        if args == ("rev-list", "--objects", "--all"):
            return ("model-blob " + relative + "\n").encode()
        if args[0] == "cat-file" and args[1].startswith("--batch-check"):
            return ("model-blob blob " + str(len(payload)) + "\n").encode()
        if args == ("cat-file", "blob", "model-blob"):
            reads.append(1)
            return payload
        if args == ("rev-list", "--all", "--count"): return b"1"
        raise AssertionError(args)
    monkeypatch.setattr(preflight, "git", fake_git)
    result = preflight.publication_scan()
    assert len(result["verified_official_model_blobs"]) == int(case == "official")
    assert len(result["skipped_blobs"]) == int(case != "official")
    assert len(result["history_findings"]) == int(case == "tampered")
    assert len(reads) == int(case != "unknown")


def test_current_binary_model_with_wrong_checksum_is_rejected(preflight, tmp_path, monkeypatch):
    name = "assets/ocr/tessdata/chi_sim.traineddata"
    model = tmp_path / name
    model.parent.mkdir(parents=True)
    model.write_bytes(b"\0untrusted-model")
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    def fake_git(*args, input=None):
        if args[0] == "ls-files": return name.encode() + b"\0"
        if args == ("rev-list", "--objects", "--all"): return b""
        if args == ("rev-list", "--all", "--count"): return b"1"
        raise AssertionError(args)
    monkeypatch.setattr(preflight, "git", fake_git)
    result = preflight.publication_scan()
    assert result["current_findings"] == [
        {"path": name, "rule": "official_model_checksum_mismatch", "line": None}]
    assert "untrusted-model" not in str(result)
