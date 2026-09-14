"""Dependency provisioning contracts; no real download or system modification."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import PROJECT_ROOT, settings
from app.parsers import ocr
from scripts import install_ocr_languages as installer


@pytest.mark.parametrize("explicit", ["", "models/chinese", "/opt/rolepilot/models"])
def test_installer_and_runtime_use_the_same_configured_path(monkeypatch, tmp_path, explicit):
    # Use an OS-native absolute path even on Windows runners.
    if explicit.startswith("/opt"):
        explicit = str(tmp_path / "explicit")
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "runtime"))
    monkeypatch.setattr(settings, "ocr_tessdata_dir", explicit)
    expected = PROJECT_ROOT / explicit if explicit else settings.data_root / "ocr/tessdata"
    assert settings.ocr_model_path == expected
    expected.mkdir(parents=True, exist_ok=True)
    try:
        assert ocr.tessdata_dir() == expected
        calls = []
        monkeypatch.setattr(installer, "install", lambda dest, cmd, source: calls.append(dest) or {})
        monkeypatch.setattr(installer.sys, "argv", ["install_ocr_languages.py"])
        installer.main()
        assert calls == [expected]
    finally:
        if explicit == "models/chinese":
            expected.rmdir()
            expected.parent.rmdir()


def test_fresh_download_uses_bundled_models_without_local_install(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "new-data"))
    monkeypatch.setattr(settings, "ocr_tessdata_dir", "")
    assert ocr.tessdata_dir() == ocr.BUNDLED_TESSDATA


def test_system_models_remain_available_if_no_bundle_or_local_models(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "new-data"))
    monkeypatch.setattr(settings, "ocr_tessdata_dir", "")
    monkeypatch.setattr(ocr, "BUNDLED_TESSDATA", tmp_path / "no-bundle")
    assert ocr.tessdata_dir() is None


def test_bundled_models_match_the_pinned_official_manifest():
    root = PROJECT_ROOT / "assets/ocr"
    manifest = json.loads((root / "models.json").read_text(encoding="utf-8"))
    assert manifest["license"] == "Apache-2.0"
    assert {model["language"] for model in manifest["models"]} == {"chi_sim", "eng"}
    for model in manifest["models"]:
        path = root / "tessdata" / (model["language"] + ".traineddata")
        assert path.stat().st_size == model["bytes"]
        assert installer.digest(path) == model["sha256"]
        assert manifest["commit"] in model["source_url"]


@pytest.fixture
def provision(monkeypatch, tmp_path):
    source = tmp_path / "system"
    source.mkdir()
    (source / "eng.traineddata").write_bytes(b"existing-English")
    monkeypatch.setattr(installer.shutil, "which", lambda _: "tesseract")
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(stdout="chi_sim\neng\n", returncode=0))
    chinese = b"checksum-verified-Chinese-model"
    monkeypatch.setattr(installer, "EXPECTED_SHA256", hashlib.sha256(chinese).hexdigest())
    monkeypatch.setattr(installer, "BUNDLED_TESSDATA", tmp_path / "no-bundle")
    return source, tmp_path / "local/tessdata", chinese


def test_verified_source_model_can_be_reused_without_network(monkeypatch, provision):
    source, destination, chinese = provision
    (source / "chi_sim.traineddata").write_bytes(chinese)
    def forbidden(*args, **kwargs):
        raise AssertionError("A verified local source must not download")
    monkeypatch.setattr(installer.urllib.request, "urlopen", forbidden)
    record = installer.install(destination, "tesseract", source)
    assert (destination / "chi_sim.traineddata").read_bytes() == chinese
    assert (destination / "eng.traineddata").read_bytes() == b"existing-English"
    assert record["languages"] == ["chi_sim", "eng"]
    assert not record["system_tessdata_modified"]
    # Re-running must preserve the installed models.
    before = {p.name: p.stat().st_mtime_ns for p in destination.iterdir()}
    installer.install(destination, "tesseract", source)
    assert before == {p.name: p.stat().st_mtime_ns for p in destination.iterdir()}


def test_download_checksum_mismatch_installs_no_model(monkeypatch, provision):
    source, destination, _ = provision
    monkeypatch.setattr(installer.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"corrupt"))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        installer.install(destination, "tesseract", source)
    assert not (destination / "chi_sim.traineddata").exists()
    assert not (destination.parent / "installation.json").exists()
    assert (source / "eng.traineddata").read_bytes() == b"existing-English"


def test_custom_existing_model_is_not_overwritten(monkeypatch, provision):
    source, destination, _ = provision
    destination.mkdir(parents=True)
    (destination / "chi_sim.traineddata").write_bytes(b"custom")
    with pytest.raises(RuntimeError, match="do not overwrite"):
        installer.install(destination, "tesseract", source)
    assert (destination / "chi_sim.traineddata").read_bytes() == b"custom"


def test_installer_refuses_to_change_system_tessdata(provision):
    source, _, _ = provision
    with pytest.raises(RuntimeError, match="not system tessdata"):
        installer.install(source, "tesseract", source)
    assert sorted(p.name for p in source.iterdir()) == ["eng.traineddata"]


def test_installer_requires_english_before_download(monkeypatch, provision):
    source, destination, _ = provision
    (source / "eng.traineddata").unlink()
    with pytest.raises(RuntimeError, match="English model first"):
        installer.install(destination, "tesseract", source)
    assert not destination.exists()
