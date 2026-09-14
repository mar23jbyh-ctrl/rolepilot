"""Offline dependency/error/format/PDF/HTTP regression. Accuracy is measured separately."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import api.deps as deps
from api.main import app, public_self_check
from api.routers import auth
from app.config import Settings, settings
from app.parsers import ocr, resume_parser
from app.security import AnonymousIdentityStore
from scripts.ocr_benchmark import edit_distance, normalize, score


@pytest.fixture
def engine_ready(monkeypatch):
    monkeypatch.setattr(ocr, "require_languages", lambda: None)


@pytest.mark.parametrize("available,missing", [("eng\nosd", ["chi_sim"]), ("chi_sim", ["eng"]), ("osd", ["chi_sim", "eng"])])
def test_missing_required_language_stops_before_engine(monkeypatch, available, missing):
    monkeypatch.setattr(settings, "ocr_languages", "chi_sim+eng")
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "tesseract")
    monkeypatch.setattr(ocr.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="header\n" + available))
    calls = []
    monkeypatch.setattr(ocr, "run_engine", lambda *a, **k: calls.append(1))
    with pytest.raises(ocr.OcrError) as caught:
        ocr.image_text(Image.new("RGB", (2, 2)))
    assert caught.value.code == "ocr_language_missing"
    assert list(caught.value.missing_languages) == missing
    assert calls == []


def test_missing_executable_has_explicit_safe_error(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    assert ocr.dependency_status()["code"] == "ocr_tesseract_missing"
    with pytest.raises(ocr.OcrError, match="ocr_tesseract_missing"):
        ocr.require_languages()


def test_explicit_missing_tessdata_is_not_replaced_by_local_or_system(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "tesseract")
    monkeypatch.setattr(settings, "ocr_tessdata_dir", str(tmp_path / "absent"))
    assert ocr.dependency_status()["code"] == "ocr_tessdata_missing"


@pytest.mark.parametrize("value", ["chi_sim+../eng", "chi_sim+", "--bad", ""])
def test_invalid_language_tokens_are_rejected(monkeypatch, value):
    monkeypatch.setattr(settings, "ocr_languages", value)
    assert ocr._language_status()["code"] == "ocr_invalid_languages"


def test_language_self_check_is_bounded_fresh_and_parses_only_language_names(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "tesseract")
    calls = []
    def execute(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='List of languages in "private/local/path" (2):\neng\nchi_sim\n')
    monkeypatch.setattr(ocr.subprocess, "run", execute)
    assert ocr._language_status()["ready"]
    assert ocr._language_status()["ready"]
    assert len(calls) == 2 and all(k["timeout"] == 5 for _, k in calls)
    assert ocr._language_status()["available_languages"] == ["chi_sim", "eng"]


def test_language_check_timeout_is_explicit(monkeypatch):
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "tesseract")
    def fail(*a, **k):
        raise subprocess.TimeoutExpired("tesseract", 5)
    monkeypatch.setattr(ocr.subprocess, "run", fail)
    assert ocr._language_status()["code"] == "ocr_language_check_failed"


def test_engine_path_with_spaces_is_one_unquoted_argument(monkeypatch, tmp_path):
    folder = tmp_path / "models with spaces"
    folder.mkdir()
    monkeypatch.setattr(settings, "ocr_tessdata_dir", str(folder))
    captured = []
    def execute(args, **kwargs):
        captured.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="中文".encode(), stderr=b"")
    monkeypatch.setattr(ocr.subprocess, "run", execute)
    assert ocr.run_engine(Image.new("L", (2, 2))) == "中文"
    args, options = captured[0]
    assert args[args.index("--tessdata-dir") + 1] == str(folder)
    assert options["timeout"] == settings.ocr_timeout_seconds
    assert options["input"].startswith(b"\x89PNG")
    assert "shell" not in options


@pytest.mark.parametrize("failure,code,status", [
    (FileNotFoundError("private engine path"), "ocr_tesseract_missing", 503),
    (subprocess.TimeoutExpired("private command", 30), "ocr_timeout", 504),
    (PermissionError("private detail"), "ocr_failed", 422),
])
def test_engine_exceptions_are_safe_and_never_fallback(monkeypatch, failure, code, status):
    calls = []
    def execute(*a, **k):
        calls.append(1)
        raise failure
    monkeypatch.setattr(ocr.subprocess, "run", execute)
    with pytest.raises(ocr.OcrError) as caught:
        ocr.run_engine(Image.new("L", (2, 2)))
    assert caught.value.public_detail() == {"code": code}
    assert caught.value.status_code == status and calls == [1]


@pytest.mark.parametrize("exit_code,stderr", [(1, b"sensitive engine error"), (0, b"Failed loading language 'chi_sim'")])
def test_nonzero_or_partial_language_loading_is_not_success(monkeypatch, exit_code, stderr):
    monkeypatch.setattr(ocr.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=exit_code, stderr=stderr, stdout=b"partial"))
    with pytest.raises(ocr.OcrError, match="ocr_failed"):
        ocr.run_engine(Image.new("L", (2, 2)))


def test_cjk_spacing_repair_does_not_rewrite_words_urls_or_numeric_colons():
    text = "中 文 技能 : Python FastAPI\nSQL query https://example.org:8080 12:30\n识 别 错 字：畦 等 性"
    assert ocr.clean_ocr_text(text) == "中文技能：Python FastAPI\nSQL query https://example.org:8080 12:30\n识别错字：畦等性"


def test_cjk_adaptive_pass_uses_script_not_reference(engine_ready, monkeypatch):
    monkeypatch.setattr(settings, "ocr_adaptive_languages", True)
    monkeypatch.setattr(settings, "ocr_languages", "chi_sim+eng")
    calls = []
    def execute(image, languages=None):
        calls.append(languages)
        return "中文简历项目实践合同审查风险评估 Python" if not languages else "中文识别结果 Python"
    monkeypatch.setattr(ocr, "run_engine", execute)
    assert ocr.image_text(Image.new("L", (2, 2))) == "中文识别结果 Python"
    assert calls == [None, "chi_sim"]


def test_english_does_not_trigger_chinese_second_pass(engine_ready, monkeypatch):
    calls = []
    monkeypatch.setattr(ocr, "run_engine", lambda *a, **k: calls.append(k) or "English Python FastAPI SQL")
    assert ocr.image_text(Image.new("L", (2, 2))).startswith("English")
    assert len(calls) == 1


def test_adaptive_second_pass_failure_is_not_hidden(engine_ready, monkeypatch):
    def execute(image, languages=None):
        if languages:
            raise ocr.OcrError("ocr_failed", 422)
        return "中文简历项目实践合同审查风险评估"
    monkeypatch.setattr(ocr, "run_engine", execute)
    with pytest.raises(ocr.OcrError, match="ocr_failed"):
        ocr.image_text(Image.new("L", (2, 2)))


def test_large_image_rejected_before_engine(engine_ready, monkeypatch):
    monkeypatch.setattr(ocr, "MAX_RENDER_PIXELS", 1)
    monkeypatch.setattr(ocr, "run_engine", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(ocr.OcrError, match="ocr_image_too_large"):
        ocr.image_text(Image.new("L", (2, 2)))


def fake_pdf(monkeypatch, pages):
    import pdfplumber
    class Document:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    document = Document()
    document.pages = pages
    monkeypatch.setattr(pdfplumber, "open", lambda *a, **k: document)


def test_short_native_text_pdf_does_not_require_ocr(monkeypatch):
    fake_pdf(monkeypatch, [SimpleNamespace(extract_text=lambda: "Short native PDF", images=[])])
    monkeypatch.setattr(resume_parser, "render_pdf_page", lambda *a: pytest.fail("must not render"))
    parsed = resume_parser.extract_resume_text("synthetic.pdf")
    assert parsed["text"] == "Short native PDF" and parsed["method"] == "text"


def test_scanned_pdf_uses_one_ocr_per_page_and_reports_ocr(monkeypatch):
    fake_pdf(monkeypatch, [SimpleNamespace(extract_text=lambda: "", images=[{}]) for _ in range(2)])
    render_calls, ocr_calls = [], []
    monkeypatch.setattr(resume_parser, "render_pdf_page", lambda path, i: render_calls.append(i) or Image.new("L", (2, 2)))
    monkeypatch.setattr(resume_parser, "image_text", lambda im: ocr_calls.append(1) or "Scanned text")
    parsed = resume_parser.extract_resume_text("synthetic.pdf")
    assert render_calls == [0, 1] and len(ocr_calls) == 2
    assert parsed["method"] == "ocr" and parsed["ocr_used"]
    assert parsed["text"] == "Scanned text\nScanned text"


def test_mixed_pdf_dependency_failure_does_not_return_partial_native_text(monkeypatch):
    fake_pdf(monkeypatch, [SimpleNamespace(extract_text=lambda: "Native text", images=[]), SimpleNamespace(extract_text=lambda: "", images=[{}])])
    def fail(*a):
        raise ocr.OcrError("ocr_pdf_renderer_missing")
    monkeypatch.setattr(resume_parser, "render_pdf_page", fail)
    with pytest.raises(ocr.OcrError, match="ocr_pdf_renderer_missing"):
        resume_parser.extract_resume_text("synthetic.pdf")


def test_pdf_page_limit_is_enforced_for_direct_parser(monkeypatch):
    fake_pdf(monkeypatch, [None] * 31)
    with pytest.raises(ocr.OcrError, match="too_many_pdf_pages"):
        resume_parser.extract_resume_text("synthetic.pdf")


def test_pdfium_unavailable_has_explicit_dependency_status(monkeypatch):
    monkeypatch.setattr(settings, "ocr_pdf_renderer", "pdfium")
    monkeypatch.setattr(ocr.importlib.util, "find_spec", lambda _: None)
    assert ocr.pdf_renderer_status() == {"renderer": "pdfium", "ready": False, "code": "ocr_pdf_renderer_missing"}


def test_explicit_poppler_path_is_checked_separately_from_path(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ocr_pdf_renderer", "poppler")
    monkeypatch.setattr(settings, "poppler_path", str(tmp_path))
    monkeypatch.setattr(ocr.shutil, "which", lambda _: "irrelevant PATH entry")
    assert not ocr.pdf_renderer_status()["ready"]
    names = ("pdftoppm.exe", "pdfinfo.exe") if ocr.os.name == "nt" else ("pdftoppm", "pdfinfo")
    for name in names:
        (tmp_path / name).touch()
    assert ocr.pdf_renderer_status()["ready"]


@pytest.fixture
def ocr_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    store = AnonymousIdentityStore(tmp_path / "auth.db")
    monkeypatch.setattr(auth, "identity_store", lambda: store)
    monkeypatch.setattr(deps, "identity_store", lambda: store)
    client = TestClient(app)
    token = client.post("/api/auth/anonymous").json()["token"]
    client.headers["Authorization"] = "Bearer " + token
    return client, tmp_path


@pytest.mark.parametrize("endpoint", ["resume", "jd-image"])
@pytest.mark.parametrize("code,status", [("ocr_language_missing", 503), ("ocr_timeout", 504), ("ocr_failed", 422)])
def test_upload_returns_explicit_ocr_error_and_cleans_new_file(ocr_client, monkeypatch, endpoint, code, status):
    client, root = ocr_client
    def fail(*a, **k):
        raise ocr.OcrError(code, status, ["chi_sim"] if code == "ocr_language_missing" else [])
    monkeypatch.setattr(ocr, "image_bytes_text", fail)
    monkeypatch.setattr(resume_parser, "image_bytes_text", fail)
    buffer = BytesIO()
    Image.new("L", (2, 2)).save(buffer, "PNG")
    response = client.post("/api/uploads/" + endpoint, files={"file": ("synthetic.png", buffer.getvalue())})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert not list((root / "uploads").glob("*"))
    assert "tessdata" not in response.text and "C:\\" not in response.text


def test_public_self_check_exposes_languages_not_installation_paths():
    status = public_self_check()["ocr"]
    assert "available_languages" in status and "pdf" in status
    assert "tessdata_dir" not in status and "command" not in status


@pytest.mark.parametrize("key,value", [("ocr_pdf_dpi", 0), ("ocr_timeout_seconds", 0), ("ocr_psm", 99), ("ocr_pdf_renderer", "unknown")])
def test_invalid_ocr_settings_rejected(key, value):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(**{key: value})


def test_metrics_keep_punctuation_case_and_actual_edit_errors():
    assert edit_distance("abc", "adc") == 1
    assert edit_distance("", "abc") == 3
    assert normalize("中 文\nSQL，") == "中文SQL，"
    metrics = score("中文：SQL", "中文:sql", ["SQL"])
    assert metrics["errors"] == 4  # punctuation + 3 case differences
    assert metrics["skills_recognized"] == 1  # explicitly case-insensitive skill metric


def test_skill_sql_does_not_match_only_postgresql():
    metrics = score("PostgreSQL SQL", "PostgreSQL", ["PostgreSQL", "SQL"])
    assert metrics["recognized"] == ["PostgreSQL"]
    assert metrics["missing"] == ["SQL"]
