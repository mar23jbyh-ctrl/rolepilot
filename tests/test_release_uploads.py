"""Offline uploads: temporary files, no OCR process or model/search request."""
from __future__ import annotations

from io import BytesIO
import zipfile

from fastapi.testclient import TestClient
from PIL import Image
import pytest

import api.deps as deps
from api.main import app
from api.routers import auth, uploads
from app.config import settings
from app.security import AnonymousIdentityStore


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    store = AnonymousIdentityStore(tmp_path / "auth.db")
    monkeypatch.setattr(auth, "identity_store", lambda: store)
    monkeypatch.setattr(deps, "identity_store", lambda: store)
    client = TestClient(app)
    token = client.post("/api/auth/anonymous").json()["token"]
    client.headers["Authorization"] = "Bearer " + token
    return client, tmp_path


def test_text_resume_real_parser_succeeds_and_filename_is_server_generated(upload_client):
    client, root = upload_client
    response = client.post("/api/uploads/resume", files={"file": ("../../synthetic.txt", b"Synthetic candidate\nPython project")})
    assert response.status_code == 200
    assert response.json()["text"] == "Synthetic candidate\nPython project"
    assert response.json()["method"] == "text"
    assert "/" not in response.json()["file_name"]
    assert not list((root / "uploads").iterdir())


@pytest.mark.parametrize("endpoint, name, content, expected", [
    ("resume", "synthetic.exe", b"anything", 415),
    ("jd-image", "synthetic.txt", b"anything", 415),
    ("resume", "synthetic.txt", b"", 400),
    ("resume", "synthetic.txt", b"binary\x00payload", 415),
    ("resume", "synthetic.pdf", b"not a pdf", 415),
    ("resume", "synthetic.png", b"not an image", 400),
    ("resume", "synthetic.docx", b"not a zip", 400),
])
def test_invalid_uploads_are_rejected_without_files(upload_client, endpoint, name, content, expected):
    client, root = upload_client
    response = client.post("/api/uploads/" + endpoint, files={"file": (name, content)})
    assert response.status_code == expected
    assert not list((root / "uploads").glob("*"))


def test_bounded_file_stream_removes_partial_file(upload_client, monkeypatch):
    client, root = upload_client
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/api/uploads/resume", files={"file": ("synthetic.txt", b"a" * 11)})
    assert response.status_code == 413
    assert not list((root / "uploads").glob("*"))


def test_raw_body_limit_rejects_before_multipart_parser(upload_client, monkeypatch):
    client, root = upload_client
    monkeypatch.setattr(uploads, "MAX_REQUEST_BYTES", 32)
    response = client.post("/api/uploads/resume", content=b"a" * 33,
                           headers={"Content-Type": "multipart/form-data; boundary=test"})
    assert response.status_code == 413
    assert not (root / "uploads").exists()


def test_chunked_raw_body_limit_is_enforced(upload_client, monkeypatch):
    client, root = upload_client
    monkeypatch.setattr(uploads, "MAX_REQUEST_BYTES", 32)
    response = client.post("/api/uploads/resume", content=iter([b"a" * 20, b"b" * 20]),
                           headers={"Content-Type": "multipart/form-data; boundary=test"})
    assert response.status_code == 413
    assert not (root / "uploads").exists()


def test_parser_failure_is_redacted_and_new_file_is_cleaned(upload_client, monkeypatch):
    client, root = upload_client

    def fail(*args):
        raise RuntimeError("synthetic confidential parser detail")

    monkeypatch.setattr(uploads, "extract_resume_text", fail)
    response = client.post("/api/uploads/resume", files={"file": ("synthetic.txt", b"valid text")})
    assert response.status_code == 400
    assert response.json()["detail"] == "resume_parse_failed"
    assert "confidential" not in response.text
    assert not list((root / "uploads").glob("*"))


def test_image_pixel_work_is_bounded(upload_client, monkeypatch):
    client, root = upload_client
    image = BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    monkeypatch.setattr(uploads, "MAX_IMAGE_PIXELS", 1)
    response = client.post("/api/uploads/jd-image", files={"file": ("synthetic.png", image.getvalue())})
    assert response.status_code == 415
    assert not list((root / "uploads").glob("*"))


def test_docx_container_is_validated(upload_client):
    client, root = upload_client
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as container:
        container.writestr("not-word.xml", "synthetic")
    response = client.post("/api/uploads/resume", files={"file": ("synthetic.docx", archive.getvalue())})
    assert response.status_code == 415
    assert not list((root / "uploads").glob("*"))


def test_valid_image_ocr_boundary_receives_only_validated_file(upload_client, monkeypatch):
    client, _ = upload_client
    image = BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    monkeypatch.setattr(uploads, "ocr_image_file", lambda path: "Synthetic job description")
    response = client.post("/api/uploads/jd-image", files={"file": ("synthetic.png", image.getvalue())})
    assert response.status_code == 200
    assert response.json()["text"] == "Synthetic job description"
