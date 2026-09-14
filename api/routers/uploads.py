from __future__ import annotations

import uuid
import warnings
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import settings
from app.parsers.ocr import OcrError
from app.parsers.resume_parser import SUPPORTED_EXTENSIONS, extract_resume_text, ocr_image_file

from api.schemas import UploadOut

router = APIRouter(prefix="/api/uploads", tags=["uploads"])
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 256 * 1024  # multipart envelope allowance
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_IMAGE_PIXELS = 20_000_000


class UploadBodyLimitMiddleware:
    """Bound the raw body before multipart parsing, including chunked requests."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/uploads/"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "invalid_content_length"}, status_code=400)(scope, receive, send)
        if length < 0 or length > MAX_REQUEST_BYTES:
            return await JSONResponse({"detail": "upload_too_large"}, status_code=413)(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BYTES:
                    raise HTTPException(413, "upload_too_large")
            return message

        return await self.app(scope, limited_receive, send)


def _validate_content(target: Path) -> None:
    """Reject mismatched containers and bounded image/ZIP expansion hazards."""
    suffix = target.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        formats = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP", ".webp": "WEBP"}
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(target) as image:
                if image.format != formats[suffix] or image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(415, "invalid_image_upload")
                image.verify()
    elif suffix == ".pdf":
        with target.open("rb") as stream:
            if not stream.read(1024).lstrip().startswith(b"%PDF-"):
                raise HTTPException(415, "invalid_pdf_upload")
        # A tiny PDF can have many pages; cap the parser/OCR work as well.
        import pdfplumber
        with pdfplumber.open(str(target)) as document:
            if len(document.pages) > 30:
                raise HTTPException(413, "too_many_pdf_pages")
    elif suffix == ".docx":
        with zipfile.ZipFile(target) as archive:
            entries = archive.infolist()
            if len(entries) > 1000 or sum(item.file_size for item in entries) > 32 * 1024 * 1024:
                raise HTTPException(413, "docx_expansion_too_large")
            if "word/document.xml" not in archive.namelist():
                raise HTTPException(415, "invalid_docx_upload")
    elif suffix == ".txt":
        with target.open("rb") as stream:
            if b"\x00" in stream.read(MAX_UPLOAD_BYTES):
                raise HTTPException(415, "invalid_text_upload")


def _save(file: UploadFile, allowed: set[str]) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(415, "unsupported_upload_format")
    folder = settings.data_root / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (uuid.uuid4().hex + suffix)
    try:
        total = 0
        with target.open("xb") as stream:
            while chunk := file.file.read(64 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "upload_too_large")
                stream.write(chunk)
        if total == 0:
            raise HTTPException(400, "empty_upload")
        _validate_content(target)
    except Exception:
        target.unlink(missing_ok=True)  # only this request's newly created UUID file
        raise
    return target


@router.post("/resume", response_model=UploadOut)
def upload_resume(file: UploadFile = File(...)) -> UploadOut:
    target = None
    try:
        target = _save(file, SUPPORTED_EXTENSIONS)
        parsed = extract_resume_text(target)
        return UploadOut(
            text=parsed["text"],
            method=parsed.get("method", "text"),
            file_name=parsed.get("file_name", target.name),
        )
    except OcrError as exc:
        raise HTTPException(exc.status_code, exc.public_detail()) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="resume_parse_failed") from exc
    finally:
        _remove_original(target)


@router.post("/jd-image", response_model=UploadOut)
def upload_jd_image(file: UploadFile = File(...)) -> UploadOut:
    target = None
    try:
        target = _save(file, IMAGE_EXTENSIONS)
        text = ocr_image_file(target)
        if not text.strip():
            raise HTTPException(400, "no_readable_text")
        return UploadOut(text=text, method="ocr", file_name=target.name)
    except OcrError as exc:
        raise HTTPException(exc.status_code, exc.public_detail()) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="jd_image_parse_failed") from exc
    finally:
        _remove_original(target)


def _remove_original(target: Path | None) -> None:
    """No retained attachments: extraction returns text, never a file URL."""
    if target is not None:
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            # Never claim success when sensitive source cleanup failed.
            raise HTTPException(503, "upload_cleanup_failed") from exc
