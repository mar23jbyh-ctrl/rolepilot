"""Local OCR dependencies, bounded execution and scanned-PDF rendering.

Never silently drop a requested language or send document images to an LLM.
"""
from __future__ import annotations

import importlib.util
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading

from PIL import Image, ImageOps

from app.config import PROJECT_ROOT, settings

MAX_RENDER_PIXELS = 20_000_000
MAX_PDF_PAGES = 30
_PDFIUM_LOCK = threading.RLock()  # PDFium's C API is not thread-safe.
_HAN = r"\u3400-\u9fff"
BUNDLED_TESSDATA = PROJECT_ROOT / "assets" / "ocr" / "tessdata"


class OcrError(RuntimeError):
    """Only safe error codes/labels are exposed; no document text or stderr."""
    def __init__(self, code: str, status_code: int = 503, missing_languages=()):
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.missing_languages = tuple(missing_languages)

    def public_detail(self) -> dict:
        result = {"code": self.code}
        if self.missing_languages:
            result["missing_languages"] = list(self.missing_languages)
        return result


def tessdata_dir() -> Path | None:
    path = settings.ocr_model_path
    # An explicit missing directory is an error, rather than a system fallback.
    if settings.ocr_tessdata_dir or path.is_dir():
        return path
    return BUNDLED_TESSDATA if BUNDLED_TESSDATA.is_dir() else None


def engine_config() -> str:
    path = tessdata_dir()
    prefix = f'--tessdata-dir "{path.as_posix()}" ' if path else ""
    return prefix + f"--oem 1 --psm {settings.ocr_psm}"


def engine_args(languages: str | None = None) -> list[str]:
    args = [settings.tesseract_cmd, "stdin", "stdout", "-l", languages or settings.ocr_languages,
            "--oem", "1", "--psm", str(settings.ocr_psm),
            "-c", "tessedit_load_sublangs="]  # Horizontal documents; don't silently require chi_sim_vert.
    if tessdata_dir():
        args += ["--tessdata-dir", str(tessdata_dir())]
    return args


def _language_status() -> dict:
    required = settings.ocr_languages.split("+")
    if not required or any(not re.fullmatch(r"[A-Za-z0-9_]+", s) for s in required):
        return {"ready": False, "code": "ocr_invalid_languages", "available_languages": [], "missing_languages": []}
    command = shutil.which(settings.tesseract_cmd)
    if not command:
        return {"ready": False, "code": "ocr_tesseract_missing", "available_languages": [], "missing_languages": required}
    folder = tessdata_dir()
    if folder is not None and not folder.is_dir():
        return {"ready": False, "code": "ocr_tessdata_missing", "available_languages": [], "missing_languages": required}
    args = [command, "--list-langs"]
    if folder:
        args += ["--tessdata-dir", str(folder)]
    try:
        completed = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, check=False)
        if completed.returncode:
            raise OSError("list-langs failed")
        available = sorted(set(line.strip() for line in completed.stdout.splitlines() if re.fullmatch(r"[A-Za-z0-9_]+", line.strip())))
    except (OSError, subprocess.TimeoutExpired):
        return {"ready": False, "code": "ocr_language_check_failed", "available_languages": [], "missing_languages": required}
    missing = [lang for lang in required if lang not in available]
    return {"ready": not missing, "code": "ocr_language_missing" if missing else "ok",
            "available_languages": available, "missing_languages": missing}


def pdf_renderer_status() -> dict:
    renderer = settings.ocr_pdf_renderer
    if renderer == "pdfium":
        try:
            if importlib.util.find_spec("pypdfium2") is None:
                raise ImportError
            import pypdfium2  # Check the native library loads, not merely package presence.
            ready = bool(pypdfium2.PdfDocument)
        except (ImportError, OSError, AttributeError):
            ready = False
    else:
        path = Path(settings.poppler_path) if settings.poppler_path else None
        ready = all((path / name).is_file() if path else bool(shutil.which(name))
                    for name in (("pdftoppm.exe", "pdfinfo.exe") if os.name == "nt" else ("pdftoppm", "pdfinfo")))
    return {"renderer": renderer, "ready": ready, "code": "ok" if ready else "ocr_pdf_renderer_missing"}


def dependency_status() -> dict:
    """Safe, fresh readiness check; does not promise recognition accuracy."""
    return {**_language_status(), "required_languages": settings.ocr_languages.split("+"),
            "pdf": pdf_renderer_status(), "timeout_seconds": settings.ocr_timeout_seconds,
            "pdf_dpi": settings.ocr_pdf_dpi, "psm": settings.ocr_psm}


def require_languages() -> None:
    status = _language_status()
    if not status["ready"]:
        raise OcrError(status["code"], missing_languages=status["missing_languages"])


def clean_ocr_text(text: str) -> str:
    """Repair CJK spacing and colon width only; never invent missing words.

    Tesseract often inserts spaces between Chinese words and emits ASCII ':'
    after a Chinese label. Keep English spaces, URL/numeric colons, case and
    other punctuation unchanged. This is formatting, not semantic correction.
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(rf"(?<=[{_HAN}])\s+(?=[{_HAN}，。；：、！？])", "", line)
        line = re.sub(rf"(?<=[，。；：、！？])\s+(?=[{_HAN}])", "", line)
        line = re.sub(rf"(?<=[{_HAN}])\s*:\s*", "：", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def run_engine(image: Image.Image, languages: str | None = None) -> str:
    """Argument-list invocation works for tessdata paths containing spaces.

    Avoid pytesseract's Windows shlex retaining quotes in --tessdata-dir.
    Image bytes go to local stdin; subprocess.run kills timed-out children.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", dpi=(300, 300))
    try:
        completed = subprocess.run(engine_args(languages), input=buffer.getvalue(), capture_output=True,
                                   timeout=settings.ocr_timeout_seconds, check=False,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except FileNotFoundError as exc:
        raise OcrError("ocr_tesseract_missing") from exc
    except subprocess.TimeoutExpired as exc:
        raise OcrError("ocr_timeout", 504) from exc
    except OSError as exc:
        raise OcrError("ocr_failed", 422) from exc
    if completed.returncode or b"Failed loading language" in completed.stderr:
        raise OcrError("ocr_failed", 422)
    return completed.stdout.decode("utf-8", errors="replace")


def image_text(image: Image.Image) -> str:
    require_languages()
    if image.width * image.height > MAX_RENDER_PIXELS:
        raise OcrError("ocr_image_too_large", 413)
    prepared = ImageOps.autocontrast(ImageOps.exif_transpose(image).convert("L"))
    text = run_engine(prepared)
    han_count = len(re.findall(rf"[{_HAN}]", text))
    letters = len(re.findall(rf"[{_HAN}A-Za-z]", text))
    # The Chinese model also contains Latin characters. For CJK-bearing
    # documents the English word model can misclassify entire Chinese lines.
    # Detect script from OCR output (not a reference answer), then use the
    # Chinese model. A failed pass is an error, never an English fallback.
    if (settings.ocr_adaptive_languages and {"chi_sim", "eng"}.issubset(settings.ocr_languages.split("+"))
            and han_count >= 12 and han_count / max(1, letters) >= .10):
        text = run_engine(prepared, languages="chi_sim")
    return clean_ocr_text(text)


def image_bytes_text(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        return image_text(image)


def render_pdf_page(path: Path, index: int) -> Image.Image:
    """Return an owned PIL copy. Index is zero-based; native resources close."""
    status = pdf_renderer_status()
    if not status["ready"]:
        raise OcrError(status["code"])
    try:
        if settings.ocr_pdf_renderer == "pdfium":
            import pypdfium2 as pdfium
            with _PDFIUM_LOCK:
                with pdfium.PdfDocument(str(path)) as document:
                    if len(document) > MAX_PDF_PAGES:
                        raise OcrError("too_many_pdf_pages", 413)
                    page = document[index]
                    try:
                        width, height = page.get_size()
                        scale = settings.ocr_pdf_dpi / 72
                        if math.ceil(width * scale) * math.ceil(height * scale) > MAX_RENDER_PIXELS:
                            raise OcrError("ocr_image_too_large", 413)
                        bitmap = page.render(scale=scale)
                        try:
                            return bitmap.to_pil().copy()
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
        else:
            from pdf2image import convert_from_path
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                if len(pdf.pages) > MAX_PDF_PAGES:
                    raise OcrError("too_many_pdf_pages", 413)
                page = pdf.pages[index]
                scale = settings.ocr_pdf_dpi / 72
                if math.ceil(page.width * scale) * math.ceil(page.height * scale) > MAX_RENDER_PIXELS:
                    raise OcrError("ocr_image_too_large", 413)
            pages = convert_from_path(str(path), dpi=settings.ocr_pdf_dpi, first_page=index + 1,
                                      last_page=index + 1, poppler_path=settings.poppler_path or None,
                                      timeout=settings.ocr_timeout_seconds)
            if not pages:
                raise OcrError("pdf_render_failed", 422)
            return pages[0]
    except OcrError:
        raise
    except Exception as exc:
        raise OcrError("pdf_render_failed", 422) from exc
