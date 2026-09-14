from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import self_check
from app.branding import API_TITLE
from app.parsers.ocr import dependency_status
from api.routers import auth, sessions, uploads
from api.deps import resolve_owner

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def public_self_check() -> dict[str, Any]:
    """Do not expose local paths, URL credentials, query strings or fragments."""
    raw = self_check()
    allowed = ("api_key_set", "tavily_key_set", "env_file_exists")
    result: dict[str, Any] = {key: bool(raw.get(key)) for key in allowed}
    model = str(raw.get("model", ""))
    result["model"] = model if re.fullmatch(r"[A-Za-z0-9._-]{1,80}", model) else "configured"
    try:
        result["model_host"] = urlsplit(str(raw.get("base_url", ""))).hostname or ""
    except ValueError:
        result["model_host"] = ""
    result["ocr"] = raw.get("ocr") or dependency_status()
    return result


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Bounded local dependency check on startup; no model/network request."""
    summary = public_self_check()
    logger.info("startup self_check: %s", json.dumps(summary, ensure_ascii=False))
    if not summary["ocr"]["ready"]:
        logger.warning("OCR unavailable: %s; missing languages: %s", summary["ocr"]["code"], summary["ocr"]["missing_languages"])
    if not summary["ocr"]["pdf"]["ready"]:
        logger.warning("Scanned PDF renderer unavailable: %s", summary["ocr"]["pdf"]["renderer"])
    yield


app = FastAPI(title=API_TITLE, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(uploads.UploadBodyLimitMiddleware)

app.include_router(auth.router)
app.include_router(uploads.router, dependencies=[Depends(resolve_owner)])
app.include_router(sessions.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/self-check", dependencies=[Depends(resolve_owner)])
def self_check_endpoint() -> dict[str, Any]:
    return public_self_check()


# Serve the built SPA when it exists (production-style single command);
# during development Vite proxies /api to this server instead.
if (FRONTEND_DIST / "index.html").is_file():
    if (FRONTEND_DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(404, "Not Found")
        index = FRONTEND_DIST / "index.html"
        return FileResponse(index)
