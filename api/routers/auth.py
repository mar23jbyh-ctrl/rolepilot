from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.security import identity_store

router = APIRouter(prefix="/api/auth", tags=["anonymous-auth"])


class AnonymousCredential(BaseModel):
    token: str
    token_type: str = "Bearer"


@router.post("/anonymous", response_model=AnonymousCredential, status_code=201)
def create_anonymous_identity(response: Response) -> AnonymousCredential:
    """Return the capability once; never log it or persist its plaintext."""
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    try:
        token = identity_store().issue()
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(503, "Anonymous identity storage unavailable") from exc
    return AnonymousCredential(token=token)
