from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel

from app.security import identity_store

router = APIRouter(prefix="/api/auth", tags=["anonymous-auth"])


class AnonymousCredential(BaseModel):
    token: str
    token_type: str = "Bearer"
    server_id: str
    identity_id: str
    reused: bool


class AnonymousContext(BaseModel):
    server_id: str


def no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


@router.get("/context", response_model=AnonymousContext)
def anonymous_context(response: Response) -> AnonymousContext:
    """Identify the current data environment without granting access to it."""
    no_store(response)
    try:
        return AnonymousContext(server_id=identity_store().server_id())
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(503, "Anonymous identity storage unavailable") from exc


@router.post("/anonymous", response_model=AnonymousCredential, status_code=201)
def create_anonymous_identity(
    response: Response,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> AnonymousCredential:
    """Reuse a valid capability or issue one for this environment; never log it."""
    no_store(response)
    try:
        store = identity_store()
        parts = (authorization or "").split()
        token = parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else ""
        owner = store.resolve(token) if token else None
        reused = owner is not None
        if owner is None:
            token = store.issue()
            owner = store.resolve(token)
        if owner is None:
            raise sqlite3.DatabaseError("Anonymous identity issuance failed")
        server_id = store.server_id()
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(503, "Anonymous identity storage unavailable") from exc
    if reused:
        response.status_code = 200
    return AnonymousCredential(token=token, server_id=server_id, identity_id=owner, reused=reused)
