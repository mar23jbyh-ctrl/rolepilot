from __future__ import annotations

import threading
import sqlite3
from typing import Optional

from fastapi import Depends, Header, HTTPException, Query

from app.service import InterviewService
from app.security import identity_store

_services: dict[str, InterviewService] = {}
_lock = threading.Lock()


def resolve_owner(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_owner_id: Optional[str] = Header(default=None, alias="X-Owner-Id"),
    owner: Optional[str] = Query(default=None),
) -> str:
    """Derive the canonical owner exclusively from a server-issued capability."""
    if x_owner_id is not None or owner is not None:
        raise HTTPException(400, "Client-supplied owner identity is not supported")
    parts = (authorization or "").split()
    resolved = None
    if len(parts) == 2 and parts[0].lower() == "bearer":
        try:
            resolved = identity_store().resolve(parts[1])
        except (sqlite3.Error, OSError) as exc:
            raise HTTPException(503, "Anonymous identity storage unavailable") from exc
    if resolved is None:
        raise HTTPException(401, "Anonymous bearer credential required", headers={"WWW-Authenticate": "Bearer"})
    return resolved


def get_service(owner: str) -> InterviewService:
    """One InterviewService (graph + checkpointer + store) per owner."""
    with _lock:
        service = _services.get(owner)
        if service is None:
            service = InterviewService(owner=owner)
            _services[owner] = service
        return service


def service_dep(owner: str = Depends(resolve_owner)) -> InterviewService:
    return get_service(owner)
