# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Projects and chats, stored encrypted by services/history.py."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import history
from services.vault import VaultUnavailable

router = APIRouter()


class HistoryBody(BaseModel):
    db: dict


@router.put("/")
def save_history(body: HistoryBody):
    try:
        history.save(body.db)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except VaultUnavailable as e:
        raise HTTPException(status_code=503, detail=f"No OS credential store to hold the encryption key: {e}")
    return {"ok": True}
