# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Model settings: local Ollama model (default) or the opt-in cloud model."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import settings
from services.llm import installed_models

router = APIRouter()

# Stronger local models worth suggesting when they aren't installed yet,
# with rough RAM needs, so users know a better option exists
SUGGESTED_LOCAL_MODELS = [
    {"name": "qwen2.5-coder:14b", "note": "better answers, needs ~16 GB RAM"},
    {"name": "qwen2.5-coder:32b", "note": "best local quality, needs ~32 GB RAM"},
]


class SettingsUpdate(BaseModel):
    provider: Optional[str] = None
    local_model: Optional[str] = None
    claude_model: Optional[str] = None
    claude_api_key: Optional[str] = None   # "" clears the saved key


@router.get("/")
async def get_settings():
    view = settings.public_view()
    installed = await installed_models()
    view["local_models"] = installed
    view["suggested_local_models"] = [m for m in SUGGESTED_LOCAL_MODELS if m["name"] not in installed]
    return view


@router.post("/")
async def update_settings(req: SettingsUpdate):
    try:
        settings.save(req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await get_settings()
