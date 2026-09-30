# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Where DataForge saves what it makes, and a button to open that folder."""

from fastapi import APIRouter, HTTPException

from services import outputs

router = APIRouter()


@router.get("/")
async def info():
    return {"folder": str(outputs.OUTPUT_ROOT)}


@router.post("/open")
async def open_folder():
    try:
        return {"folder": outputs.open_folder()}
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Couldn't open the folder: {e}")
