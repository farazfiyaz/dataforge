# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.executor import run_code

router = APIRouter()

class ExecuteRequest(BaseModel):
    code: str
    csv_data: str = ""        # raw CSV fallback (small files only)
    dataset_id: str = ""      # server-side dataset reference (big files)

@router.post("/")
async def execute(req: ExecuteRequest):
    """
    Run user/LLM-generated Python code in a sandboxed environment.
    Returns stdout, a table result (JSON), and/or a base64-encoded chart image.
    """
    result = run_code(req.code, req.csv_data, dataset_id=req.dataset_id or None)
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
    return result
