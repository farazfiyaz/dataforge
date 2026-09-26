# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.eda import run_eda
from services.datastore import put_dataset
from services.recommend import recommend
from services.loader import read_table
import uuid

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a CSV or Excel file, run EDA, and return a profile report.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    contents = await file.read()
    try:
        df = read_table(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    # Store server-side so chat/agent requests reference it by id
    # instead of re-uploading the raw CSV every message
    dataset_id = uuid.uuid4().hex
    put_dataset(dataset_id, df)

    profile = run_eda(df)
    profile["recommendations"] = recommend(df)
    return {"filename": file.filename, "profile": profile, "dataset_id": dataset_id}
