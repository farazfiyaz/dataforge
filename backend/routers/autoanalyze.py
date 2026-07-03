"""
DataForge — Auto Analyze Router
Copyright (C) 2026 Mohammed Farazuddin

POST /api/autoanalyze/
  - Accepts a CSV or Excel upload
  - Auto-cleans the data
  - Generates a full plot suite
  - Returns cleaned CSV (base64) + plots + cleaning report
"""

import io
import base64
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.eda import run_eda
from services.cleaner import auto_clean
from services.autoplot import generate_all_plots

router = APIRouter()

@router.post("/")
async def autoanalyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported.")

    contents = await file.read()
    try:
        df_original = pd.read_csv(io.BytesIO(contents)) if ext == "csv" else pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    # 1. EDA on original
    profile = run_eda(df_original)

    # 2. Auto-clean
    df_cleaned, cleaning_report = auto_clean(df_original)

    # 3. Generate plots
    plots = generate_all_plots(df_original, df_cleaned)

    # 4. Encode cleaned CSV as base64 for download
    csv_bytes = df_cleaned.to_csv(index=False).encode()
    cleaned_csv_b64 = base64.b64encode(csv_bytes).decode()

    return {
        "filename": file.filename,
        "profile": profile,
        "cleaning_report": cleaning_report,
        "cleaned_shape": {"rows": len(df_cleaned), "cols": len(df_cleaned.columns)},
        "cleaned_csv_b64": cleaned_csv_b64,
        "plots": plots,
    }
