# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Size cap for uploaded files.

`await file.read()` pulls the whole upload into memory, and parsing then needs
several times that again (decoded text, then the DataFrame). A multi-gigabyte
CSV would exhaust a laptop's RAM and freeze the app with no message. Reading in
chunks lets us stop at the cap and tell the user instead.
"""

import os

from fastapi import HTTPException, UploadFile

MAX_UPLOAD_MB = int(os.environ.get("DATAFORGE_MAX_UPLOAD_MB", "200"))
_CHUNK = 1024 * 1024


async def read_upload(file: UploadFile, max_mb: int | None = None) -> bytes:
    """Read an upload's bytes, raising 413 once it passes `max_mb` megabytes."""
    limit = (MAX_UPLOAD_MB if max_mb is None else max_mb) * 1024 * 1024
    chunks, total = [], 0
    while chunk := await file.read(_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"{file.filename} is larger than the {limit // (1024 * 1024)} MB "
                       "upload limit. Load a sample or a subset of the columns, or set "
                       "DATAFORGE_MAX_UPLOAD_MB to raise the limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)
