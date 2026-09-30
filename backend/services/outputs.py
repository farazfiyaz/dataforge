# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Everything DataForge makes is also saved as ordinary files on this computer:
Documents/DataForge/<date>/ holds charts (PNG), result tables and cleaned data (CSV).

The user may delete any of it at any time. Nothing here keeps state about
saved files, the folder is recreated on the next save, and a failed save
(deleted folder, full disk, permissions) is skipped, never an error.
"""

import base64
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_ROOT = Path(os.environ.get("DATAFORGE_OUTPUT_DIR", Path.home() / "Documents" / "DataForge"))


def _day_folder() -> Path:
    folder = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)   # recreated if the user deleted it
    return folder


def _unique(folder: Path, stem: str, suffix: str) -> Path:
    stem = re.sub(r"[^\w.-]+", "_", stem).strip("_")[:60] or "output"
    stamp = datetime.now().strftime("%H%M%S")
    path = folder / f"{stamp}_{stem}{suffix}"
    n = 2
    while path.exists():
        path = folder / f"{stamp}_{stem}_{n}{suffix}"
        n += 1
    return path


def _relative(path: Path) -> str:
    """Shown to the user, e.g. 2026-09-30/143012_chart.png."""
    return path.relative_to(OUTPUT_ROOT).as_posix()


def save_png(b64: str, stem: str = "chart") -> str | None:
    try:
        path = _unique(_day_folder(), stem, ".png")
        path.write_bytes(base64.b64decode(b64))
        return _relative(path)
    except (OSError, ValueError):
        return None


def save_csv(data: pd.DataFrame | list[dict], stem: str = "table") -> str | None:
    try:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        path = _unique(_day_folder(), stem, ".csv")
        df.to_csv(path, index=False)
        return _relative(path)
    except (OSError, ValueError):
        return None


def open_folder() -> str:
    """Open the outputs folder in the file explorer (recreating it if deleted)."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(OUTPUT_ROOT)  # noqa: S606 — a fixed local folder, not user input
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(OUTPUT_ROOT)])
    else:
        subprocess.Popen(["xdg-open", str(OUTPUT_ROOT)])
    return str(OUTPUT_ROOT)
