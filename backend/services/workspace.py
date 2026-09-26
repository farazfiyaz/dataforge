# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Workspace folder access for the agent.

The user picks a folder; the agent can then list and load data files from it
by itself ("load sales.csv and compare it with the uploaded data"). Access is
strictly confined to the chosen folder — path traversal outside it is blocked.
"""

import os
from pathlib import Path

import pandas as pd

DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"}

_workspace_root: Path | None = None


def set_workspace(path: str) -> dict:
    """Set the workspace root folder. Returns info about it."""
    global _workspace_root
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise ValueError(f"not a folder: {path}")
    _workspace_root = p
    return {"path": str(p), "files": len(_list(p))}


def get_workspace() -> str | None:
    return str(_workspace_root) if _workspace_root else None


def _resolve(relpath: str) -> Path:
    """Resolve a path inside the workspace, blocking escapes."""
    if _workspace_root is None:
        raise PermissionError(
            "No workspace folder is set. Ask the user to set one "
            "with the 📁 Workspace button."
        )
    p = (_workspace_root / relpath).resolve()
    if _workspace_root != p and _workspace_root not in p.parents:
        raise PermissionError(f"path escapes the workspace folder: {relpath}")
    return p


def _list(folder: Path) -> list[dict]:
    out = []
    for entry in sorted(folder.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            out.append({"name": entry.name + "/", "type": "folder"})
        elif entry.suffix.lower() in DATA_EXTENSIONS:
            out.append({"name": entry.name, "type": "file",
                        "size_kb": round(entry.stat().st_size / 1024, 1)})
    return out


def list_files(subdir: str = "") -> list[dict]:
    """List data files and folders in the workspace (agent-callable)."""
    folder = _resolve(subdir)
    if not folder.is_dir():
        raise ValueError(f"not a folder: {subdir}")
    return _list(folder)


def load_file(relpath: str, **kwargs) -> pd.DataFrame:
    """Load a data file from the workspace as a DataFrame (agent-callable)."""
    p = _resolve(relpath)
    if not p.is_file():
        raise FileNotFoundError(f"no such file in workspace: {relpath}")
    ext = p.suffix.lower()
    if ext not in DATA_EXTENSIONS:
        raise ValueError(f"unsupported file type '{ext}' — allowed: {sorted(DATA_EXTENSIONS)}")
    if ext in (".csv", ".tsv"):
        return pd.read_csv(p, sep="\t" if ext == ".tsv" else ",", **kwargs)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p, **kwargs)
    if ext == ".json":
        return pd.read_json(p, **kwargs)
    return pd.read_parquet(p, **kwargs)
