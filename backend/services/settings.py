# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Which model answers: the local Ollama model (the default) or, if the user
opts in, a cloud model (Claude via the Anthropic API).

Saved as JSON in the user's home folder, never inside the repo. The API key
is never sent back to the page; only its last four characters are.
"""

import json
import os
from pathlib import Path

SETTINGS_PATH = Path(os.environ.get("DATAFORGE_SETTINGS", Path.home() / ".dataforge" / "settings.json"))

PROVIDERS = ("local", "claude")
# Current Claude models, most capable first. Opus 5 is the default.
CLAUDE_MODELS = {
    "claude-opus-5": "Claude Opus 5 — most capable",
    "claude-sonnet-5": "Claude Sonnet 5 — faster, lower cost",
    "claude-haiku-4-5": "Claude Haiku 4.5 — fastest, lowest cost",
}
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
# Electron picks a model that fits the machine's RAM and passes it in
DEFAULT_LOCAL_MODEL = os.environ.get("DATAFORGE_MODEL", "qwen2.5-coder:7b")

_DEFAULTS = {"provider": "local", "local_model": "", "claude_model": DEFAULT_CLAUDE_MODEL, "claude_api_key": "",
             "web_search": False}   # opt-in: sends only the search query to the web


def load() -> dict:
    try:
        saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {**_DEFAULTS, **{k: v for k, v in saved.items() if k in _DEFAULTS}}


def save(changes: dict) -> dict:
    """Validate and persist `changes`; raises ValueError with a user-facing message."""
    current = load()
    updated = {**current, **{k: v for k, v in changes.items() if k in _DEFAULTS and v is not None}}
    if updated["provider"] not in PROVIDERS:
        raise ValueError(f"Unknown provider '{updated['provider']}'.")
    if updated["claude_model"] not in CLAUDE_MODELS:
        raise ValueError(f"Unknown Claude model '{updated['claude_model']}'.")
    updated["claude_api_key"] = updated["claude_api_key"].strip()
    updated["web_search"] = bool(updated["web_search"])
    if updated["provider"] == "claude" and not claude_api_key(updated):
        raise ValueError("Add an Anthropic API key to use the cloud model.")
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return updated


def provider() -> str:
    return load()["provider"]


def local_model() -> str:
    return load()["local_model"] or DEFAULT_LOCAL_MODEL


def claude_model() -> str:
    return load()["claude_model"]


def web_search() -> bool:
    return bool(load()["web_search"])


def claude_api_key(settings: dict | None = None) -> str:
    """The saved key, or ANTHROPIC_API_KEY from the environment."""
    settings = settings or load()
    return settings["claude_api_key"] or os.environ.get("ANTHROPIC_API_KEY", "")


def public_view() -> dict:
    """What the page may see: everything except the key itself."""
    s = load()
    key = claude_api_key(s)
    return {
        "provider": s["provider"],
        "local_model": local_model(),
        "claude_model": s["claude_model"],
        "claude_models": [{"id": k, "label": v} for k, v in CLAUDE_MODELS.items()],
        "has_api_key": bool(key),
        "api_key_hint": f"…{key[-4:]}" if len(key) >= 8 else "",
        "api_key_from_env": bool(key) and not s["claude_api_key"],
        "web_search": bool(s["web_search"]),
    }
