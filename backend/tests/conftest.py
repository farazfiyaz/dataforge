# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
import os
import sys

import pytest
from fastapi.testclient import TestClient

# The backend uses top-level imports (`from routers import ...`), as when run
# via `python app.py` — mirror that here.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, SESSION_COOKIE, TOKEN  # noqa: E402

APP_ORIGIN = "http://localhost:8000"


@pytest.fixture
def client():
    """Signed in, like the DataForge window (Electron sets this cookie on it)."""
    return TestClient(app, base_url=APP_ORIGIN, cookies={SESSION_COOKIE: TOKEN})


@pytest.fixture
def anonymous_client():
    """Any other program on the computer: right host and origin, no token."""
    return TestClient(app, base_url=APP_ORIGIN)


@pytest.fixture(autouse=True)
def temp_settings(tmp_path, monkeypatch):
    """Never read or write the user's real ~/.dataforge/settings.json in tests."""
    from services import settings
    monkeypatch.setattr(settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def no_workspace():
    """The workspace folder is process-global; don't let one test's leak into another."""
    from services import workspace
    workspace._workspace_root = None
    yield
    workspace._workspace_root = None
