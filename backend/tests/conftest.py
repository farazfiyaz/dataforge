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

from app import app  # noqa: E402

APP_ORIGIN = "http://localhost:8000"


@pytest.fixture
def client():
    return TestClient(app, base_url=APP_ORIGIN)
