# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Oversized uploads get a clear 413 instead of being read wholly into memory."""
import io

import pytest

from services import upload_limit

ROW = b"1,2.5,hello\n"


def _csv(size_bytes: int) -> bytes:
    return b"a,b,c\n" + ROW * (size_bytes // len(ROW))


@pytest.fixture
def one_mb_limit(monkeypatch):
    monkeypatch.setattr(upload_limit, "MAX_UPLOAD_MB", 1)


@pytest.mark.parametrize("endpoint", ["/api/upload/", "/api/autoanalyze/"])
def test_oversized_upload_is_rejected_with_413(client, one_mb_limit, endpoint):
    data = _csv(3 * 1024 * 1024 // 2)
    res = client.post(endpoint, files={"file": ("big.csv", io.BytesIO(data), "text/csv")})
    assert res.status_code == 413
    detail = res.json()["detail"]
    assert "big.csv" in detail and "1 MB" in detail


def test_upload_under_the_limit_still_loads(client, one_mb_limit):
    data = _csv(900 * 1024)
    res = client.post("/api/upload/", files={"file": ("ok.csv", io.BytesIO(data), "text/csv")})
    assert res.status_code == 200
    assert res.json()["profile"]["shape"]["rows"] == data.count(b"\n") - 1
