# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""The code-execution API must only be reachable by the DataForge window."""
import io

from fastapi.testclient import TestClient

import app as app_module
from conftest import APP_ORIGIN

EVIL = "https://evil.example"
CSV = b"a,b\n1,2\n3,4\n"


def test_same_origin_requests_work(client):
    assert client.get("/health").json() == {"status": "ok"}
    res = client.post("/api/execute/", json={"code": "x = 1 + 1\nprint(x)"},
                      headers={"Origin": APP_ORIGIN})
    assert res.status_code == 200
    assert res.json()["stdout"].strip() == "2"


def test_foreign_origin_cannot_execute_code(client):
    res = client.post("/api/execute/", json={"code": "print('pwned')"},
                      headers={"Origin": EVIL})
    assert res.status_code == 403
    assert "pwned" not in res.text


def test_foreign_origin_cannot_upload_via_simple_form_post(client):
    # multipart POSTs skip CORS preflight, so CORS alone wouldn't stop this
    res = client.post("/api/upload/", files={"file": ("x.csv", io.BytesIO(CSV), "text/csv")},
                      headers={"Origin": EVIL})
    assert res.status_code == 403


def test_cors_preflight_rejects_foreign_origin(client):
    res = client.options("/api/execute/", headers={
        "Origin": EVIL,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert res.headers.get("access-control-allow-origin") != EVIL
    assert res.headers.get("access-control-allow-origin") != "*"


def test_cors_preflight_allows_app_origin(client):
    res = client.options("/api/execute/", headers={
        "Origin": APP_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == APP_ORIGIN


def test_dns_rebinding_host_is_rejected():
    rebound = TestClient(app_module.app, base_url="http://attacker.example:8000")
    assert rebound.get("/health").status_code == 400


def test_server_binds_loopback_only():
    assert app_module.HOST == "127.0.0.1"
