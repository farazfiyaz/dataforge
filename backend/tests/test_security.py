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


# ── Other programs on the same computer ──

def test_api_needs_the_launch_token(anonymous_client):
    res = anonymous_client.post("/api/execute/", json={"code": "print('pwned')"},
                                headers={"Origin": APP_ORIGIN})
    assert res.status_code == 401
    assert "pwned" not in res.text
    assert anonymous_client.get("/api/settings/").status_code == 401


def test_wrong_token_is_rejected(anonymous_client):
    res = anonymous_client.post("/api/execute/", json={"code": "print(1)"},
                                headers={"X-DataForge-Token": "guess"})
    assert res.status_code == 401


def test_token_header_also_works(anonymous_client):
    res = anonymous_client.post("/api/execute/", json={"code": "print(1)"},
                                headers={"X-DataForge-Token": app_module.TOKEN})
    assert res.status_code == 200


def test_page_needs_the_token_and_the_link_swaps_it_for_a_cookie(anonymous_client):
    assert anonymous_client.get("/").status_code == 401
    assert anonymous_client.get("/?t=wrong").status_code == 401
    res = anonymous_client.get(f"/?t={app_module.TOKEN}", follow_redirects=False)
    assert res.status_code == 303 and res.headers["location"] == "/"
    cookie = res.headers["set-cookie"]
    assert "httponly" in cookie.lower() and "samesite=strict" in cookie.lower()
    assert anonymous_client.get("/").status_code == 200        # cookie now set


def test_health_proves_the_server_knows_the_token(client):
    import hashlib, hmac
    proof = client.get("/health?nonce=abc").json()["proof"]
    assert proof == hmac.new(app_module.TOKEN.encode(), b"abc", hashlib.sha256).hexdigest()


def test_api_docs_are_off():
    assert not app_module.DEV
    c = TestClient(app_module.app, base_url=APP_ORIGIN)
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404


def test_page_has_a_strict_content_security_policy(client):
    csp = client.get("/").headers["content-security-policy"]
    assert "script-src 'self' 'sha256-" in csp
    assert "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]
    assert "connect-src 'self'" in csp and "object-src 'none'" in csp
