# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time

from routers import upload, chat, execute, autoanalyze, agent, settings, history as history_router
from services import history
from services.vault import VaultUnavailable

HOST = "127.0.0.1"
# Electron picks a free port per launch, so another program sitting on a fixed
# port can't pose as the backend; 8000 when run by hand.
PORT = int(os.environ.get("DATAFORGE_PORT", "8000"))
DEV = os.environ.get("DATAFORGE_DEV") == "1"

# Origin and Host checks (below) keep websites out, but any other program on
# this computer could still call the API. So every /api request must carry
# this per-launch secret: Electron
# generates it and sets it as a cookie on its own window; run by hand, the
# server makes one and prints a link containing it.
TOKEN = os.environ.get("DATAFORGE_TOKEN") or secrets.token_urlsafe(32)
SESSION_COOKIE = "df_session"
TOKEN_HEADER = "x-dataforge-token"

# The API runs arbitrary Python (/api/execute, /api/agent), so it must only be
# reachable by the DataForge window itself — never by other websites open in
# the user's browser, and never by other machines on the network.
ALLOWED_ORIGINS = [f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"]

# No interactive API docs outside development: they are a ready-made console
# for the code-execution endpoints.
app = FastAPI(title="DataForge", version="0.1.0",
              docs_url="/docs" if DEV else None, redoc_url=None,
              openapi_url="/openapi.json" if DEV else None)


def _authorized(request: Request) -> bool:
    supplied = request.cookies.get(SESSION_COOKIE) or request.headers.get(TOKEN_HEADER) or ""
    return hmac.compare_digest(supplied.encode(), TOKEN.encode())


@app.middleware("http")
async def require_session_token(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method != "OPTIONS" and not _authorized(request):
        return JSONResponse({"detail": "Not authorized. Open DataForge from the app."}, status_code=401)
    return await call_next(request)


@app.middleware("http")
async def reject_foreign_origins(request: Request, call_next):
    """
    CORS alone doesn't stop cross-site *simple* requests (e.g. a multipart
    form POST from a malicious page) — the browser still sends them, it just
    hides the response. Refuse any request that declares a foreign Origin.
    """
    origin = request.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Blocks DNS-rebinding: a hostile domain resolved to 127.0.0.1 would otherwise
# look same-origin to the browser.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

app.include_router(upload.router,      prefix="/api/upload",      tags=["upload"])
app.include_router(chat.router,        prefix="/api/chat",        tags=["chat"])
app.include_router(execute.router,     prefix="/api/execute",     tags=["execute"])
app.include_router(autoanalyze.router, prefix="/api/autoanalyze", tags=["autoanalyze"])
app.include_router(agent.router,       prefix="/api/agent",       tags=["agent"])
app.include_router(settings.router,    prefix="/api/settings",    tags=["settings"])
app.include_router(history_router.router, prefix="/api/history", tags=["history"])

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _content_security_policy(html: str) -> str:
    """
    Only this page's own inline script may run (pinned by hash), and the page
    may only talk to this server. If a dataset value or model reply ever
    slipped past escaping into the DOM, it still couldn't run script or send
    data anywhere.
    """
    hashes = " ".join(
        f"'sha256-{base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()}'"
        for body in re.findall(r"<script>(.*?)</script>", html, re.S)
    )
    return ("default-src 'self'; "
            f"script-src 'self' {hashes}; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


@app.get("/")
def root(request: Request, t: str | None = None):
    if t is not None:
        # The printed link: swap the token in the URL for a cookie, then drop it from the address bar
        if not hmac.compare_digest(t.encode(), TOKEN.encode()):
            return HTMLResponse("This DataForge link has expired. Restart DataForge and use the new link.", status_code=401)
        res = RedirectResponse("/", status_code=303)
        res.set_cookie(SESSION_COOKIE, TOKEN, httponly=True, samesite="strict")
        return res
    if not _authorized(request):
        return HTMLResponse("Open DataForge from the app (or the link printed when the server started).", status_code=401)
    with open(os.path.join(FRONTEND_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("<script>", _history_block() + "\n<script>", 1)
    return HTMLResponse(html, headers={"Content-Security-Policy": _content_security_policy(html),
                                       "Cache-Control": "no-store"})


def _history_block() -> str:
    """
    The decrypted history, inlined as inert JSON so the page has it the moment
    its script starts (it used to read localStorage synchronously). Not run as
    script, so CSP doesn't apply; '<' is escaped so data can't close the tag.
    """
    try:
        saved = history.load()
    except history.HistoryUnreadable:
        # Don't let the next save overwrite it: set it aside and start fresh
        history.HISTORY_PATH.replace(history.HISTORY_PATH.with_name(f"history.unreadable-{int(time.time())}.bin"))
        saved = None
    except VaultUnavailable:
        saved = None
    payload = json.dumps(saved).replace("<", "\\u003c")
    return f'<script type="application/json" id="df-history">{payload}</script>'


@app.get("/health")
def health(nonce: str | None = None):
    """With ?nonce=..., also proves this server knows the launch token, so Electron can tell it's ours."""
    if nonce is None:
        return {"status": "ok"}
    proof = hmac.new(TOKEN.encode(), nonce.encode(), hashlib.sha256).hexdigest()
    return {"status": "ok", "proof": proof}


if __name__ == "__main__":
    import uvicorn
    if "DATAFORGE_TOKEN" not in os.environ:
        print(f"\n  DataForge is running. Open: http://127.0.0.1:{PORT}/?t={TOKEN}\n", flush=True)
    # uvicorn imports "app:app" afresh, which would mint a different token
    os.environ["DATAFORGE_TOKEN"] = TOKEN
    # Loopback only — binding 0.0.0.0 would expose the code-execution API to
    # the whole LAN. Auto-reload is a dev convenience (it spawns a file-watcher
    # process), so it's opt-in via DATAFORGE_DEV=1.
    uvicorn.run("app:app", host=HOST, port=PORT, reload=DEV)
