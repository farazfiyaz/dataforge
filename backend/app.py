# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os

from routers import upload, chat, execute, autoanalyze, agent

HOST = "127.0.0.1"
PORT = 8000

# The API runs arbitrary Python (/api/execute, /api/agent), so it must only be
# reachable by the DataForge window itself — never by other websites open in
# the user's browser, and never by other machines on the network.
ALLOWED_ORIGINS = [f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"]

app = FastAPI(title="DataForge", version="0.1.0")


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

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Loopback only — binding 0.0.0.0 would expose the code-execution API to
    # the whole LAN. Auto-reload is a dev convenience (it spawns a file-watcher
    # process), so it's opt-in via DATAFORGE_DEV=1.
    uvicorn.run("app:app", host=HOST, port=PORT, reload=os.environ.get("DATAFORGE_DEV") == "1")
