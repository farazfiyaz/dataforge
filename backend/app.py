# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from routers import upload, chat, execute, autoanalyze, agent

app = FastAPI(title="DataForge", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
