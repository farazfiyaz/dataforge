@echo off
title DataForge
echo.
echo  ██████╗  █████╗ ████████╗ █████╗ ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
echo  ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
echo  ██║  ██║███████║   ██║   ███████║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗
echo  ██║  ██║██╔══██║   ██║   ██╔══██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
echo  ██████╔╝██║  ██║   ██║   ██║  ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
echo  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
echo.
echo  AI-powered data science tool  ^|  Copyright 2026 Mohammed Farazuddin
echo  ─────────────────────────────────────────────────────────────
echo.

REM ── Check Python ──
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)

REM ── Check Node.js ──
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Node.js not found. Install from https://nodejs.org
    pause & exit /b 1
)

REM ── Check Ollama ──
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [WARN] Ollama not found. AI chat features won't work.
    echo         Install from https://ollama.com  then run: ollama pull qwen2.5-coder:7b
    echo.
)

REM ── Install Python deps (first run only) ──
if not exist "backend\venv" (
    echo  [SETUP] Creating Python virtual environment...
    python -m venv backend\venv
    echo  [SETUP] Installing Python dependencies...
    backend\venv\Scripts\pip install -r backend\requirements.txt --quiet
    echo  [SETUP] Python deps installed.
    echo.
)

REM ── Install Node/Electron deps (first run only) ──
if not exist "node_modules" (
    echo  [SETUP] Installing Electron dependencies...
    npm install --quiet
    echo  [SETUP] Electron deps installed.
    echo.
)

REM ── Launch DataForge as a desktop app ──
echo  [START] Launching DataForge desktop app...
npm start
pause
