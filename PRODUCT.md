# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Two audiences, weighted equally (confirmed):
- **Working analysts / students** who load a CSV or Excel file and spend long sessions asking questions, reading charts, tables and generated Python.
- **Evaluators of the author's work** (recruiters, reviewers) who see the app in demos, screenshots and the README and judge it in seconds.

## Product Purpose

DataForge is a desktop AI data scientist that runs entirely on the user's machine. The user attaches a dataset, gets an instant profile, and asks questions in plain language; an agent writes Python, runs it in a sandboxed kernel, reads its own output and errors, fixes itself, and streams every step live. Success: the user gets a correct chart, table or answer without leaving the app or sending data anywhere.

## Positioning

Fully local: FastAPI backend + Ollama (`qwen2.5-coder` 1.5B/3B/7B chosen by available RAM) + Electron shell. No cloud, no API keys, the data never leaves the machine. The agent loop is visible, not a black box: each step's code, result, and self-correction is shown.

## Operating Context

- Electron window, default 1280x800, minimum 900x600. The frontend is a single `frontend/index.html` served by the backend at localhost:8000.
- First launch may download a model; the UI shows a status banner driven by `window.dataforge.onModelStatus`.
- Workflows: attach file → automatic profile (rows, columns, duplicates, missing) → suggested plots / Auto Analyze (clean + generate charts, download cleaned CSV) → conversation in Agent, Explain, Code, or Plot mode. Code blocks can be copied or re-run. Chats are grouped into projects and stored in localStorage.
- A workspace folder can be set so the agent can list and load files from disk.

## Capabilities and Constraints

- All existing behaviour, endpoints (`/api/upload`, `/api/agent` SSE, `/api/chat`, `/api/execute`, `/api/autoanalyze`, `/api/agent/workspace`) and the localStorage schema (`dataforge_db`) must remain unchanged.
- Charts arrive as server-rendered matplotlib PNGs; tables as row arrays.
- Offline-first: no remote fonts, scripts, or assets can be assumed at runtime.
- Theme follows the OS setting (light and dark both required).

## Brand Commitments

- Name: DataForge. Author credit "© 2026 Mohammed Farazuddin" and AGPL-3.0 licence header must remain.
- The local-first privacy promise is the core message.

## Evidence on Hand

- `../demo_churn.csv` is a real sample dataset usable for demos.
- No testimonials, user counts, or benchmarks exist; none may be invented.

## Product Principles

1. The data is the protagonist: charts, tables and numbers get the space and the contrast; chrome recedes.
2. Show the work: the agent's steps, code, and self-corrections are legible and inspectable.
3. Local and honest: never imply cloud, speed, or accuracy claims the product does not make.
4. Demo-ready at a glance, comfortable over hours.
