# DataForge

**AI-powered data science tool that runs 100% locally — your data never leaves your machine.**

Upload a CSV or Excel file and talk to your data: DataForge profiles it, cleans it, plots it, and answers questions by writing and executing Python — autonomously, like an AI data scientist in a notebook.

Copyright © 2026 Mohammed Farazuddin · [farazfiyaz2@gmail.com](mailto:farazfiyaz2@gmail.com) · Licensed under [AGPL-3.0](LICENSE)

## Highlights

**🤖 Agentic analysis loop** — the core feature. Ask a question and DataForge plans, writes code, runs it, reads the output (including its own errors), fixes itself, and iterates until it has an answer — streaming every step live to the UI. Conversation memory means you can say *"now color it by category"* or *"that's wrong, fix it"* and it builds on what it already did.

**🧭 Recommended next steps** — not sure what to ask? DataForge reads the data and suggests concrete next moves: predict the outcome column it detects, compare it across segments, fix missing values, explore the strongest correlation, check outliers, plot trends over time. Suggestions appear after upload, after every answer, and as *"Stuck? Try one of these"* when the agent runs out of steps. One click puts the prompt in the box.

**📊 Instant EDA** — upload a file and get an automatic profile: shape, dtypes, nulls, duplicates, distributions, and suggested plots.

**🔬 Auto Analyze** — one click cleans the dataset and generates a full set of charts.

**📈 Any plot type** — matplotlib, seaborn, numpy, scipy, and sklearn are available in the sandboxed executor: heatmaps, violin plots, pairplots, regression plots, whatever the question needs.

**📂 Real-world files** — CSVs saved by Excel on Windows (cp1252), semicolon-separated files with decimal commas (European Excel exports), tab-separated files, `.xlsx` and `.xls` all load correctly.

**🗄️ Big-data aware** — datasets are parsed once and stored server-side (referenced by id, never re-uploaded per message), and the agent is trained to sample before plotting and never dump full DataFrames.

**🔒 Fully local** — FastAPI backend + Ollama LLM + Electron desktop shell. No cloud, no API keys, no data leaving your machine. The backend listens on `127.0.0.1` only and rejects requests from other websites, so nothing else on your machine or network can reach the code-execution API.

**🛟 Works with small local models** — `qwen2.5-coder` often writes tool calls as text instead of native calls; the agent recovers them, runs notebook-style code (`df.head()` shows its output), stops runaway code after 2 minutes, and explains Ollama problems in plain terms ("run `ollama pull …`").

**💻 Adapts to your hardware** — on first launch, DataForge checks how much RAM the machine has and picks a matching `qwen2.5-coder` size (1.5B / 3B / 7B), downloading it automatically if it isn't installed yet. No manual config needed to run on a low-spec laptop vs. a workstation.

## Architecture

```
Electron shell
└── frontend/index.html ── chat UI, live agent steps, EDA cards, charts
    └── FastAPI backend (localhost:8000)
        ├── /api/upload      → parse file, run EDA, store dataset server-side
        ├── /api/agent       → agentic loop (SSE): LLM ⇄ sandboxed Python, self-correcting
        ├── /api/chat        → single-shot explain/code/plot modes
        ├── /api/execute     → sandboxed code execution (persistent notebook-style kernel)
        └── /api/autoanalyze → auto-clean + auto-plot
            └── Ollama (qwen2.5-coder:1.5b/3b/7b, picked by available RAM) — local LLM with native tool calling
```

## Quick start

Prerequisites: [Python 3.10+](https://python.org), [Node.js](https://nodejs.org), [Ollama](https://ollama.com)

```bash
git clone https://github.com/farazfiyaz/dataforge.git
cd dataforge
start.bat          # Windows — installs deps on first run, launches the app
```

DataForge picks a `qwen2.5-coder` size based on the machine's RAM (≥12GB → 7B, ≥6GB → 3B, else 1.5B) and pulls it automatically on first launch if it isn't already installed — no manual `ollama pull` needed. To pre-download it yourself instead (e.g. on a slow connection), run `ollama pull qwen2.5-coder:7b` (or `:3b` / `:1.5b`) before starting the app.

## Development

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q tests        # LLM calls are mocked — no Ollama needed
DATAFORGE_DEV=1 python app.py    # auto-reload while editing
```

CI runs the suite on every pull request against both pandas 2.2 and pandas 3. [`IMPROVEMENTS.md`](IMPROVEMENTS.md) logs each change and keeps the backlog of ideas.

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE). If you use this code, your project must credit the author and be open-sourced under the same license.
