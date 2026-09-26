# DataForge

**AI-powered data science tool that runs 100% locally — your data never leaves your machine.**

Upload a CSV or Excel file and talk to your data: DataForge profiles it, cleans it, plots it, and answers questions by writing and executing Python — autonomously, like an AI data scientist in a notebook.

Copyright © 2026 Mohammed Farazuddin · [farazfiyaz2@gmail.com](mailto:farazfiyaz2@gmail.com) · Licensed under [AGPL-3.0](LICENSE)

## Highlights

**🤖 Agentic analysis loop** — the core feature. Ask a question and DataForge plans, writes code, runs it, reads the output (including its own errors), fixes itself, and iterates until it has an answer — streaming every step live to the UI. Conversation memory means you can say *"now color it by category"* or *"that's wrong, fix it"* and it builds on what it already did.

**📊 Instant EDA** — upload a file and get an automatic profile: shape, dtypes, nulls, duplicates, distributions, and suggested plots.

**🔬 Auto Analyze** — one click cleans the dataset and generates a full set of charts.

**📈 Any plot type** — matplotlib, seaborn, numpy, scipy, and sklearn are available in the sandboxed executor: heatmaps, violin plots, pairplots, regression plots, whatever the question needs.

**🗄️ Big-data aware** — datasets are parsed once and stored server-side (referenced by id, never re-uploaded per message), and the agent is trained to sample before plotting and never dump full DataFrames.

**🔒 Fully local** — FastAPI backend + Ollama LLM + Electron desktop shell. No cloud, no API keys, no data leaving your machine.

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

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE). If you use this code, your project must credit the author and be open-sourced under the same license.
