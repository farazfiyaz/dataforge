# DataForge improvement log

A running log of incremental improvements (one PR each), plus a backlog of ideas.
Run the backend tests with:

```
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q tests
```

## Log

### 2026-09-27 — Lock the code-execution API to the DataForge window
The backend listened on `0.0.0.0` with CORS `*`, and `/api/execute` runs Python.
Any website open in the user's browser, or anyone on the same network, could
send it code. Now it binds `127.0.0.1` only, allows only the app's own origin,
rejects foreign `Origin` headers (which also covers multipart posts that skip
CORS preflight), and blocks DNS rebinding with a trusted-host check. Auto-reload
is opt-in (`DATAFORGE_DEV=1`). Added the first pytest suite (`backend/tests/`).

### 2026-09-27 — Recommended next steps ("stuck? try these")
New `services/recommend.py` suggests concrete next steps from the data itself:
predict a detected outcome column (flagging class imbalance and ID columns to
exclude), compare the outcome across segments, handle missing values, remove
duplicates, explore the strongest correlation, investigate outliers, and plot
trends over time. They're rule-based, so they work even with the 1.5B model. They're shown
after upload, after every agent answer (skipping ones already asked), and as
"Stuck? Try one of these instead" when the agent hits its step limit. Clicking
one fills the prompt in Agent mode.

## Ideas / backlog
- Executor sandbox: `getattr`/`type` in builtins allow classic `__subclasses__`
  escapes, and there is no execution timeout; an infinite loop hangs the kernel.
  Consider running code in a subprocess with a timeout.
- `/api/upload` reads the whole file into memory with no size limit.
- "Suggested charts" chips offer to plot ID columns (e.g. histogram of `customer_id`).
- Recommendations could offer "one-click run" instead of just filling the input.
