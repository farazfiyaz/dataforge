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

## Ideas / backlog
- Executor sandbox: `getattr`/`type` in builtins allow classic `__subclasses__`
  escapes, and there is no execution timeout; an infinite loop hangs the kernel.
  Consider running code in a subprocess with a timeout.
- `/api/upload` reads the whole file into memory with no size limit.
