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

### 2026-09-27 — pandas 3 fixes: Auto Analyze cleaning and train_model
The venv resolves `pandas>=2.2` to pandas 3, which broke two things silently:
- **Auto clean didn't fill nulls.** Under Copy-on-Write, `df[col].fillna(..., inplace=True)`
  is a no-op, so the report said "filled 31 values with median" while the nulls stayed.
- **`train_model` crashed on text targets** (e.g. `churn` yes/no in the demo data):
  labels are dtype `str`, not `object`, so they were never encoded.
Also made text-column selection work on both pandas versions, and stopped whitespace
stripping from turning missing values into the literal text `"nan"`.

### 2026-09-27 — Keep ID columns out of charts and models
`customer_id`-style columns were suggested as charts ("histogram of customer_id"),
drawn in Auto Analyze histograms/box plots/heatmaps, and fed to `train_model` as
features. The profile now flags `is_id` per column and a `date_column`. Charts and
models skip IDs (models only when features aren't chosen explicitly, and the
dropped IDs are reported). "Line chart over time" is only suggested when a date
column exists. ID/date detection now lives in `services/eda.py`, shared with recommendations.

### 2026-09-27 — Sandbox time limit and non-blocking execution
`run_code` ran synchronously inside the async handlers, so the whole server
froze while agent code ran, and a `while True:` from the model hung DataForge
until restart. Sandbox code now runs on a dedicated worker thread (serialized,
since pyplot and the stdout swap are global), and a trace hook limited to
sandbox-compiled frames stops it after `EXEC_TIMEOUT_S` (120s). The exception
is a BaseException, so `except Exception:` in generated code can't swallow it.
Overhead is about 10% on pandas-heavy code and about 5x on pure-Python loops.

### 2026-09-27 — Load messy real-world files
New `services/loader.py` (`read_table`) is used by upload, Auto Analyze and
workspace `load_file`. It handles cp1252/latin-1 CSVs (what Excel on Windows
writes), UTF-8 BOMs, `;`/tab/`|` delimiters, and European decimal commas with `.`
thousands separators. It also gives clear 400 errors for empty or unsupported files.
Added `xlrd` so `.xls`, which the file picker always accepted, actually loads.
Auto Analyze's cleaned download of an Excel upload is now named `.csv`.

### 2026-09-27 — CI on every PR
GitHub Actions runs the backend suite on Python 3.11 + pandas 2.2 and on
Python 3.13 + pandas 3, and syntax-checks the frontend's inline script.

### 2026-09-27 — Actionable Ollama errors; proper chat roles for Explain/Code/Plot
Ollama failures now say what to do: "Start it with `ollama serve`", "Run
`ollama pull <model>`", or that the model is still loading. They used to show raw
exceptions like "All connection attempts failed". Explain/Code/Plot modes now use
`/api/chat` with real system/user roles. The old `/api/generate` call hand-wrote
`<|system|>` tags that Qwen doesn't use, so the system prompt arrived as user text.

### 2026-09-27 — Make the agent work with qwen2.5-coder (text tool calls)
Tested live: `qwen2.5-coder:7b`, the default model, never emits native tool calls
through Ollama (0 of 4 probes). It writes calls as text, so the agent showed raw
JSON as its "final answer" and ran nothing. The agent now recovers:
- call JSON that's bare, in `<tool_call>` tags, or in any fenced block (even after prose);
- helpers "called" as tools (`train_model`, `list_files`, `load_file`), translated to code;
- ```python blocks when nothing has run yet, or when the reply says "please run the above code".
Unknown tools get a corrective message instead of a bare "unknown tool".
Live result: "Predict churn" now trains the model and answers with real metrics.

### 2026-09-27 — Notebook-style output; real feature importances
The sandbox now echoes a trailing expression the way a notebook cell does:
`df.head()` or `df.groupby(...).mean()` alone used to print nothing, so the model
saw empty output and repeated itself. A trailing DataFrame also becomes the result
table. `train_model` prints its top-5 feature importances as text. Before, the
model could only guess them from a chart it can't see, and it did guess, wrongly.
Also tightened ID detection: an unnamed column now needs to be an exact +1 row
counter, since a sorted unique feature was being dropped as an "ID".
Live: "churn rate per plan" answered 57.69/41.41/44.22%, matching pandas exactly.

### 2026-09-27 — Describe charts to the model
The agent can't see the PNGs it draws; it was only told "chart rendered", so a
live run answered "the plan with the highest churn rate is the one with the
highest bar". `services/chartsummary.py` reads the plotted data back out of each
figure (bar labels and heights, histogram peak, line/scatter ranges, heatmap
range) and passes it to the model as `chart_data`. The same question now answers
"basic, 57.7% (pro 44.2%, enterprise 41.4%)", matching pandas exactly. Also
recovers a call object that has prose before or after it.

### 2026-09-27 — Don't hide DataFrame columns from the model
Live: asked for the most extreme `support_tickets` values, the model printed the
rows, but at pandas' default 80-char width the middle columns collapsed to `...`.
`support_tickets` was hidden, so the model invented values (4, 5; the real max is
7). The sandbox now uses a 250-char display width and up to 40 columns.

## Ideas / backlog
- Executor sandbox: `getattr`/`type` in builtins allow classic `__subclasses__`
  escapes. Real isolation needs a subprocess kernel (which would also allow a
  hard memory limit and stopping long C-level operations).
- A bare `except:` in generated code can still catch the timeout.
- `/api/upload` reads the whole file into memory with no size limit.
- Recommendations could offer "one-click run" instead of just filling the input.
- Chart descriptions don't cover box plots yet (median/quartiles/outliers). The model can't read them from the image.
- Heatmap descriptions could list the strongest off-diagonal pairs, not just the value range.
