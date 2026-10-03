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

### 2026-09-27 — Describe box plots and heatmap pairs to the model
Box plots were described as a pile of 2-point line segments. They're now
recognised by their structure and summarised as whisker span plus outlier values.
Square labelled heatmaps (correlation matrices) name their three strongest
off-diagonal pairs. Live: the outlier question now answers "most values between
0 and 3, outliers at 4, 5, 6, 7", which is exactly right.

### 2026-09-27 — Agent retries after errors; schema with real values
Live two-turn test: turn 1 drew a wrong chart (`.mean()` on the yes/no `churn`
text column, reported "0% / 0% / 22%"), and turn 2 hit the same error, then gave
up with "Let's update the code…" because its fix, a ```python block, was taken
as the final answer. Now:
- a code block right after a failed step is run as the fix;
- the system prompt's schema is built from the data (`churn: text, 2 values: 'no', 'yes'`,
  numeric ranges). The frontend only sent `churn(str)`;
- one prompt line explains yes/no rates.
Live rerun: both turns correct (57.7/44.2/41.4%), and turn 2 sorted horizontally as asked.

### 2026-09-30 — Model settings: bigger local models, optional cloud (Claude)
A **Model** button chooses who answers:
- **This computer (default):** any model installed in Ollama, with suggestions for
  stronger ones to download (`qwen2.5-coder:14b` / `:32b`).
- **Cloud (opt-in):** Claude via the Anthropic API (Opus 5 by default, or Sonnet 5 or
  Haiku 4.5) with the user's own key. It warns what is sent, and the input hint and
  welcome text change so it's always visible that cloud mode is on. The key is kept in
  `~/.dataforge/settings.json` and never returned to the page.
`services/claude.py` translates the agent's Ollama-style conversation to the Messages API
and back, echoing Claude's raw blocks (thinking included) unchanged. Opus 5 uses
server-side refusal fallbacks. Errors (bad key, rate limit, offline) become plain messages.
### 2026-09-28 — "No file attached" message and a Stop button
- With no file attached, general questions ("what is a p-value?", "write code for
  a moving average") are answered normally, in any mode, with no code execution.
  Questions that need the user's data get "There's no file attached, and this
  question needs your data" plus an Attach button; the model signals this with a
  marker. There's a separate message when a restored chat's file is gone after a restart.
  Live, qwen2.5-coder:7b classified 7 of 8 test questions as expected.
- While a reply is in progress, the send button becomes Stop (Esc also works).
  Stopping aborts the request; the backend cancels the run and drops the unfinished
  turn from the agent's memory, so the next question starts clean.

### 2026-09-30 — Opt-in web search for general questions
A "Search the web" switch in Model settings (off by default). Local model: a `web_search`
tool backed by DuckDuckGo's HTML results (no key; only the search words leave the machine),
shown in the chat as "Searched the web for …" with links. Cloud: Claude's own web search
tool (`web_search_20260209`, or `_20250305` on Haiku 4.5), with `pause_turn` resumed and a
Sources list under the answer. The no-file prompt now says questions about software,
facts and news aren't "your data", and with search on it includes today's date.
Live (qwen2.5-coder:7b): "latest pandas version" searched and answered 3.0.6; "what is a
p-value" answered without searching; "churn rate in my data" asked for a file. The
7B model still searched "2022 World Cup" for "most recent World Cup" despite the date.
### 2026-09-30 — Download buttons, tables in answers, auto-save to Documents\DataForge
- Every chart has **Download PNG**; every result table (and table in an answer) has **Download CSV**.
- Answers render Markdown tables as real tables, and the model is told to use tables
  when listing values per group.
- Charts, tables produced by the code, and Auto Analyze's cleaned data and charts are
  saved to `Documents/DataForge/<date>/`, each shown with a "Saved to … · Open folder"
  note. Only tables made in that step are saved (not the unchanged dataset every step).
  The files are the user's to delete: the folder is recreated on the next save, and a
  failed save is skipped, never an error. Folder: `DATAFORGE_OUTPUT_DIR`.
### 2026-09-28 — Describe each box in a grouped box plot
"Compare monthly_bill across plans" as a box plot was described to the model as
one combined span ("boxes and whiskers span 3.04 to 124.5"), so it couldn't say
which plan was higher. Box plots (seaborn, pandas `df.boxplot(by=...)`,
`plt.boxplot`, vertical or horizontal) are now summarised per box: median,
middle 50%, whiskers and that box's outliers, named by category. Seaborn `hue`
boxes are named "category / hue" from the legend. Single boxes also get their
median. On the demo data every median and quartile matches pandas exactly.

## Ideas / backlog
- Executor sandbox: `getattr`/`type` in builtins allow classic `__subclasses__`
  escapes. Real isolation needs a subprocess kernel (which would also allow a
  hard memory limit and stopping long C-level operations).
- A bare `except:` in generated code can still catch the timeout.
- `/api/upload` reads the whole file into memory with no size limit.
- Recommendations could offer "one-click run" instead of just filling the input.
- Box plots with several boxes are summarised as one combined span; per-box medians/quartiles would be better.
- Live: for "churn rate per plan as a bar chart" the model ran `sns.barplot(y='churn', estimator=...)` on the yes/no text column, giving an empty chart (the table was right). Tell it to compute the rate first and plot that.
- Violin/strip/swarm plots aren't described beyond their raw collections.
