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
### 2026-09-28 — European CSVs: dates stay dates, not huge integers
Semicolon files with decimal commas are read with `thousands="."`, and pandas
then drops every dot, so German-Excel-style dates silently became numbers
(`01.02.2024` → `1022024`, `15.03.2024` → `15032024`) and codes like `1.10`
became `110`. The agent would happily average them. The loader now keeps a
parsed column only if every value is a well-formed European number (dots only
between groups of three digits); other columns keep their original text.
### 2026-09-28 — A bare `except:` can no longer swallow the time limit
The sandbox's `ExecutionTimeout` is a BaseException, so `except Exception:`
can't catch it, but a bare `except:` can. `try: while True: pass / except: print('caught')`
ran for 0.5s, printed "caught" and reported success. `while True: try: ... except: pass`
hung past the limit (killed after 20s with a 0.5s limit): once the trace hook
raises, CPython removes it. The sandbox now compiles bare `except:` as
`except Exception:`, which still catches every ordinary error.
### 2026-09-28 — Upload no longer crashes on True/False columns or Excel year headers
Two common file shapes returned a 500 error:
- **Any True/False column** (e.g. `paid,True`): pandas counts bool as numeric, so
  the profiler took IQR quantiles and numpy raised "boolean subtract is not
  supported". Both upload and Auto Analyze failed. Bool columns are now profiled
  as categorical (True/False counts).
- **Excel sheets with year headers** (`2023`, `2024`): `read_excel` returns int
  column names, and Auto Analyze crashed joining them into the cleaning report
  and again when labelling the missing-values plot. Excel headers are now text,
  the same as a CSV with the same header.
### 2026-09-29 — Upload size limit
`/api/upload` and `/api/autoanalyze` read the whole file into memory with
`await file.read()`, and parsing needs several times that again (decoded text,
then the DataFrame). A multi-gigabyte CSV could exhaust a laptop's RAM and
freeze the app with no message. Uploads are now read in 1 MB chunks and stopped
at 200 MB (`DATAFORGE_MAX_UPLOAD_MB` to change it) with a 413 that names the
file, the limit, and what to do. The existing frontend shows that message as-is.
### 2026-09-29 — train_model no longer crashes on 1/2 labels or rare classes
`train_model` decided "binary" from the test split and called `roc_curve`
without a positive label. Two common cases crashed after printing metrics:
- a two-class target coded `1`/`2` → `ValueError: y_true takes value in {1, 2}`;
- a three-class target whose rare class landed only in training: the test
  split looked binary, so it printed a meaningless `ROC AUC: 0.379` from the
  wrong probability column (accuracy was 1.000), then crashed the same way.
Binary is now judged from the model's own classes, `roc_curve` gets
`pos_label`, and the confusion matrix is built over every class seen, so the
matrix and its tick labels always match. New `tests/test_model_report.py`
(both tests fail without the fix).
### 2026-09-29 — Auto clean handles CSVs written with ", " separators
In a file like `id, plan, age` every header and cell starts with a space. Auto
clean trimmed cells only at the end, after the other steps had already run, so
the downloaded CSV still had ` plan`/` age` headers. Blank cells came out as
empty text instead of being filled. `age` stayed text during the median fill, so
its gap was never filled, and a row differing only by `"basic "` vs `"basic"`
wasn't removed as a duplicate. Now headers and cells are trimmed first,
whitespace-only cells count as missing, and numeric text is converted before
nulls and duplicates are handled. A header is not trimmed if that would give it
the same name as another column. New `tests/test_cleaner_padded_csv.py`: 5 of
its 6 tests failed before the fix.
### 2026-09-29 — Don't crash on infinite values
A CSV with `inf` in it (a spreadsheet's `#DIV/0!` saved as text, or a ratio
column) made `/api/upload` and `/api/autoanalyze` answer 500: `inf` isn't valid
JSON, so the profile's sample rows couldn't be sent, and matplotlib refused to
draw histograms over a range reaching `inf`. The same went for code results:
`df.a / df.b` with a zero in `b` put `inf` in the result table, which broke
`/api/execute` and wrote `Infinity` into the agent's stream, which the browser
can't parse. Tables and samples now show `"inf"` / `"-inf"` as text
(`eda.table_records`), and Auto Analyze's histograms, box plots and heatmap
leave infinite values out. The data itself is unchanged.
### 2026-09-30 — Tell the agent True/False columns are booleans, not text
The schema described a True/False column as `text, 2 values: 'False', 'True'`.
Those cells are real booleans (dtype `bool`, or `object` when some are blank), so
the model wrote `(s == 'True').mean()`, got 0.0 for every group, and answered
"there are no active customers". Boolean columns are now described as
`boolean True/False, N True` with a `.mean()` hint, and all-blank columns as
`empty (all N values missing)` instead of `numeric (float64), range nan to nan`.
Live: "What percentage of customers are active, per plan?" went from 0%/0% to
32.63%/37.63%, matching pandas exactly.
### 2026-09-30 — Clear error for files with headers but no rows
A CSV or Excel file with only a header row (an export whose filter matched
nothing, or an empty template) made upload and Auto Analyze answer a bare 500.
The profile's null percentages were `0/0 = NaN`, which isn't valid JSON, and
the browser then failed to parse the error body. `read_table` now rejects it
with a 400: "The file has column headers (3) but no data rows." A workbook with
a blank sheet says "The file is empty." The agent's workspace `load_file` gets
the same message.
### 2026-09-30 — Workspace `load_file` accepts the agent's pandas options
Sniffing only ran when `load_file` got no options. With any keyword, CSVs went to a
bare `pd.read_csv` with a forced `sep`, so `load_file("x.csv", sep=";")` raised
"got multiple values for keyword argument 'sep'", `delimiter=";"` raised "Specified
a sep and a delimiter", and `usecols=`/`nrows=` on an Excel-on-Windows (cp1252) file
raised `UnicodeDecodeError`. Now the caller's `sep`/`delimiter`/`encoding` win, and
whatever they leave out is still sniffed. Tests in `test_workspace_load_file.py`.
### 2026-09-30 — Explain Ollama crashes and out-of-memory in words
Only "not running", "too slow" and "model missing" were translated. Other
failures leaked raw text: a 500 showed `Ollama returned an error (500):
{"error":"model requires more system memory (5.6 GiB) than is available (3.1 GiB)"}`,
a model process that crashed or was killed mid-answer (`ReadError`,
`RemoteProtocolError`) became "Ollama error: …" (a 502 in Explain/Code/Plot
modes), and a non-JSON reply raised a JSON decode error. Now: out of memory says
so and says to close other apps, other errors show Ollama's own message without
JSON, dropped connections say Ollama stopped answering, and a garbled reply is an
OllamaError (503). Tests in `test_llm_errors.py`.
### 2026-10-01 — Keep follow-up suggestions after the agent reshapes `df`
Suggestions after an agent answer are built from the session's `df` as the agent
left it. If a step added a list column (`df["words"] = df["text"].str.split()`)
or a duplicate column name (`pd.concat([df, df[["plan"]]], axis=1)`), every rule
raised (`unhashable type: 'list'`, "truth value of a Series is ambiguous"), the
error was swallowed, and the suggestions silently vanished. `recommend()` now
keeps the first column of each name and skips columns whose values can't be counted.
### 2026-10-01 — Result tables with bins, dates or grouped columns no longer break runs
The sandbox turned result DataFrames into table rows with `fillna("")`, which caused two bugs:
- **A binned column poisoned the session.** `pd.cut` leaves gaps for out-of-range
  values, and `fillna("")` raises on a categorical with gaps. `df` is the fallback
  table, so after `df['bucket'] = pd.cut(...)` *every* later run reported a pandas
  traceback, even `print(df['x'].mean())` that had printed the right answer.
- **Dates killed the agent's stream.** Timestamps stayed as objects, and the
  agent's event stream died with "Object of type Timestamp is not JSON serializable".
Tables are now built from plain values (dates, intervals and tuple column names
as text, blanks for gaps), and a table that still can't be built is left out
without failing the run. Tests: `tests/test_result_tables.py` (3 of 4 fail on the old code).
### 2026-10-01 — train_model on binned (categorical) targets
"Predict the tenure group" leads the agent to `pd.cut` a column and train on it.
`train_model` crashed with sklearn's "Unknown label type: unknown" because the
labels were `Interval` objects. Named bins (`labels=["new", "mid", "loyal"]`)
trained, but the confusion matrix sorted them alphabetically, and a categorical
with more than 10 numeric categories was treated as regression. Categorical
targets are now always classification, encoded in the categories' own order.
### 2026-10-01 — Describe grouped bars and pie charts to the model
"Churn by plan" is usually drawn as `groupby(['plan','churn']).size().unstack().plot.bar()`
or `sns.countplot(x='plan', hue='churn')`. Those bars sit beside their tick, so the
model was told `bars 'no': -0.125=308, 0.875=133, 1.875=304`, and seaborn's series
weren't named at all. Pie charts produced no description. Now a series offset by
a constant from the ticks takes the tick labels (histograms are unaffected),
unnamed seaborn series take their legend names, and pies list each slice's share:
`bars 'no': basic=308, enterprise=133, pro=304; bars 'yes': basic=420, …` and
`pie, 3 slices: basic 48.5%, pro 36.3%, enterprise 15.1%`.
### 2026-10-02 — Chart styles no longer leak between sandbox runs
Matplotlib's style is global to the process, and the sandbox never reset it. The
agent prompt asks for `plt.style.use('dark_background')`, so after the first
agent chart every later chart was dark. That included plain `/api/execute` runs
and other datasets' sessions. Any `rcParams['figure.figsize'] = (30, 20)` or
`sns.set_theme(...)` from generated code also stuck until restart. Each run now
executes inside `matplotlib.rc_context()`. A style still applies to the charts
of the run that sets it, and is restored afterwards, including after errors and
timeouts. Trade-off: a style set in one agent step no longer carries over to the
next step of the same session.
### 2026-10-02 — Auto Analyze: no ID charts, and charts for True/False columns
Auto Analyze's "Top Values" charts used the first six text columns as they came.
A text identifier (`order_id` = "A-1001", an email column) became a chart of ten
bars that are all 1 high, and it used up one of the six slots. Numeric IDs were
already skipped. True/False columns, often the outcome itself (`churned`), got no
chart at all, since they count as neither text nor numeric. Now ID columns and
all-unique columns are skipped, bool columns are charted, charts follow the
file's column order, and the six-chart cap counts only charts that are drawn.

## Ideas / backlog
- Executor sandbox: `getattr`/`type` in builtins allow classic `__subclasses__`
  escapes. Real isolation needs a subprocess kernel (which would also allow a
  hard memory limit and stopping long C-level operations).
- A bare `except:` in generated code can still catch the timeout. (Probed 2026-09-28:
  `while True: try/except: pass` still stops, because the tracer re-raises inside the handler.)
- `dd.mm.yyyy` text dates from European files could be parsed with `dayfirst=True`;
  `find_date_column` currently parses them month-first.
- Ragged CSV rows (one line with an extra field) fail the whole upload with a C tokenizer error.
- `/api/upload` reads the whole file into memory with no size limit.
- A bare `except:` in generated code can still catch the timeout.
- Recommendations could offer "one-click run" instead of just filling the input.
- Box plots with several boxes are summarised as one combined span; per-box medians/quartiles would be better.
- Live: for "churn rate per plan as a bar chart" the model ran `sns.barplot(y='churn', estimator=...)` on the yes/no text column, giving an empty chart (the table was right). Tell it to compute the rate first and plot that.
- Violin/strip/swarm plots aren't described beyond their raw collections.
- Uploads (`/api/upload`, agent `df`) still keep ", "-padded headers and cells
  (` plan`, `" 34"` as text); the loader could read with `skipinitialspace=True`.
- A numeric column with any `inf` gets `mean`/`max` of null in the profile; stats over the
  finite values plus an `inf_count` would be clearer (and Auto clean could treat inf as missing).
