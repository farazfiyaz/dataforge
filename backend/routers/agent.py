# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Agentic loop — the thing that makes DataForge work like Claude.

Instead of one LLM call → one answer, we loop:
    LLM → wants to run code? → execute → feed result (or traceback) back → LLM again
until the model gives a final text answer or we hit MAX_ITERATIONS.

Errors are NOT shown to the user as failures — they're sent back to the model,
which fixes its own code and retries (self-correction).

Each step is streamed to the frontend as a Server-Sent Event so the user can
watch the agent work in real time.
"""

import json
import re
import uuid
from collections import OrderedDict
from typing import Optional

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.llm import OllamaError, chat_ollama
from services.datastore import get_dataset
from services.executor import run_code_async, reset_session, peek_session_df
from services.recommend import recommend
from services.workspace import set_workspace, get_workspace

router = APIRouter()

MAX_ITERATIONS = 10
MAX_HISTORY = 40   # messages kept per conversation (excluding system prompt)
MAX_CONVERSATIONS = 20   # bounded like services/executor.py's session cache — caps memory on long-running low-RAM machines

# Conversation history per session — this is what makes follow-ups like
# "that's wrong, fix it" or "make the bars horizontal" actually work:
# the model sees everything it did before, and the kernel still has its variables.
_CONVERSATIONS: "OrderedDict[str, list[dict]]" = OrderedDict()

# Tool schema the model sees (Ollama / OpenAI style)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code against the loaded dataset. "
                "Available: `df` (pandas DataFrame), `pd`, `np` (numpy), `plt` (matplotlib), "
                "`sns` (seaborn), `train_model` (one-call ML pipeline with metrics + charts), "
                "`list_files()` / `load_file(name)` (browse and load files from the user's workspace folder). "
                "You may also import: scipy, sklearn, plotly, math, statistics, "
                "datetime, re, json, itertools, collections. "
                "Variables PERSIST between calls, like a notebook. "
                "Each call is stopped after a 2-minute time limit, so prefer vectorized code. "
                "Use print() to see values. Returns stdout, an optional table, and any error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are DataForge, an autonomous AI data scientist.

You have a `run_python` tool. The user's dataset is loaded as a pandas DataFrame `df`.
Work step by step like a data scientist in a notebook:
1. Inspect the data first if you're unsure of its shape (df.head(), df.dtypes, etc.).
2. Run small steps, look at the output, then decide what to do next.
3. If your code errors, read the traceback and fix it — do not apologize, just retry.
4. Variables persist between run_python calls AND between user messages —
   if the user says "fix it" or "change the color", build on what you already did.

Plotting — any chart type is fine (matplotlib `plt`, seaborn `sns`, numpy `np`):
- Always plt.style.use('dark_background'); add titles and axis labels.
- Heatmaps, boxplots, violin plots, pairplots, regression plots — use seaborn when it's cleaner.

Machine learning — a `train_model` helper is preloaded:
    model = train_model(df, target="churn")              # auto-detects classification/regression
    model = train_model(df, target="price", model="gb")  # "rf" (default) | "gb" | "linear"
It handles preprocessing (impute/encode/scale), train/test split, prints metrics
(accuracy/F1/ROC AUC or R²/MAE/RMSE), and draws confusion matrix / ROC / feature
importance charts automatically. It returns a fitted sklearn Pipeline — reuse it
in later steps for .predict() on new data. Use it whenever the user asks to train,
predict, or model something; interpret the metrics in plain English in your answer.

Workspace files — if the user has set a workspace folder, you can browse it yourself:
    list_files()                      # what's in the folder (csv/xlsx/json/parquet/tsv)
    df2 = load_file("sales.csv")      # load any file as a DataFrame
Use these when the user mentions files by name, asks what data is available, or wants
to combine multiple files. If they raise "No workspace folder is set", tell the user
to set one with the 📁 Workspace button.

Text columns can't be averaged. For the rate of a yes/no column use
(df['churn'] == 'yes').mean(), or df.groupby('plan')['churn'].apply(lambda s: (s == 'yes').mean()).

Big datasets — be memory- and output-conscious:
- NEVER print an entire large DataFrame; use .head(), .describe(), .value_counts().head(20).
- For scatter plots with >5000 rows, plot a random sample: df.sample(5000, random_state=42).
- Prefer vectorized pandas/numpy operations over Python loops.

When you have fully answered the user's question, reply with plain text (no tool call):
a concise summary of what you found, including concrete numbers.

Dataset schema:
{schema}
"""


class AgentRequest(BaseModel):
    message: str
    context: Optional[str] = None    # dataset schema string
    csv_data: str = ""               # raw CSV fallback (small files only)
    dataset_id: Optional[str] = None # server-side dataset reference (big files)
    session_id: Optional[str] = None


_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_FENCE = re.compile(r"```([\w+-]*)[ \t]*\n?(.*?)```", re.DOTALL)
_INLINE_CALL = re.compile(r'\{\s*"name"\s*:')
# "Please run the above code…": the model handing work back instead of doing it
_HANDOFF = re.compile(
    r"\b(run|execute)\b[^.\n]{0,40}\b(code|snippet|script|above|below|following)\b"
    r"|\byou (can|could|should) (run|execute)\b",
    re.IGNORECASE,
)


def _as_call(raw: str) -> list[dict] | None:
    """Parse `raw` as one or more call-shaped objects ({"name", "arguments"}), else None."""
    try:
        obj = json.loads(raw.strip(), strict=False)   # strict=False: raw newlines inside code strings
    except json.JSONDecodeError:
        return None
    calls = []
    for item in obj if isinstance(obj, list) else [obj]:
        if not (isinstance(item, dict) and isinstance(item.get("name"), str)
                and ("arguments" in item or "parameters" in item)):
            return None
        args = item.get("arguments", item.get("parameters"))
        calls.append({"function": {"name": item["name"], "arguments": args}})
    return calls


def _text_tool_calls(content: str, allow_code_blocks: bool = False) -> list[dict]:
    """
    Recover tool calls a model wrote as text instead of native `tool_calls`.

    qwen2.5-coder:7b (the default model) never emits native tool calls through
    Ollama. It writes `{"name": "run_python", "arguments": {...}}` as content:
    bare, in <tool_call> tags, or in a ```json/```python fence, often after a
    sentence of preamble. Taken literally, the agent would show that JSON as its
    "final answer" and never run anything.

    - Call-shaped JSON (with "name" AND "arguments") is recognised in tags, in
      any fenced block, or as the whole reply. Unknown names are kept; the loop
      translates helpers (train_model…) or answers with a corrective error.
    - Plain ```python blocks become a call when the model clearly meant them
      to run: nothing has run yet this request (`allow_code_blocks`), or the
      reply hands them to the user ("please run the above code"). Otherwise
      they're examples inside a final answer and are left alone.
    """
    text = (content or "").strip()
    if not text:
        return []
    calls = [c for raw in _TOOL_CALL_TAG.findall(text) for c in (_as_call(raw) or [])]
    if calls:
        return calls
    fences = _FENCE.findall(text)
    calls = [c for _, body in fences for c in (_as_call(body) or [])]
    if calls:
        return calls
    if text.startswith("{") and (whole := _as_call(text)) is not None:
        return whole
    # A call object with prose before or after it: decode just that object
    decoder = json.JSONDecoder(strict=False)
    for m in _INLINE_CALL.finditer(text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if (found := _as_call(json.dumps(obj))) is not None:
            calls.extend(found)
    if calls:
        return calls
    if allow_code_blocks or _HANDOFF.search(text):
        code = "\n\n".join(body.strip() for lang, body in fences if lang.lower() in ("python", "py"))
        if code:
            return [{"function": {"name": "run_python", "arguments": {"code": code}}}]
    return []


def _normalize_calls(calls: list[dict]) -> list[dict]:
    """Decode string arguments and turn helper "tools" into run_python calls."""
    out = []
    for call in calls:
        fn = call.get("function", {})
        name, args = fn.get("name"), fn.get("arguments") or {}
        if isinstance(args, str):          # some models return JSON strings
            try:
                args = json.loads(args, strict=False)
            except json.JSONDecodeError:
                args = {"code": args}
        if name != "run_python" and (code := _helper_call_as_code(name, args)):
            name, args = "run_python", {"code": code}
        out.append({"function": {"name": name, "arguments": args}})
    return out


# Sandbox helpers that models "call" as if they were tools. Rather than bounce
# the call back (small models often just repeat it), run the equivalent code.
def _helper_call_as_code(name: str, args: dict) -> str | None:
    if not isinstance(args, dict):
        return None
    args = dict(args)
    if name == "train_model":
        data = args.pop("df", "df")
        data = data if isinstance(data, str) and data.strip() else "df"   # may be an expression, e.g. df.drop(...)
        kwargs = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"model = train_model({data}{', ' + kwargs if kwargs else ''})"
    if name == "list_files":
        return f"print(list_files({args.get('subdir', '')!r}))"
    if name == "load_file":
        target = args.get("relpath") or args.get("name") or args.get("path") or args.get("filename")
        return f"df2 = load_file({target!r})\nprint(df2.shape)\nprint(df2.head())" if target else None
    return None


def _schema(dataset_id: Optional[str]) -> Optional[str]:
    """
    Describe the dataset for the system prompt, from the data itself: types
    plus value hints. The frontend only sent `churn(str)`, so the model didn't
    know churn was yes/no and kept calling .mean() on it.
    """
    df = get_dataset(dataset_id) if dataset_id else None
    if df is None:
        return None
    lines = [f"{len(df):,} rows × {len(df.columns)} columns:"]
    for col in df.columns[:60]:
        s = df[col]
        nulls = f", {int(s.isna().sum()):,} missing" if s.isna().any() else ""
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            lines.append(f"- {col}: numeric ({s.dtype}), range {s.min():g} to {s.max():g}{nulls}")
        elif pd.api.types.is_datetime64_any_dtype(s):
            lines.append(f"- {col}: datetime, {s.min()} to {s.max()}{nulls}")
        else:
            n = s.nunique()
            if n <= 10:
                values = ", ".join(repr(v) for v in sorted(s.dropna().astype(str).unique()))
                lines.append(f"- {col}: text, {n} values: {values}{nulls}")
            else:
                sample = ", ".join(repr(v) for v in s.dropna().astype(str).unique()[:3])
                lines.append(f"- {col}: text, {n:,} distinct values, e.g. {sample}{nulls}")
    if len(df.columns) > 60:
        lines.append(f"- … and {len(df.columns) - 60} more columns")
    return "\n".join(lines)


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _tool_result_for_llm(result: dict) -> str:
    """Compact, token-friendly version of an execution result for the model."""
    if result.get("error"):
        return json.dumps({"error": result["error"][-2000:]})
    payload = {"stdout": (result.get("stdout") or "")[-3000:]}
    if result.get("table") is not None:
        payload["table_preview"] = result["table"][:10]
        payload["table_rows_returned"] = len(result["table"])
    if result.get("charts"):
        n = len(result["charts"])
        payload["chart"] = f"{n} chart(s) rendered and shown to the user" if n != 1 else "rendered and shown to the user"
        # what's actually plotted, since the model can't see the image; lets it
        # say "basic is highest (0.58)" instead of "the tallest bar"
        described = [s for s in result.get("chart_summaries") or [] if s]
        if described:
            payload["chart_data"] = described
    return json.dumps(payload, default=str)


def _recommendations_event(session_id: str, dataset_id: Optional[str],
                           messages: list[dict], stuck: bool) -> Optional[str]:
    """
    Suggest what to try next, based on the data as the agent has left it
    (the session's transformed `df`, falling back to the uploaded dataset).
    Skips anything the user has already asked. None if there's no data.
    """
    df = peek_session_df(session_id)
    if df is None and dataset_id:
        df = get_dataset(dataset_id)
    if df is None:
        return None
    asked = [m["content"] for m in messages if m.get("role") == "user"]
    try:
        recs = recommend(df, asked=asked, limit=3)
    except Exception:
        return None   # suggestions are a nice-to-have — never break the answer over them
    return _sse("recommendations", {"items": recs, "stuck": stuck}) if recs else None


def _save_conversation(session_id: str, messages: list[dict]) -> None:
    """Persist history, trimmed to system prompt + last MAX_HISTORY messages."""
    _CONVERSATIONS[session_id] = [messages[0]] + messages[1:][-MAX_HISTORY:]
    _CONVERSATIONS.move_to_end(session_id)
    while len(_CONVERSATIONS) > MAX_CONVERSATIONS:
        _CONVERSATIONS.popitem(last=False)


@router.post("/")
async def agent(req: AgentRequest):
    """Run the agentic loop, streaming steps as SSE."""
    session_id = req.session_id or uuid.uuid4().hex

    # Continue the existing conversation if there is one — this is what lets
    # the user say "fix it" / "now color by category" and have it just work
    messages = _CONVERSATIONS.get(session_id)
    if messages is not None:
        _CONVERSATIONS.move_to_end(session_id)
    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                schema=_schema(req.dataset_id) or req.context or "unknown")},
        ]
    messages.append({"role": "user", "content": req.message})

    async def stream():
        yield _sse("start", {"session_id": session_id})

        # ```python blocks in a reply are code to run (not examples) until
        # something has run, and again right after a failed step: then they're
        # the model's fix, and treating them as a final answer ends the retry
        expect_code = True
        for step in range(1, MAX_ITERATIONS + 1):
            try:
                assistant = await chat_ollama(messages, tools=TOOLS)
            except OllamaError as e:          # already says what to do about it
                yield _sse("error", {"message": str(e)})
                return
            except Exception as e:
                yield _sse("error", {"message": f"Ollama error: {e}"})
                return

            tool_calls = (assistant.get("tool_calls")
                          or _text_tool_calls(assistant.get("content", ""), allow_code_blocks=expect_code))
            if tool_calls:
                # store the normalized call so the model's history shows correct usage
                tool_calls = _normalize_calls(tool_calls)
                assistant = {"role": "assistant", "content": "", "tool_calls": tool_calls}
            messages.append(assistant)

            # No tool call → model is done; its content is the final answer
            if not tool_calls:
                _save_conversation(session_id, messages)
                yield _sse("final", {"content": assistant.get("content", ""), "steps": step})
                recs = _recommendations_event(session_id, req.dataset_id, messages, stuck=False)
                if recs:
                    yield recs
                return

            for call in tool_calls:
                fn = call.get("function", {})
                if fn.get("name") != "run_python":
                    # tell the model how to fix it — small models retry sensibly
                    # when told what to do, but loop on a bare "unknown tool"
                    messages.append({"role": "tool", "content": json.dumps({"error": (
                        f"There is no tool named '{fn.get('name')}'. The only tool is run_python: "
                        "put Python code in its `code` argument. Helpers such as train_model, "
                        "list_files and load_file are functions you call inside that code, e.g. "
                        "run_python(code=\"model = train_model(df, target='churn')\")."
                    )})})
                    continue

                args = fn.get("arguments") or {}
                code = args.get("code", "") if isinstance(args, dict) else str(args)

                yield _sse("code", {"step": step, "code": code})

                result = await run_code_async(code, req.csv_data, session_id=session_id,
                                              dataset_id=req.dataset_id)

                # Stream the human-facing result (full chart, full table)
                yield _sse("result", {
                    "step": step,
                    "stdout": result.get("stdout"),
                    "table": result.get("table"),
                    "charts": result.get("charts"),
                    "error": result.get("error"),
                    "retrying": bool(result.get("error")),
                })

                # Feed a compact version back to the model — including errors,
                # which is what lets it self-correct
                messages.append({"role": "tool", "content": _tool_result_for_llm(result)})
                expect_code = bool(result.get("error"))

        _save_conversation(session_id, messages)
        yield _sse("final", {
            "content": "Reached the maximum number of steps without a final answer. "
                       "Here is what was done so far — try narrowing the question.",
            "steps": MAX_ITERATIONS,
        })
        # The agent got stuck — offer concrete, answerable questions instead
        recs = _recommendations_event(session_id, req.dataset_id, messages, stuck=True)
        if recs:
            yield recs

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class WorkspaceRequest(BaseModel):
    path: str = ""


@router.post("/workspace")
async def workspace(req: WorkspaceRequest):
    """Set (or query, with empty path) the workspace folder the agent may access."""
    if not req.path:
        return {"path": get_workspace()}
    try:
        return set_workspace(req.path)
    except ValueError as e:
        return {"error": str(e)}


@router.post("/reset")
async def reset(req: AgentRequest):
    """Clear a session's variables and history (e.g. when a new file is uploaded)."""
    if req.session_id:
        reset_session(req.session_id)
        _CONVERSATIONS.pop(req.session_id, None)
    return {"status": "ok"}
