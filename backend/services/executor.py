# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
import ast
import sys
import io
import time
import asyncio
import base64
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
from typing import Any

from services.datastore import get_dataset
from services.ml import train_model
from services.workspace import list_files, load_file

try:
    import seaborn as sns
except ImportError:      # seaborn optional — sandbox still works without it
    sns = None

# Libraries generated code is allowed to import (root package names)
_ALLOWED_IMPORT_ROOTS = {
    "math", "statistics", "datetime", "re", "json", "itertools", "collections",
    "functools", "random", "textwrap", "io",
    "numpy", "pandas", "matplotlib", "seaborn", "scipy", "sklearn", "plotly",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Whitelist-based __import__ so code can `import seaborn as sns` etc."""
    if name.split(".")[0] in _ALLOWED_IMPORT_ROOTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(
        f"import of '{name}' is not allowed in the sandbox. "
        f"Allowed: {', '.join(sorted(_ALLOWED_IMPORT_ROOTS))}"
    )


# Modules the sandbox makes available to generated code
SAFE_GLOBALS = {
    "__builtins__": {
        "print": print, "range": range, "len": len, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "list": list, "dict": dict,
        "tuple": tuple, "set": set, "str": str, "int": int, "float": float,
        "bool": bool, "round": round, "abs": abs, "min": min, "max": max,
        "sum": sum, "sorted": sorted, "reversed": reversed,
        "isinstance": isinstance, "type": type, "repr": repr,
        "all": all, "any": any, "divmod": divmod, "pow": pow, "format": format,
        "frozenset": frozenset, "complex": complex, "iter": iter, "next": next,
        "slice": slice, "getattr": getattr, "hasattr": hasattr, "callable": callable,
        "__import__": _safe_import,
        # exceptions so generated code can use try/except
        "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError,
        "TypeError": TypeError, "IndexError": IndexError, "AttributeError": AttributeError,
        "ZeroDivisionError": ZeroDivisionError, "StopIteration": StopIteration,
        "ImportError": ImportError, "RuntimeError": RuntimeError,
    },
    "pd": pd,
    "np": np,
    "plt": plt,
    "train_model": train_model,   # one-call ML pipeline (services/ml.py)
    "list_files": list_files,     # workspace folder access (services/workspace.py)
    "load_file": load_file,       # → DataFrame from workspace file
}
if sns is not None:
    SAFE_GLOBALS["sns"] = sns

# Wall-clock budget for one run_code call. Generated code that loops forever
# used to hang the kernel (and, run on the event loop, the whole server).
EXEC_TIMEOUT_S = 120
SANDBOX_FILENAME = "<sandbox>"   # compile() name — lets the tracer tell sandbox frames apart

# One worker: runs are serialized (matplotlib's pyplot state and the stdout
# swap below are process-global) but never block the async event loop.
_EXEC_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sandbox")


class ExecutionTimeout(BaseException):
    """BaseException so generated `except Exception:` blocks can't swallow it."""


def _deadline_tracer(deadline: float, timeout: float):
    """
    sys.settrace hook that aborts sandbox code once past `deadline`.
    Only frames compiled from the sandbox are line-traced, so library code
    (pandas, numpy, sklearn) runs at full speed.
    """
    def local(frame, event, arg):
        if time.monotonic() > deadline:
            raise ExecutionTimeout(
                f"Execution stopped after {timeout:g}s time limit. "
                "Use vectorized pandas/numpy instead of Python loops, or work on df.sample(...)."
            )
        return local

    def global_(frame, event, arg):
        return local if frame.f_code.co_filename == SANDBOX_FILENAME else None

    return global_


# Persistent namespaces per session — like a notebook kernel, variables
# survive across run_code calls so multi-step (agentic) analysis works.
# Bounded (like services/datastore.py's dataset cache) so a long-running session
# on a low-RAM machine doesn't accumulate namespaces — and their DataFrames — forever.
MAX_SESSIONS = 20
_SESSIONS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def get_session(session_id: str) -> dict[str, Any]:
    """Return (creating if needed) the persistent namespace for a session."""
    if session_id in _SESSIONS:
        _SESSIONS.move_to_end(session_id)
    ns = _SESSIONS.setdefault(session_id, {})
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    return ns


def peek_session_df(session_id: str | None) -> pd.DataFrame | None:
    """The session's current `df` (as the agent has transformed it), without creating a session."""
    df = _SESSIONS.get(session_id, {}).get("df") if session_id else None
    return df if isinstance(df, pd.DataFrame) else None


def reset_session(session_id: str) -> None:
    """Drop a session's namespace (e.g. when a new dataset is loaded)."""
    _SESSIONS.pop(session_id, None)


def run_code(
    code: str,
    csv_data: str = "",
    session_id: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute LLM-generated code in a restricted namespace.

    If session_id is given, the namespace persists across calls (notebook-style),
    so `df` and intermediate variables survive between agent steps.

    The dataset is injected once per session: preferably from the server-side
    store by dataset_id (fast, no re-parsing — handles big files), falling back
    to parsing csv_data if provided.

    Returns:
        - stdout  : captured print output
        - table   : list-of-dicts if the last expression is a DataFrame
        - chart   : base64 PNG string if a matplotlib figure was drawn
        - error   : traceback string on failure
    """
    local_ns: dict[str, Any] = get_session(session_id) if session_id else {}

    # Inject the dataset if provided (only once per session — don't clobber
    # a df the agent may have transformed in earlier steps)
    if "df" not in local_ns:
        if dataset_id:
            df = get_dataset(dataset_id)
            if df is not None:
                local_ns["df"] = df
        if "df" not in local_ns and csv_data.strip():
            try:
                local_ns["df"] = pd.read_csv(io.StringIO(csv_data))
            except Exception:
                pass

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    old_trace = sys.gettrace()

    result: dict[str, Any] = {"stdout": "", "table": None, "chart": None, "charts": None, "error": None}

    keys_before = set(local_ns.keys())

    try:
        plt.close("all")   # drop any figures left over from a previous run — keeps memory flat across a long session
        body, last_expr = _split_last_expression(code)
        namespace = {**SAFE_GLOBALS, **local_ns}
        sys.settrace(_deadline_tracer(time.monotonic() + EXEC_TIMEOUT_S, EXEC_TIMEOUT_S))
        try:
            exec(body, namespace, local_ns)  # noqa: S102
            last_value = eval(last_expr, namespace, local_ns) if last_expr else None  # noqa: S307
        finally:
            sys.settrace(old_trace)
        if (shown := _echo(last_value)) is not None:
            print(shown)

        result["stdout"] = stdout_capture.getvalue()

        # A DataFrame the code ended on is what the user asked to see
        if isinstance(last_value, pd.DataFrame):
            result["table"] = last_value.head(100).fillna("").to_dict(orient="records")

        # Otherwise prefer a DataFrame assigned in THIS run; fall back to any DataFrame
        new_keys = [k for k in local_ns.keys() if k not in keys_before]
        scan = list(reversed(new_keys)) or list(reversed(list(local_ns.keys())))
        for key in scan if result["table"] is None else []:
            val = local_ns[key]
            if isinstance(val, pd.DataFrame):
                result["table"] = val.head(100).fillna("").to_dict(orient="records")
                break

        # Encode EVERY figure the code drew, not just the last one — code that
        # plots more than once per run (e.g. train_model's diagnostics figure
        # followed by an extra chart) used to lose all but the final figure.
        charts = []
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if not fig.get_axes():
                continue
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            charts.append(base64.b64encode(buf.read()).decode())
        plt.close("all")

        if charts:
            result["charts"] = charts
            result["chart"] = charts[0]   # back-compat: first chart for older consumers

    except ExecutionTimeout as e:
        result["stdout"] = stdout_capture.getvalue()
        result["error"] = f"ExecutionTimeout: {e}"
        plt.close("all")
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    return result


MAX_ECHO_CHARS = 4000


def _split_last_expression(code: str):
    """
    Notebook semantics: if the code ends with a bare expression (`df.head()`),
    return (body, expr) so the expression's value can be shown. Small models
    write notebook-style code constantly; without this they get empty output
    back and repeat themselves.
    """
    tree = ast.parse(code, SANDBOX_FILENAME)
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return compile(tree, SANDBOX_FILENAME, "exec"), None
    last = tree.body.pop()
    return (compile(tree, SANDBOX_FILENAME, "exec"),
            compile(ast.Expression(last.value), SANDBOX_FILENAME, "eval"))


def _echo(value: Any) -> str | None:
    """Text for a notebook-style Out[]: None for values a notebook wouldn't show usefully."""
    if value is None or type(value).__module__.startswith("matplotlib"):
        return None   # plt.show(), sns.histplot(...) → the chart itself is the output
    try:
        if isinstance(value, pd.DataFrame):
            text = value.to_string(max_rows=30, max_cols=20)
        elif isinstance(value, pd.Series):
            text = value.to_string(max_rows=30)
        else:
            text = repr(value)
    except Exception:   # a broken __repr__ must not turn working code into an error
        return None
    return text if len(text) <= MAX_ECHO_CHARS else text[:MAX_ECHO_CHARS] + "\n… (truncated)"


async def run_code_async(code: str, csv_data: str = "", **kwargs) -> dict[str, Any]:
    """run_code on the sandbox worker thread, so the server stays responsive meanwhile."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXEC_POOL, partial(run_code, code, csv_data, **kwargs))
