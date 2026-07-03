# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
import sys
import io
import base64
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
from typing import Any

from services.datastore import get_dataset

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
}
if sns is not None:
    SAFE_GLOBALS["sns"] = sns

# Persistent namespaces per session — like a notebook kernel, variables
# survive across run_code calls so multi-step (agentic) analysis works.
_SESSIONS: dict[str, dict[str, Any]] = {}


def get_session(session_id: str) -> dict[str, Any]:
    """Return (creating if needed) the persistent namespace for a session."""
    return _SESSIONS.setdefault(session_id, {})


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

    result: dict[str, Any] = {"stdout": "", "table": None, "chart": None, "error": None}

    keys_before = set(local_ns.keys())

    try:
        plt.clf()
        exec(code, {**SAFE_GLOBALS, **local_ns}, local_ns)  # noqa: S102

        result["stdout"] = stdout_capture.getvalue()

        # Prefer a DataFrame assigned in THIS run; fall back to any DataFrame
        new_keys = [k for k in local_ns.keys() if k not in keys_before]
        scan = list(reversed(new_keys)) or list(reversed(list(local_ns.keys())))
        for key in scan:
            val = local_ns[key]
            if isinstance(val, pd.DataFrame):
                result["table"] = val.head(100).fillna("").to_dict(orient="records")
                break

        # Check if a matplotlib figure was drawn → encode as base64 PNG
        fig = plt.gcf()
        if fig.get_axes():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            result["chart"] = base64.b64encode(buf.read()).decode()
            plt.clf()

    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    return result
