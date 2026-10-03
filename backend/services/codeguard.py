# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Static checks on LLM-generated code before the sandbox runs it.

The builtins/import allowlist in executor.py alone was escapable: whitelisted
libraries re-export `os` (`pd.io.common.os.system(...)`), dunder chains reach
any class (`type(()).__mro__[-1].__subclasses__()`), and pandas/numpy read and
write arbitrary paths (`pd.read_csv("C:/...")`, `df.to_csv("C:/...")`). A
prompt-injected dataset cell or web result could steer the model into any of
these. This module rejects them before anything executes.

It is a denylist over a small surface (attribute names, a few call shapes),
so analysis code — pandas, numpy, matplotlib, sklearn, f-strings — is
unaffected. Code that writes files gets a clear error instead: the app shows
tables and charts itself.
"""

import ast
import re

# Attribute names that lead to the OS, the filesystem, the network, or
# interpreter internals. Checked on every `a.b` and every `import a.b`.
BLOCKED_ATTRS = frozenset({
    # modules re-exported by whitelisted libraries
    "os", "sys", "subprocess", "builtins", "importlib", "imp", "shutil", "socket",
    "ctypes", "ctypeslib", "cffi", "pickle", "cPickle", "marshal", "shelve", "dbm",
    "io", "npyio", "pathlib", "Path", "PurePath", "tempfile", "urllib", "urllib3",
    "request", "requests", "http", "httpx", "ftplib", "smtplib", "telnetlib",
    "multiprocessing", "threading", "concurrent", "asyncio", "signal", "pty",
    "posix", "nt", "winreg", "msvcrt", "webbrowser", "glob", "fileinput",
    "zipfile", "tarfile", "gzip", "bz2", "lzma", "sqlite3", "inspect", "gc",
    "runpy", "pkgutil", "site", "sysconfig", "code", "codeop", "joblib", "loky",
    "cloudpickle", "dill", "fsspec", "platform", "getpass", "f2py", "distutils",
    "testing", "offline", "externals",
    # process / file / network functions
    "system", "popen", "spawn", "startfile", "load", "save", "savez",
    "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile",
    "memmap", "DataSource", "ExcelWriter", "HDFStore", "imread", "imsave",
    "load_dataset", "get_dataset_names", "load_svmlight_file", "dump_svmlight_file",
    "loadmat", "savemat",
    # frames, code objects and tracebacks (their f_globals hold real modules)
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "gi_frame", "gi_code",
    "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "ag_frame", "ag_code",
    "tb_frame", "tb_next", "co_code", "mro",
})
BLOCKED_PREFIXES = ("_", "read_", "open", "write", "fetch_", "exec", "spawn")

# Names that must never appear as variables (none are in the sandbox builtins,
# but a clear message beats a NameError the model will just retry around).
BLOCKED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "breakpoint", "input", "help", "exit", "quit", "memoryview",
})

# DataFrame/Series exporters: fine with no target (they return a string),
# never with a path or buffer.
TEXT_EXPORTERS = frozenset({"to_csv", "to_json", "to_html", "to_xml", "to_latex",
                            "to_markdown", "to_string"})
FILE_EXPORTERS = frozenset({"to_excel", "to_parquet", "to_feather", "to_hdf", "to_stata",
                            "to_pickle", "to_sql", "to_orc", "to_gbq", "to_clipboard"})
TARGET_KWARGS = frozenset({"path_or_buf", "buf", "path", "excel_writer", "fname", "con"})

# Strings pandas parses as code: run them through the same checks.
EXPR_METHODS = frozenset({"query", "eval"})
# "{0.__class__}".format(x) walks attributes from inside a string. Only the
# field name (before ':' or '!') can do that; "{:.1f}" is just a format spec.
_FORMAT_TRAVERSAL = re.compile(r"\{[^{}:!]*[.\[]")


class SandboxViolation(Exception):
    """Generated code tried something the sandbox does not allow."""


def _attr_problem(name: str) -> str | None:
    if name in BLOCKED_ATTRS or name.startswith(BLOCKED_PREFIXES):
        # a column that happens to share a name (df.open_date) — say how to reach it
        column_hint = "" if name.startswith("_") else f" (For a column named {name!r}, use df[{name!r}].)"
        if name.startswith("read_") or name.startswith("open") or name in {"load", "loadtxt", "fromfile"}:
            return f"'{name}' reads files. Use the dataset already loaded as `df`, or load_file(name) for workspace files.{column_hint}"
        if name.startswith("write") or name in {"save", "savez", "savetxt", "tofile", "imsave", "savemat"}:
            return f"'{name}' writes files. The app shows tables and charts itself; just display them.{column_hint}"
        return f"access to '{name}' is not allowed in the sandbox.{column_hint}"
    return None


class _Guard(ast.NodeVisitor):
    def visit_Name(self, node: ast.Name):
        if node.id in BLOCKED_NAMES or node.id.startswith("__"):
            raise SandboxViolation(f"'{node.id}' is not allowed in the sandbox.")

    def visit_Attribute(self, node: ast.Attribute):
        if problem := _attr_problem(node.attr):
            raise SandboxViolation(problem)
        if node.attr in TEXT_EXPORTERS | FILE_EXPORTERS | EXPR_METHODS:
            # only reachable when not called directly (e.g. `f = df.to_csv`)
            raise SandboxViolation(f"call .{node.attr}(...) directly.")
        self.visit(node.value)

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
            if name in FILE_EXPORTERS:
                raise SandboxViolation(f"'{name}' writes files. The app shows tables itself; just display the DataFrame.")
            if name in TEXT_EXPORTERS and (node.args or any(k.arg in TARGET_KWARGS for k in node.keywords)):
                raise SandboxViolation(f"'{name}' to a file is not allowed. Call it with no path to get text, or just display the DataFrame.")
            if name in EXPR_METHODS:
                self._check_expression_arg(node, name)
            if name in {"format", "format_map"} and isinstance(func.value, ast.Constant) \
                    and isinstance(func.value.value, str) and _FORMAT_TRAVERSAL.search(func.value.value):
                raise SandboxViolation("format strings may not access attributes or items; use an f-string.")
            if name in TEXT_EXPORTERS | FILE_EXPORTERS | EXPR_METHODS:
                self.visit(func.value)          # the method itself is fine; check what it's called on
            else:
                self.visit(func)
        else:
            self.visit(func)
        for a in node.args:
            self.visit(a)
        for k in node.keywords:
            self.visit(k.value)

    def _check_expression_arg(self, node: ast.Call, name: str):
        expr = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "expr"), None)
        if expr is None:
            return
        if not (isinstance(expr, ast.Constant) and isinstance(expr.value, str)):
            raise SandboxViolation(f".{name}() needs a literal string expression.")
        # `@var` refers to a local, `col name` in backticks is a column — neither
        # is valid Python on its own, so neutralise them before parsing.
        text = re.sub(r"`[^`]*`", "_col", expr.value).replace("@", "")
        try:
            tree = ast.parse(text, mode="exec")
        except SyntaxError:
            return   # pandas will reject it too
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Attribute) and (problem := _attr_problem(sub.attr)):
                raise SandboxViolation(f"in .{name}(): {problem}")
            if isinstance(sub, ast.Call):
                raise SandboxViolation(f".{name}() expressions may not call functions; compute it with pandas instead.")
            if isinstance(sub, ast.Name) and (sub.id in BLOCKED_NAMES or sub.id.startswith("__")):
                raise SandboxViolation(f"in .{name}(): '{sub.id}' is not allowed.")

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._check_module(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level:
            raise SandboxViolation("relative imports are not allowed in the sandbox.")
        self._check_module(node.module or "")
        for alias in node.names:
            if alias.name == "*" or (problem := _attr_problem(alias.name)):
                raise SandboxViolation(f"importing '{alias.name}' is not allowed in the sandbox.")

    @staticmethod
    def _check_module(dotted: str):
        for part in dotted.split(".")[1:]:
            if _attr_problem(part):
                raise SandboxViolation(f"importing '{dotted}' is not allowed in the sandbox.")


class _NoSaveFig(ast.NodeTransformer):
    """
    `plt.savefig("chart.png")` is in nearly every matplotlib example, so models
    write it constantly. Drop the call rather than fail the run: the figure is
    still captured and shown, it just isn't written to an arbitrary path.
    """
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "savefig":
            return ast.copy_location(ast.Constant(value=None), node)
        return node


def check(tree: ast.AST) -> ast.AST:
    """Raise SandboxViolation if `tree` is unsafe; return it with savefig calls removed."""
    tree = _NoSaveFig().visit(tree)
    _Guard().visit(tree)
    return ast.fix_missing_locations(tree)
