# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Generated code must not reach the OS, the filesystem or interpreter internals.

Every attack below worked before services/codeguard.py: whitelisted libraries
re-exported `os`, dunder chains reached any class, and pandas read and wrote
arbitrary paths. A prompt-injected CSV cell could steer the model into them.
"""
import pytest

from services.executor import run_code

ATTACKS = {
    "library re-exports os": "print(pd.io.common.os.getcwd())",
    "dunder chain": "print(type(()).__mro__[-1].__subclasses__())",
    "io.open": "import io\nio.open('x.txt', 'w')",
    "read any path": "pd.read_csv(r'C:/Windows/win.ini')",
    "write any path": "pd.DataFrame({'a': [1]}).to_csv('x.csv')",
    "aliased writer": "f = pd.DataFrame({'a': [1]}).to_csv\nf('x.csv')",
    "file exporter": "pd.DataFrame({'a': [1]}).to_pickle('x.pkl')",
    "query reaches modules": "d = pd.DataFrame({'a': [1]})\nd.query('@pd.io.common.os.getcwd() == a')",
    "query calls": "d = pd.DataFrame({'a': [1]})\nd.query('@print(1) == a')",
    "non-literal query": "d = pd.DataFrame({'a': [1]})\nq = 'a > 0'\nd.query(q)",
    "format string traversal": "print('{0.__class__}'.format(1))",
    "generator frame": "g = (i for i in [1])\nprint(g.gi_frame.f_back)",
    "numpy pickle load": "np.load('x.npy', allow_pickle=True)",
    "submodule import": "import scipy.io as sio",
    "from-import of a loader": "from numpy import load",
    "network dataset fetch": "sns.load_dataset('iris')",
    "getattr with a string": "getattr(pd, 'io')",
    "builtins dict": "print(__builtins__)",
    "import of os": "import os",
}


@pytest.mark.parametrize("code", ATTACKS.values(), ids=ATTACKS.keys())
def test_escape_is_blocked(code, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = run_code(code)
    assert res["error"], f"ran without error: {res['stdout']!r}"
    assert not list(tmp_path.iterdir()), "the attack wrote a file"


def test_blocked_code_gets_a_clear_reason():
    res = run_code("pd.read_csv('sales.csv')")
    assert res["error"].startswith("SandboxViolation:")
    assert "df" in res["error"]          # tells the model what to use instead


def test_builtins_are_not_shared_between_runs():
    from services.executor import SAFE_GLOBALS
    before = dict(SAFE_GLOBALS["__builtins__"])
    run_code("print = None")
    assert SAFE_GLOBALS["__builtins__"] == before


ANALYSIS = {
    "describe": "pd.DataFrame({'a': [1, 2, 3]}).describe()",
    "query with a local": "d = pd.DataFrame({'a': [1, 2, 3]})\nlim = 1\nd.query('a > @lim')",
    "query with a backticked column": "d = pd.DataFrame({'my col': [1, 2]})\nd.query('`my col` > 1')",
    "csv as text": "print(pd.DataFrame({'a': [1]}).to_csv(index=False))",
    "markdown as text": "print(pd.DataFrame({'a': [1]}).to_string(index=False))",
    "format spec": "print('{:.1f}'.format(2.0), f'{3.14159:.2f}')",
    "sklearn": "from sklearn.linear_model import LinearRegression\n"
               "LinearRegression().fit(np.arange(4).reshape(-1, 1), np.arange(4))",
    "scipy stats": "from scipy import stats\nprint(stats.pearsonr([1, 2, 3], [2, 4, 7])[0] > 0.9)",
    "column via brackets": "d = pd.DataFrame({'open_date': [1]})\nprint(d['open_date'].sum())",
}


@pytest.mark.parametrize("code", ANALYSIS.values(), ids=ANALYSIS.keys())
def test_ordinary_analysis_still_runs(code):
    res = run_code(code)
    assert res["error"] is None, res["error"]


def test_savefig_is_dropped_but_the_chart_is_kept(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = run_code("plt.plot([1, 2, 3])\nplt.title('t')\nplt.savefig('chart.png')")
    assert res["error"] is None
    assert res["charts"]
    assert not (tmp_path / "chart.png").exists()
