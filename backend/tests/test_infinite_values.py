# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Files and code results holding inf (a spreadsheet's #DIV/0!, x / 0) must not break the API."""
import io
import json

import numpy as np
import pandas as pd

from conftest import APP_ORIGIN
from services.autoplot import generate_all_plots
from services.eda import table_records
from services.executor import run_code

CSV = b"ratio,score,grp\n1.5,3,a\ninf,4,b\n2.5,5,a\n-inf,6,b\n3.0,7,a\n"


def _post(client, endpoint):
    return client.post(endpoint, files={"file": ("x.csv", io.BytesIO(CSV), "text/csv")},
                       headers={"Origin": APP_ORIGIN})


def test_upload_accepts_a_file_with_inf(client):
    res = _post(client, "/api/upload/")
    assert res.status_code == 200
    sample = res.json()["profile"]["sample"]
    assert [row["ratio"] for row in sample] == [1.5, "inf", 2.5, "-inf", 3.0]


def test_autoanalyze_accepts_a_file_with_inf(client):
    res = _post(client, "/api/autoanalyze/")
    assert res.status_code == 200
    assert any(p["title"] == "Distributions" for p in res.json()["plots"])


def test_plots_skip_infinite_values():
    df = pd.DataFrame({"a": [1.0, 2.0, np.inf, 4.0] * 5, "b": np.arange(20.0, 0, -1)})
    titles = [p["title"] for p in generate_all_plots(df, df)]
    assert {"Distributions", "Box Plots", "Correlation Heatmap"} <= set(titles)


def test_code_result_table_with_inf_is_valid_json():
    res = run_code("pd.DataFrame({'x': [1.0, 0.0], 'y': [0.0, 0.0]}).assign(r=lambda d: d.x / d.y)")
    assert res["error"] is None
    json.dumps(res["table"], allow_nan=False)
    assert [row["r"] for row in res["table"]] == ["inf", ""]   # 0/0 is NaN → blank, as before


def test_table_records_handles_nullable_floats():
    df = pd.DataFrame({"f": pd.array([np.inf, None], dtype="Float64"), "s": ["a", "b"]})
    assert table_records(df, 5) == [{"f": "inf", "s": "a"}, {"f": "", "s": "b"}]
