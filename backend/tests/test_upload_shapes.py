# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Common file shapes that used to 500 on upload or Auto Analyze."""
import io

import pandas as pd

from services.eda import run_eda
from services.loader import read_table

BOOL_CSV = b"plan,paid,score\nbasic,True,1\npro,False,2\nbasic,True,3\npro,True,4\nbasic,False,5\n"


def _xlsx(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _post(client, endpoint, data, filename):
    return client.post(endpoint, files={"file": (filename, io.BytesIO(data), "application/octet-stream")})


def test_bool_column_is_profiled_as_categorical():
    df = read_table(BOOL_CSV, "b.csv")
    assert pd.api.types.is_bool_dtype(df["paid"])
    col = next(c for c in run_eda(df)["columns"] if c["name"] == "paid")
    assert col["kind"] == "categorical"
    assert col["top_values"] == {True: 3, False: 2}


def test_upload_and_autoanalyze_accept_bool_column(client):
    for endpoint in ("/api/upload/", "/api/autoanalyze/"):
        res = _post(client, endpoint, BOOL_CSV, "b.csv")
        assert res.status_code == 200, (endpoint, res.text[:200])


def test_excel_year_headers_become_text():
    df = read_table(_xlsx(pd.DataFrame({"region": ["N", "S"], 2023: [1.0, 2.0]})), "y.xlsx")
    assert list(df.columns) == ["region", "2023"]


def test_autoanalyze_accepts_excel_with_year_headers(client):
    # an empty year column gets dropped, and its name is joined into the report
    df = pd.DataFrame({"region": ["N", "S", "E", "W"] * 3,
                       2023: [1.0, 2.0, 3.0, None] * 3,
                       2024: [None] * 12})
    res = _post(client, "/api/autoanalyze/", _xlsx(df), "y.xlsx")
    assert res.status_code == 200, res.text[:200]
    report = " ".join(step["detail"] for step in res.json()["cleaning_report"])
    assert "2024" in report
