# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Auto clean on a CSV written with ", " separators: every header and cell
arrives padded with a space, so numbers are text and blanks are " ".
Whitespace has to go before nulls, duplicates and numbers are handled.
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from conftest import APP_ORIGIN
from services.cleaner import auto_clean
from services.loader import read_table

PADDED_CSV = (
    b"id, plan, age, churn\n"
    b"1, basic, 34, yes\n"
    b"2, pro , 41, no\n"
    b"3,  , , no\n"
    b"4, pro, 29, \n"
    b"5, basic, 50, yes\n"
    b"5, basic , 50, yes\n"
)


def _clean():
    return auto_clean(read_table(PADDED_CSV, "padded.csv"))


def test_column_names_are_trimmed():
    cleaned, report = _clean()
    assert list(cleaned.columns) == ["id", "plan", "age", "churn"]
    assert "Trimmed spaces from column names" in {r["step"] for r in report}


def test_padded_numbers_become_numeric_and_get_median_filled():
    cleaned, _ = _clean()
    assert pd.api.types.is_numeric_dtype(cleaned["age"])
    assert cleaned["age"].isna().sum() == 0


def test_blank_cells_are_filled_not_left_as_empty_text():
    cleaned, report = _clean()
    assert cleaned["plan"].isna().sum() == 0
    assert "" not in set(cleaned["plan"]) | set(cleaned["churn"])
    assert set(cleaned["plan"]) == {"basic", "pro"}
    assert "Treated blank cells as missing" in {r["step"] for r in report}


def test_rows_differing_only_by_padding_are_duplicates():
    cleaned, _ = _clean()
    assert len(cleaned) == 5


def test_trimming_never_merges_two_columns():
    df = pd.DataFrame({"a": [1, 2], " a": [3, 4]})
    cleaned, _ = auto_clean(df)
    assert list(cleaned.columns) == ["a", " a"]


def test_autoanalyze_download_of_padded_csv_has_no_gaps(client):
    res = client.post("/api/autoanalyze/",
                      files={"file": ("padded.csv", io.BytesIO(PADDED_CSV), "text/csv")},
                      headers={"Origin": APP_ORIGIN})
    plt.close("all")
    assert res.status_code == 200
    cleaned = pd.read_csv(io.BytesIO(base64.b64decode(res.json()["cleaned_csv_b64"])))
    assert list(cleaned.columns) == ["id", "plan", "age", "churn"]
    assert cleaned.isna().sum().sum() == 0
