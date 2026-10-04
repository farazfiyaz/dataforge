# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""One row with an extra field (a stray comma) shouldn't sink the whole file."""
import io

import pytest

from conftest import APP_ORIGIN
from services.loader import read_table


def test_extra_field_row_is_skipped_and_reported():
    df = read_table(b"a,b,c\n1,2,3\n4,5,6,7\n8,9,10\n", "x.csv")
    assert df.values.tolist() == [[1, 2, 3], [8, 9, 10]]
    assert df.attrs["skipped_lines"] == [3]


def test_quoted_separators_are_not_counted_as_extra_fields():
    df = read_table(b'a,b,c\n"x,y",1,2\n4,5,6,7\n', "x.csv")
    assert df["a"].tolist() == ["x,y"]
    assert df.attrs["skipped_lines"] == [3]


def test_clean_file_has_no_skipped_lines():
    df = read_table(b"a,b\n1,2\n3\n", "x.csv")   # short rows are fine: filled with NaN
    assert len(df) == 2
    assert "skipped_lines" not in df.attrs


@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_ragged_row_doesnt_change_the_detected_separator(sep):
    lines = ["name", "price", "qty"], ["A", "1", "2"], ["B", "2", "3"], ["C", "3", "4", "oops"], ["D", "4", "5"]
    data = "\n".join(sep.join(l) for l in lines).encode()
    df = read_table(data, "x.csv")
    assert list(df.columns) == ["name", "price", "qty"]
    assert df["name"].tolist() == ["A", "B", "D"]
    assert df.attrs["skipped_lines"] == [4]


def test_european_decimals_survive_a_ragged_row():
    data = b"name;price;qty\nA;1,50;2\nB;2,75;3\nC;3,00;4;oops\nD;1.234,50;5\n"
    df = read_table(data, "eu.csv")
    assert df["price"].tolist() == [1.5, 2.75, 1234.5]


@pytest.mark.parametrize("endpoint", ["/api/upload/", "/api/autoanalyze/"])
def test_endpoints_accept_ragged_file_and_list_skipped_lines(client, endpoint):
    data = b"a,b,c\n1,2,3\n4,5,6,7\n8,9,10\n11,12,13,14,15\n"
    res = client.post(endpoint, files={"file": ("r.csv", io.BytesIO(data), "text/csv")},
                      headers={"Origin": APP_ORIGIN})
    assert res.status_code == 200, res.text
    profile = res.json()["profile"]
    assert profile["shape"]["rows"] == 2
    assert profile["skipped_lines"] == [3, 5]


def test_frontend_shows_the_skipped_lines_note():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[2] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "profile.skipped_lines" in html
