# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Real-world files: Windows encodings, European separators, tabs, Excel."""
import io

import pandas as pd
import pytest

from conftest import APP_ORIGIN
from services import workspace
from services.loader import read_table


def test_windows_1252_csv_from_excel():
    df = read_table("name,city\nJosé,Zürich\nRenée,Köln\n".encode("cp1252"), "people.csv")
    assert list(df["city"]) == ["Zürich", "Köln"]


def test_utf8_bom_is_stripped():
    df = read_table("﻿id,val\n1,2\n".encode("utf-8"), "x.csv")
    assert list(df.columns) == ["id", "val"]


def test_semicolon_separator_with_decimal_commas():
    df = read_table("product;price;qty\nA;1,50;3\nB;2,75;4\n".encode(), "eu.csv")
    assert list(df.columns) == ["product", "price", "qty"]
    assert df["price"].tolist() == [1.5, 2.75]


def test_semicolon_separator_with_decimal_points():
    df = read_table("product;price\nA;1.50\nB;2.75\n".encode(), "x.csv")
    assert df["price"].tolist() == [1.5, 2.75]


def test_comma_csv_with_quoted_commas_is_unchanged():
    df = read_table('name,note\n"Smith, J","a; b"\n"Lee, K","c; d"\n'.encode(), "x.csv")
    assert list(df.columns) == ["name", "note"]
    assert df["name"].tolist() == ["Smith, J", "Lee, K"]


@pytest.mark.parametrize("filename", ["tabs.csv", "tabs.tsv", "tabs.txt"])
def test_tab_separated(filename):
    df = read_table(b"a\tb\n1\t2\n3\t4\n", filename)
    assert list(df.columns) == ["a", "b"]


def test_single_column_file():
    df = read_table(b"value\n1\n2\n3\n", "one.csv")
    assert df["value"].tolist() == [1, 2, 3]


def test_xlsx_round_trip():
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False)
    df = read_table(buf.getvalue(), "book.xlsx")
    assert df.to_dict("list") == {"a": [1, 2], "b": ["x", "y"]}


@pytest.mark.parametrize("data, filename, message", [
    (b"", "empty.csv", "empty"),
    (b"  \n ", "blank.csv", "empty"),
    (b"a,b\n1,2", "data.json", "supported"),
    (b"a,b\n1,2", "noextension", "supported"),
])
def test_clear_errors(data, filename, message):
    with pytest.raises(ValueError, match=message):
        read_table(data, filename)


def test_upload_endpoint_accepts_european_excel_export(client):
    data = "Kunde;Umsatz\nMüller;1.234,50\nSchön;99,90\n".encode("cp1252")
    res = client.post("/api/upload/", files={"file": ("umsatz.csv", io.BytesIO(data), "text/csv")},
                      headers={"Origin": APP_ORIGIN})
    assert res.status_code == 200
    cols = {c["name"]: c for c in res.json()["profile"]["columns"]}
    assert list(cols) == ["Kunde", "Umsatz"]
    assert cols["Umsatz"]["kind"] == "numeric"
    assert cols["Umsatz"]["max"] == 1234.5


def test_upload_endpoint_rejects_empty_file_with_400(client):
    res = client.post("/api/upload/", files={"file": ("e.csv", io.BytesIO(b""), "text/csv")},
                      headers={"Origin": APP_ORIGIN})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"]


@pytest.mark.parametrize("data", [b"a,b,c\n", b"a,b,c", b"a;b;c\r\n\r\n"])
def test_header_only_csv_is_a_clear_error(data):
    with pytest.raises(ValueError, match=r"column headers \(3\) but no data rows"):
        read_table(data, "filtered.csv")


def test_header_only_and_blank_excel_are_clear_errors():
    buf = io.BytesIO()
    pd.DataFrame(columns=["a", "b"]).to_excel(buf, index=False)
    with pytest.raises(ValueError, match="no data rows"):
        read_table(buf.getvalue(), "template.xlsx")
    buf = io.BytesIO()
    pd.DataFrame().to_excel(buf, index=False)
    with pytest.raises(ValueError, match="empty"):
        read_table(buf.getvalue(), "blank.xlsx")


@pytest.mark.parametrize("endpoint", ["/api/upload/", "/api/autoanalyze/"])
def test_endpoints_reject_header_only_file_with_400(client, endpoint):
    # used to be a 500: the profile's null percentages were NaN, not valid JSON
    res = client.post(endpoint, files={"file": ("h.csv", io.BytesIO(b"id,churn\n"), "text/csv")},
                      headers={"Origin": APP_ORIGIN})
    assert res.status_code == 400
    assert "no data rows" in res.json()["detail"]


def test_workspace_load_file_header_only(tmp_path):
    (tmp_path / "h.csv").write_bytes(b"a,b\n")
    workspace.set_workspace(str(tmp_path))
    with pytest.raises(ValueError, match="no data rows"):
        workspace.load_file("h.csv")


def test_workspace_load_file_uses_sniffing(tmp_path):
    (tmp_path / "eu.csv").write_bytes("a;b\n1,5;2\n".encode("cp1252"))
    workspace.set_workspace(str(tmp_path))
    df = workspace.load_file("eu.csv")
    assert df["a"].tolist() == [1.5]


def test_xls_without_xlrd_explains_the_fix(monkeypatch):
    def no_xlrd(*args, **kwargs):
        raise ImportError("Missing optional dependency 'xlrd'.")
    monkeypatch.setattr(pd, "read_excel", no_xlrd)
    with pytest.raises(ValueError, match="xlrd"):
        read_table(b"\xd0\xcf\x11\xe0 fake xls", "old.xls")
