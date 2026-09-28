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


def test_european_dates_are_not_read_as_numbers():
    # thousands="." used to strip the dots: 01.02.2024 → 1022024
    data = "Datum;Umsatz;Menge\n01.02.2024;1.234,50;3\n15.03.2024;99,90;1\n".encode("cp1252")
    df = read_table(data, "umsatz.csv")
    assert df["Datum"].tolist() == ["01.02.2024", "15.03.2024"]
    assert df["Umsatz"].tolist() == [1234.5, 99.9]
    assert df["Menge"].tolist() == [3, 1]


def test_dotted_codes_beside_decimal_commas_stay_text():
    df = read_table("id;version;preis\n1;1.2;3,5\n2;1.10;4,0\n".encode(), "x.csv")
    assert df["version"].tolist() == ["1.2", "1.10"]
    assert df["id"].tolist() == [1, 2]
    assert df["preis"].tolist() == [3.5, 4.0]


def test_european_thousands_with_missing_values():
    df = read_table("a;b\n1.000,5;x\n;y\n-2.500.000;z\n".encode(), "x.csv")
    assert df["a"].iloc[0] == 1000.5 and pd.isna(df["a"].iloc[1]) and df["a"].iloc[2] == -2500000


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
