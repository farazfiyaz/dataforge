# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""load_file() called with pandas options, the way the agent writes it."""
import pytest

from services import workspace


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "semi.csv").write_text("x;y\n1;2\n3;4\n")
    (tmp_path / "people.csv").write_bytes("name,city\nJosé,München\nRenée,Köln\n".encode("cp1252"))
    (tmp_path / "t.tsv").write_text("a\tb\n1\t2\n")
    workspace.set_workspace(str(tmp_path))
    return tmp_path


def test_explicit_sep_is_used(ws):
    df = workspace.load_file("semi.csv", sep=";")
    assert df.to_dict("list") == {"x": [1, 3], "y": [2, 4]}


def test_explicit_delimiter_is_used(ws):
    df = workspace.load_file("semi.csv", delimiter=";")
    assert list(df.columns) == ["x", "y"]


def test_other_options_still_sniff_the_delimiter(ws):
    df = workspace.load_file("semi.csv", nrows=1)
    assert df.to_dict("list") == {"x": [1], "y": [2]}


def test_options_on_a_windows_1252_file(ws):
    df = workspace.load_file("people.csv", usecols=["city"])
    assert df["city"].tolist() == ["München", "Köln"]


def test_explicit_encoding_is_respected(ws):
    df = workspace.load_file("people.csv", encoding="cp1252")
    assert df["name"].tolist() == ["José", "Renée"]


def test_tsv_with_options(ws):
    df = workspace.load_file("t.tsv", usecols=["b"])
    assert df["b"].tolist() == [2]
