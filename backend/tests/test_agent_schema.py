# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""The dataset schema in the agent's system prompt must describe columns as they really are."""

from services.datastore import put_dataset
from services.loader import read_table
from routers.agent import _schema


def _schema_of(csv: bytes) -> str:
    put_dataset("schema-ds", read_table(csv, "data.csv"))
    return _schema("schema-ds")


def test_true_false_columns_are_booleans_not_text():
    # `active` parses as bool; `trial` as object bools because of the blank cell.
    # Described as text 'False'/'True', the model wrote df['active'] == 'True' → always 0%
    schema = _schema_of(b"active,trial,plan\nTrue,True,basic\nFalse,,pro\nTrue,False,basic\n")
    assert "- active: boolean True/False, 2 True (rate: df['active'].mean(), per group: df.groupby(...)['active'].mean())" in schema
    assert "- trial: boolean True/False, 1 True (rate: df['trial'].mean(), per group: df.groupby(...)['trial'].mean()), 1 missing" in schema
    assert "'True'" not in schema
    assert "- plan: text, 2 values: 'basic', 'pro'" in schema


def test_blank_column_is_empty_not_numeric_nan_range():
    schema = _schema_of(b"a,notes,b\n1,,x\n2,,y\n3,,x\n")
    assert "- notes: empty (all 3 values missing)" in schema
    assert "nan" not in schema


def test_numeric_and_text_unchanged():
    schema = _schema_of(b"tenure,churn\n1,yes\n5,no\n\n3,\n")
    assert "- tenure: numeric (int64), range 1 to 5" in schema
    assert "- churn: text, 2 values: 'no', 'yes', 1 missing" in schema
