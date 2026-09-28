# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Tolerant file → DataFrame loading for real-world files.

A bare pd.read_csv rejects the CSVs Excel on Windows writes (cp1252, not
UTF-8), and reads semicolon- or tab-separated files (Excel's export in many
European locales) as one giant column. This sniffs the encoding, delimiter
and decimal mark first.
"""

import csv
import io
import re

import pandas as pd

TEXT_EXTENSIONS = {"csv", "tsv", "txt"}
EXCEL_EXTENSIONS = {"xlsx", "xls"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | EXCEL_EXTENSIONS

# Tried in order; latin-1 decodes any byte sequence, so it's the last resort
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
_DELIMITERS = ",;\t|"
_SNIFF_BYTES = 64 * 1024
_DECIMAL_COMMA = re.compile(r"(^|;)\s*-?[\d.]*\d,\d+\s*(;|$)", re.MULTILINE)
# A number as European Excel writes it: "." only between groups of three digits
_EU_NUMBER = re.compile(r"[+-]?(\d{1,3}(\.\d{3})+|\d+)(,\d+)?([eE][+-]?\d+)?")


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded file's bytes. Raises ValueError with a user-facing message."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only CSV, TSV and Excel (.xlsx, .xls) files are supported.")
    if not data.strip():
        raise ValueError("The file is empty.")
    if ext in EXCEL_EXTENSIONS:
        return _read_excel(data, ext)
    return _read_text(data, ext)


def _read_excel(data: bytes, ext: str) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(data))
    except ImportError:
        if ext == "xls":
            raise ValueError("Old-style .xls files need the 'xlrd' package "
                             "(pip install xlrd), or re-save the file as .xlsx.") from None
        raise


def _read_text(data: bytes, ext: str) -> pd.DataFrame:
    text = _decode(data)
    sample = text[:_SNIFF_BYTES]
    sep = "\t" if ext == "tsv" else _sniff_delimiter(sample)
    # "1,50" style decimals travel with ';' separators (European Excel)
    if sep == ";" and _DECIMAL_COMMA.search(sample):
        # ...and then "." groups thousands: 1.234,50
        return _read_decimal_comma(text, sep)
    return pd.read_csv(io.StringIO(text), sep=sep)


def _read_decimal_comma(text: str, sep: str) -> pd.DataFrame:
    """
    pandas drops every "." once thousands="." is set, so the dd.mm.yyyy dates
    these exports carry became integers (01.02.2024 → 1022024) and versions
    like 1.10 became 110. Keep a parsed column only if every value really is
    a European number; otherwise keep the text as written.
    """
    df = pd.read_csv(io.StringIO(text), sep=sep, decimal=",", thousands=".")
    raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        values = raw[col].dropna().str.strip()
        if not values.str.fullmatch(_EU_NUMBER).all():
            df[col] = raw[col]
    return df


def _decode(data: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise AssertionError("unreachable: latin-1 decodes any bytes")


def _sniff_delimiter(sample: str) -> str:
    # only whole lines — a line cut off mid-way would confuse the sniffer
    lines = sample.splitlines()[:50]
    if len(lines) > 1 and not sample.endswith("\n"):
        lines = lines[:-1]
    try:
        return csv.Sniffer().sniff("\n".join(lines), delimiters=_DELIMITERS).delimiter
    except csv.Error:
        return ","
