# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Tolerant file → DataFrame loading for real-world files.

A bare pd.read_csv rejects the CSVs Excel on Windows writes (cp1252, not
UTF-8), and reads semicolon- or tab-separated files (Excel's export in many
European locales) as one giant column. This sniffs the encoding, delimiter
and decimal mark first.

A row with more fields than the header (a stray comma in one line) would fail
the whole file, so those rows are dropped instead and their line numbers kept
in df.attrs["skipped_lines"] for the caller to report.
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
    df = _read_excel(data, ext) if ext in EXCEL_EXTENSIONS else _read_text(data, ext)
    # A header row alone (an export with its filter set to nothing, or an
    # empty template) profiled as NaN percentages, which the API couldn't
    # send as JSON: upload and Auto Analyze answered a bare 500
    if df.empty:
        if len(df.columns):
            raise ValueError(f"The file has column headers ({len(df.columns)}) but no data rows.")
        raise ValueError("The file is empty.")
    return df


def _read_excel(data: bytes, ext: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(data))
    except ImportError:
        if ext == "xls":
            raise ValueError("Old-style .xls files need the 'xlrd' package "
                             "(pip install xlrd), or re-save the file as .xlsx.") from None
        raise
    # Year headers (2023, 2024) come back as ints, which crash the cleaning
    # report and plot labels; a CSV with the same header gives text "2023"
    df.columns = [str(c) for c in df.columns]
    return df


def _read_text(data: bytes, ext: str) -> pd.DataFrame:
    text = _decode(data)
    sample = text[:_SNIFF_BYTES]
    sep = "\t" if ext == "tsv" else _sniff_delimiter(sample)
    # "1,50" style decimals travel with ';' separators (European Excel)
    # ...and then "." groups thousands: 1.234,50
    read = _read_decimal_comma if sep == ";" and _DECIMAL_COMMA.search(sample) else _read_plain
    try:
        return read(text, sep)
    except pd.errors.ParserError:
        skipped = _overlong_lines(text, sep)
        if not skipped:
            raise
        df = read(text, sep, on_bad_lines="skip")
        df.attrs["skipped_lines"] = skipped
        return df


def _overlong_lines(text: str, sep: str) -> list[int]:
    """1-based line numbers of rows with more fields than the header."""
    reader = csv.reader(io.StringIO(text), delimiter=sep)
    header = next(reader, [])
    return [reader.line_num for row in reader if len(row) > len(header)]


def _read_plain(text: str, sep: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), sep=sep, **kwargs)


def _read_decimal_comma(text: str, sep: str, **kwargs) -> pd.DataFrame:
    """
    pandas drops every "." once thousands="." is set, so the dd.mm.yyyy dates
    these exports carry became integers (01.02.2024 → 1022024) and versions
    like 1.10 became 110. Keep a parsed column only if every value really is
    a European number; otherwise keep the text as written.
    """
    df = pd.read_csv(io.StringIO(text), sep=sep, decimal=",", thousands=".", **kwargs)
    raw = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, **kwargs)
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
        sniffed = csv.Sniffer().sniff("\n".join(lines), delimiters=_DELIMITERS).delimiter
    except csv.Error:
        sniffed = ","
    # The sniffer wants every line to split the same way, so one ragged row
    # can make it pick ',' for a ';' or tab file. Prefer whichever delimiter
    # splits the header and most rows into the same number of fields.
    best = max(_DELIMITERS, key=lambda d: _consistency(lines, d))
    if _consistency(lines, best) > _consistency(lines, sniffed):
        return best
    return sniffed


def _consistency(lines: list[str], sep: str) -> tuple:
    counts = [len(row) for row in csv.reader(lines, delimiter=sep)]
    if not counts:
        return (False, 0.0, 0)
    mode = max(set(counts), key=counts.count)
    if mode < 2:
        return (False, 0.0, 0)
    return (counts[0] == mode, counts.count(mode) / len(counts), mode)
