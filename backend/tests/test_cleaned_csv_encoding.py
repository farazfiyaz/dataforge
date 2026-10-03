# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Auto Analyze's cleaned CSV must keep accented text intact when opened in
Excel on Windows, which reads a CSV without a BOM as cp1252 ("café" → "cafÃ©").
"""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from conftest import APP_ORIGIN


def _cleaned_bytes(client, raw: bytes, name="cities.csv") -> bytes:
    res = client.post("/api/autoanalyze/", files={"file": (name, io.BytesIO(raw), "text/csv")},
                      headers={"Origin": APP_ORIGIN})
    plt.close("all")
    assert res.status_code == 200
    return base64.b64decode(res.json()["cleaned_csv_b64"])


def test_cleaned_csv_starts_with_utf8_bom(client):
    # cp1252 is what Excel on Windows writes, so this is a typical upload
    raw = "city,price\nCafé Zürich,5\nSão Paulo,7\nMünchen,9\n".encode("cp1252")
    data = _cleaned_bytes(client, raw)
    assert data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig")
    assert "Café Zürich" in text and "München" in text


def test_cleaned_csv_round_trips_through_pandas(client):
    raw = "city,price\nCafé,5\n€ store,7\n".encode("utf-8")
    cleaned = pd.read_csv(io.BytesIO(_cleaned_bytes(client, raw)))
    # the BOM must not leak into the first header
    assert list(cleaned.columns) == ["city", "price"]
    assert cleaned["city"].tolist() == ["Café", "€ store"]
