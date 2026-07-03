# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Server-side dataset store.

The frontend used to ship the entire raw CSV with every chat message and the
backend re-parsed it every time — fine for toy files, terrible for big data.
Now the file is parsed ONCE at upload, kept here in memory, and referenced by
dataset_id afterwards.
"""

from collections import OrderedDict

import pandas as pd

# Keep only the most recent datasets to bound memory use
MAX_DATASETS = 8

_DATASETS: "OrderedDict[str, pd.DataFrame]" = OrderedDict()


def put_dataset(dataset_id: str, df: pd.DataFrame) -> None:
    """Store a dataset, evicting the oldest if over capacity."""
    _DATASETS[dataset_id] = df
    _DATASETS.move_to_end(dataset_id)
    while len(_DATASETS) > MAX_DATASETS:
        _DATASETS.popitem(last=False)


def get_dataset(dataset_id: str) -> pd.DataFrame | None:
    """
    Return a COPY of the dataset (sessions may mutate their df freely
    without corrupting the master), or None if unknown (e.g. after restart).
    """
    df = _DATASETS.get(dataset_id)
    if df is None:
        return None
    _DATASETS.move_to_end(dataset_id)
    return df.copy()
