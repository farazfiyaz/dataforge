# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Projects and chat history, encrypted on disk.

They used to sit in the window's localStorage: plaintext JSON in Chromium's
LevelDB files, readable by anything that can read the user's profile. Now the
page hands them to the backend, which keeps them in one file encrypted with
AES-256-GCM. The key lives in the OS credential store (services/vault.py), so
the file alone is useless: copied to another account or machine, or read by a
program that isn't DataForge, it's noise.

File layout: MAGIC | 12-byte nonce | ciphertext+tag. GCM also authenticates,
so a tampered or truncated file is rejected rather than half-loaded.
"""

import json
import os
import secrets
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from services import vault

HISTORY_PATH = Path(os.environ.get("DATAFORGE_HISTORY", Path.home() / ".dataforge" / "history.bin"))
MAGIC = b"DFH1"
MAX_BYTES = 100 * 1024 * 1024     # far above what a localStorage-sized history ever reached


class HistoryUnreadable(Exception):
    """The file exists but can't be decrypted (wrong key, tampered, or truncated)."""


def load() -> dict | None:
    """The saved history, or None if there is none yet."""
    try:
        blob = HISTORY_PATH.read_bytes()
    except FileNotFoundError:
        return None
    if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 12 + 16:
        raise HistoryUnreadable("not a DataForge history file")
    nonce, sealed = blob[len(MAGIC):len(MAGIC) + 12], blob[len(MAGIC) + 12:]
    try:
        plain = AESGCM(vault.data_key()).decrypt(nonce, sealed, MAGIC)
    except InvalidTag as e:
        raise HistoryUnreadable("history could not be decrypted") from e
    return json.loads(plain)


def save(data: dict) -> None:
    plain = json.dumps(data, separators=(",", ":")).encode()
    if len(plain) > MAX_BYTES:
        raise ValueError("History is too large to save.")
    nonce = secrets.token_bytes(12)          # fresh per write: GCM must never reuse one
    blob = MAGIC + nonce + AESGCM(vault.data_key()).encrypt(nonce, plain, MAGIC)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # write-then-rename: a crash mid-save leaves the previous history intact
    tmp = HISTORY_PATH.with_suffix(".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, HISTORY_PATH)
