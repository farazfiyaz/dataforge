# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Secrets live in the operating system's credential store, never in our files:
Windows Credential Manager (encrypted with the user's Windows login), the
macOS Keychain, or the Secret Service on Linux.

Two secrets are kept: the Anthropic API key, and the random key that encrypts
chat history on disk (services/history.py).
"""

import base64
import secrets

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

SERVICE = "DataForge"
API_KEY = "anthropic-api-key"
HISTORY_KEY = "history-key"


class VaultUnavailable(RuntimeError):
    """No usable OS credential store (e.g. a Linux machine without one)."""


def get(name: str) -> str:
    try:
        return keyring.get_password(SERVICE, name) or ""
    except KeyringError as e:
        raise VaultUnavailable(str(e)) from e


def put(name: str, value: str) -> None:
    try:
        if value:
            keyring.set_password(SERVICE, name, value)
        else:
            try:
                keyring.delete_password(SERVICE, name)
            except PasswordDeleteError:
                pass     # nothing saved: already cleared
    except KeyringError as e:
        raise VaultUnavailable(str(e)) from e


def data_key() -> bytes:
    """The 256-bit key for encrypting data at rest, created on first use."""
    encoded = get(HISTORY_KEY)
    if not encoded:
        encoded = base64.b64encode(secrets.token_bytes(32)).decode()
        put(HISTORY_KEY, encoded)
    return base64.b64decode(encoded)
