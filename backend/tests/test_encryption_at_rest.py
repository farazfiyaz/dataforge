# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""Nothing sensitive is stored in plain text: the API key and chat history."""
import json
import re

import pytest

from services import history, settings, vault

SECRET_CHAT = "quarterly churn for ACME-CONFIDENTIAL"
DB = {"projects": [{"id": "p1", "name": "General", "chats": [{"id": "c1", "title": SECRET_CHAT}]}]}


# ── API key ──

def test_api_key_goes_to_the_credential_store_not_the_file(memory_vault):
    settings.save({"provider": "claude", "claude_api_key": "sk-ant-secret-1234"})
    assert memory_vault[vault.API_KEY] == "sk-ant-secret-1234"
    assert "sk-ant" not in settings.SETTINGS_PATH.read_text()
    assert settings.claude_api_key() == "sk-ant-secret-1234"
    assert settings.public_view()["api_key_hint"] == "…1234"


def test_clearing_the_key_removes_it_from_the_store(memory_vault):
    settings.save({"claude_api_key": "sk-ant-secret-1234"})
    settings.save({"claude_api_key": ""})
    assert vault.API_KEY not in memory_vault


def test_a_plaintext_key_from_an_older_version_is_moved(memory_vault):
    settings.SETTINGS_PATH.write_text(json.dumps({"provider": "claude", "claude_api_key": "sk-ant-old-9999"}))
    assert settings.load()["claude_api_key"] == "sk-ant-old-9999"
    assert memory_vault[vault.API_KEY] == "sk-ant-old-9999"
    on_disk = json.loads(settings.SETTINGS_PATH.read_text())
    assert "claude_api_key" not in on_disk and on_disk["provider"] == "claude"


# ── Chat history ──

def test_history_is_encrypted_on_disk():
    history.save(DB)
    blob = history.HISTORY_PATH.read_bytes()
    assert blob.startswith(history.MAGIC)
    assert b"ACME" not in blob and b"projects" not in blob
    assert history.load() == DB


def test_each_save_uses_a_fresh_nonce():
    history.save(DB)
    first = history.HISTORY_PATH.read_bytes()
    history.save(DB)
    assert history.HISTORY_PATH.read_bytes() != first


def test_another_key_cannot_read_it(memory_vault):
    history.save(DB)
    memory_vault.pop(vault.HISTORY_KEY)            # e.g. the file copied to another account
    with pytest.raises(history.HistoryUnreadable):
        history.load()


def test_tampering_is_detected():
    history.save(DB)
    blob = bytearray(history.HISTORY_PATH.read_bytes())
    blob[-1] ^= 1
    history.HISTORY_PATH.write_bytes(bytes(blob))
    with pytest.raises(history.HistoryUnreadable):
        history.load()


def _embedded(page: str):
    m = re.search(r'<script type="application/json" id="df-history">(.*?)</script>', page, re.S)
    return json.loads(m.group(1))


def test_the_page_saves_and_gets_back_its_history(client):
    assert _embedded(client.get("/").text) is None
    assert client.put("/api/history/", json={"db": DB}).status_code == 200
    assert _embedded(client.get("/").text) == DB


def test_history_needs_the_launch_token(anonymous_client):
    assert anonymous_client.put("/api/history/", json={"db": DB}).status_code == 401
    assert not history.HISTORY_PATH.exists()


def test_chat_text_cannot_break_out_of_the_embedded_json(client):
    evil = {"projects": [{"id": "p", "name": "</script><script>alert(1)</script>", "chats": []}]}
    client.put("/api/history/", json={"db": evil})
    page = client.get("/").text
    assert "<script>alert(1)" not in page
    assert _embedded(page) == evil


def test_an_unreadable_file_is_set_aside_not_overwritten(client, memory_vault):
    history.save(DB)
    memory_vault.pop(vault.HISTORY_KEY)
    assert _embedded(client.get("/").text) is None
    assert list(history.HISTORY_PATH.parent.glob("history.unreadable-*.bin"))
