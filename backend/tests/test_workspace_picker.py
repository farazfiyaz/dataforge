# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
The Workspace button: a native folder picker in the desktop app (Electron
throws on window.prompt, so the button did nothing), and quoted paths pasted
from Explorer's "Copy as path" in the browser fallback.
"""
from pathlib import Path

import pytest

from conftest import APP_ORIGIN
from services import workspace

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_workspace(monkeypatch):
    monkeypatch.setattr(workspace, "_workspace_root", None)


def _set(client, path):
    return client.post("/api/agent/workspace", json={"path": path},
                       headers={"Origin": APP_ORIGIN}).json()


@pytest.mark.parametrize("wrap", ['"{}"', "'{}'", '  "{}"  '])
def test_quoted_path_from_copy_as_path_is_accepted(client, tmp_path, wrap):
    (tmp_path / "sales.csv").write_text("a,b\n1,2\n")
    res = _set(client, wrap.format(tmp_path))
    assert "error" not in res, res
    assert Path(res["path"]) == tmp_path.resolve()
    assert res["files"] == 1


def test_blank_path_is_an_error_not_the_current_folder(client):
    res = _set(client, '  ""  ')
    assert "error" in res
    assert workspace.get_workspace() is None


def test_missing_folder_still_rejected(client, tmp_path):
    assert "error" in _set(client, f'"{tmp_path / "nope"}"')


def test_desktop_app_wires_a_native_folder_picker():
    main = (ROOT / "electron" / "main.js").read_text(encoding="utf-8")
    preload = (ROOT / "electron" / "preload.js").read_text(encoding="utf-8")
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'ipcMain.handle("df:pick-folder"' in main
    assert '"openDirectory"' in main
    assert 'ipcRenderer.invoke("df:pick-folder"' in preload
    handler = html[html.index('getElementById("workspace-btn").addEventListener'):]
    handler = handler[:handler.index("\n});")]
    assert "window.dataforge?.pickFolder" in handler
