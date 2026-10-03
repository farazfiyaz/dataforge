# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""What DataForge makes is saved to Documents/DataForge, and deleting it never breaks anything."""
import io
import shutil

import matplotlib
matplotlib.use("Agg")
import pandas as pd

import services.executor as executor
from conftest import APP_ORIGIN
from services import outputs
from services.datastore import put_dataset

H = {"Origin": APP_ORIGIN}


def _files():
    return sorted(p.name for p in outputs.OUTPUT_ROOT.rglob("*") if p.is_file())


def _session(name="save"):
    put_dataset(f"{name}-ds", pd.DataFrame({"plan": ["a", "b"] * 20, "x": range(40)}))
    executor.reset_session(name)
    return {"session_id": name, "dataset_id": f"{name}-ds"}


def test_charts_and_produced_tables_are_saved():
    out = executor.run_code(
        "plt.bar(['a', 'b'], [1, 2])\nplt.title('Churn by plan')\nsummary = df.groupby('plan').x.mean().reset_index()",
        **_session())
    assert len(out["saved"]) == 2
    names = _files()
    assert any(n.endswith("_Churn_by_plan.png") for n in names)
    assert any(n.endswith("_table.csv") for n in names)
    saved_csv = next(p for p in outputs.OUTPUT_ROOT.rglob("*.csv"))
    assert list(pd.read_csv(saved_csv).columns) == ["plan", "x"]


def test_unchanged_dataset_is_not_saved_every_step():
    out = executor.run_code("n = len(df)\nprint(n)", **_session("quiet"))
    assert out["saved"] == [] and _files() == []


def test_deleting_the_folder_mid_session_is_fine():
    s = _session("del")
    executor.run_code("plt.plot([1, 2])", **s)
    shutil.rmtree(outputs.OUTPUT_ROOT)                     # the user cleans up
    out = executor.run_code("plt.plot([3, 4])", **s)
    assert out["error"] is None and len(out["saved"]) == 1  # folder recreated
    assert len(_files()) == 1


def test_a_failed_save_never_fails_the_run(monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")                                 # a file where the folder should be
    monkeypatch.setattr(outputs, "OUTPUT_ROOT", blocker)
    out = executor.run_code("plt.plot([1, 2])", **_session("fail"))
    assert out["error"] is None and out["charts"] and out["saved"] == []


def test_same_second_names_do_not_overwrite():
    a = outputs.save_png("iVBORw0KGgo=", "chart")
    b = outputs.save_png("iVBORw0KGgo=", "chart")
    assert a != b and len(_files()) == 2


def test_autoanalyze_saves_cleaned_data_and_charts(client):
    csv = "age,plan\n" + "\n".join(f"{20 + i},{'ab'[i % 2]}" for i in range(30))
    res = client.post("/api/autoanalyze/", files={"file": ("people.csv", io.BytesIO(csv.encode()), "text/csv")}, headers=H)
    saved = res.json()["saved"]
    assert any("cleaned_people" in s for s in saved)
    assert sum(s.endswith(".png") for s in saved) >= 1


def test_outputs_folder_endpoint(client, monkeypatch):
    opened = []
    monkeypatch.setattr(outputs, "open_folder", lambda: opened.append(1) or str(outputs.OUTPUT_ROOT))
    assert client.get("/api/outputs/", headers=H).json()["folder"] == str(outputs.OUTPUT_ROOT)
    assert client.post("/api/outputs/open", headers=H).status_code == 200 and opened
