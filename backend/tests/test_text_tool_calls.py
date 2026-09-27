# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Tool calls written as text (as qwen2.5-coder:7b does via Ollama) are executed,
not shown to the user as the "final answer".
"""
import io
import json

import pytest

from conftest import APP_ORIGIN
from routers import agent as agent_router
from routers.agent import _helper_call_as_code, _text_tool_calls

CODE = "x = 6 * 7\nprint(x)"
CALL = {"name": "run_python", "arguments": {"code": CODE}}


@pytest.mark.parametrize("content", [
    json.dumps(CALL),                                           # bare (seen from qwen2.5-coder:7b)
    "```json\n" + json.dumps(CALL, indent=2) + "\n```",         # fenced
    "<tool_call>\n" + json.dumps(CALL) + "\n</tool_call>",      # Qwen/Hermes tags
    "<tool_call>" + json.dumps(CALL),                           # unterminated tag
    json.dumps({"name": "run_python", "parameters": {"code": CODE}}),
    '{"name": "run_python", "arguments": {"code": "x = 6 * 7\nprint(x)"}}',   # raw newline in string
])
def test_recognises_text_tool_calls(content):
    assert _text_tool_calls(content) == [{"function": {"name": "run_python", "arguments": {"code": CODE}}}]


def test_unknown_tool_names_are_still_recognised_as_calls():
    # seen from qwen2.5-coder:7b: it "called" the train_model helper as a tool
    content = json.dumps({"name": "train_model", "arguments": {"target": "churn"}})
    assert _text_tool_calls(content)[0]["function"]["name"] == "train_model"


def test_recognises_several_calls():
    content = "<tool_call>" + json.dumps(CALL) + "</tool_call>\n<tool_call>" + json.dumps(CALL) + "</tool_call>"
    assert len(_text_tool_calls(content)) == 2


@pytest.mark.parametrize("content", [
    "The churn rate is 26.5%.",
    'Here is the result: {"name": "run_python"}',               # mentions JSON mid-answer
    '{"churn_rate": 0.265}',                                    # JSON, but not a tool call
    '{"name": "Alice", "age": 30}',                             # has "name" but no arguments
    "```python\nprint(1)\n```",                                 # example code, not a call
    "{not json",
    "",
])
def test_leaves_real_answers_alone(content):
    assert _text_tool_calls(content) == []


def _events(res):
    return [(b.split("\n")[0][7:], json.loads(b.split("\n", 1)[1][6:])) for b in res.text.strip().split("\n\n")]


def test_agent_runs_a_text_tool_call_then_answers(client, monkeypatch):
    replies = iter([
        {"role": "assistant", "content": json.dumps(CALL)},
        {"role": "assistant", "content": "The answer is 42."},
    ])
    seen_histories = []
    async def fake_llm(messages, tools=None):
        seen_histories.append([dict(m) for m in messages])
        return next(replies)
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)

    res = client.post("/api/agent/", json={"message": "what is 6*7?"}, headers={"Origin": APP_ORIGIN})
    events = _events(res)
    kinds = [e for e, _ in events]
    assert kinds == ["start", "code", "result", "final"]
    assert events[2][1]["stdout"].strip() == "42"
    assert events[3][1]["content"] == "The answer is 42."
    # the model's second turn sees a proper tool call + tool result, not stray JSON text
    second = seen_histories[1]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "run_python"
    assert second[-1]["role"] == "tool"


def _fake_replies(monkeypatch, replies):
    replies = iter(replies)
    feedback = []
    async def fake_llm(messages, tools=None):
        if messages[-1]["role"] == "tool":
            feedback.append(messages[-1]["content"])
        return next(replies)
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)
    return feedback


def _upload(client):
    rows = [f"{20 + i % 50},{'ab'[i % 2]},{'yes' if i % 3 == 0 else 'no'}" for i in range(120)]
    csv = "age,plan,churn\n" + "\n".join(rows)
    return client.post("/api/upload/", files={"file": ("c.csv", io.BytesIO(csv.encode()), "text/csv")},
                       headers={"Origin": APP_ORIGIN}).json()["dataset_id"]


@pytest.mark.parametrize("name, args, expected", [
    ("train_model", {"target": "churn"}, "model = train_model(df, target='churn')"),
    ("train_model", {"df": "df.drop(columns=['id'])", "target": "y", "model": "gb"},
     "model = train_model(df.drop(columns=['id']), target='y', model='gb')"),
    ("list_files", {}, "print(list_files(''))"),
    ("load_file", {"name": "sales.csv"}, "df2 = load_file('sales.csv')"),
    ("load_file", {}, None),
    ("delete_everything", {"all": True}, None),
])
def test_helper_calls_translate_to_code(name, args, expected):
    code = _helper_call_as_code(name, args)
    if expected is None:
        assert code is None
    else:
        assert code.startswith(expected)


def test_agent_runs_a_helper_called_as_a_tool(client, monkeypatch):
    # exactly what qwen2.5-coder:7b replies to "train a model to predict churn"
    _fake_replies(monkeypatch, [
        {"role": "assistant", "content": json.dumps({"name": "train_model", "arguments": {"target": "churn"}})},
        {"role": "assistant", "content": "Accuracy is reported above."},
    ])
    res = client.post("/api/agent/", json={"message": "predict churn", "dataset_id": _upload(client)},
                      headers={"Origin": APP_ORIGIN})
    events = _events(res)
    assert [e for e, _ in events][:4] == ["start", "code", "result", "final"]
    assert events[1][1]["code"] == "model = train_model(df, target='churn')"
    assert events[2][1]["error"] is None
    assert "CLASSIFICATION" in events[2][1]["stdout"]


def test_agent_corrects_an_unknown_tool(client, monkeypatch):
    feedback = _fake_replies(monkeypatch, [
        {"role": "assistant", "content": json.dumps({"name": "summarize", "arguments": {}})},
        {"role": "assistant", "content": json.dumps(CALL)},
        {"role": "assistant", "content": "Done: 42."},
    ])
    res = client.post("/api/agent/", json={"message": "go"}, headers={"Origin": APP_ORIGIN})
    assert [e for e, _ in _events(res)] == ["start", "code", "result", "final"]
    assert "only tool is run_python" in feedback[0]


FENCE = "`" * 3


def test_python_blocks_run_when_nothing_has_run_yet(client, monkeypatch):
    # seen after a correction: the plan written as prose + python blocks
    plan = f"Let's inspect first.\n{FENCE}python\nx = 6 * 7\n{FENCE}\nThen:\n{FENCE}python\nprint(x)\n{FENCE}"
    answer = f"It's 42. Example: {FENCE}python\nprint(x)\n{FENCE}"
    _fake_replies(monkeypatch, [
        {"role": "assistant", "content": plan},
        {"role": "assistant", "content": answer},
    ])
    res = client.post("/api/agent/", json={"message": "go"}, headers={"Origin": APP_ORIGIN})
    events = _events(res)
    assert [e for e, _ in events] == ["start", "code", "result", "final"]
    assert events[2][1]["stdout"].strip() == "42"
    assert "Example" in events[3][1]["content"]   # code in the final answer is not re-run


def test_prose_preamble_then_fenced_call():
    content = f'Let me check.\n\n{FENCE}python\n{{"name": "run_python", "arguments": {{"code": "print(1)"}}}}\n{FENCE}'
    assert _text_tool_calls(content)[0]["function"]["arguments"] == {"code": "print(1)"}


def test_python_blocks_are_not_calls_by_default():
    assert _text_tool_calls(f"Here's how:\n{FENCE}python\nprint(1)\n{FENCE}") == []


@pytest.mark.parametrize("content", [
    # seen from qwen2.5-coder:7b after one step had already run
    f"Let's calculate it:\n{FENCE}python\nprint(1)\n{FENCE}\nPlease run the above code to get the churn rates.",
    f"{FENCE}python\nprint(1)\n{FENCE}\nYou can run this to see the result.",
    f"Execute the following snippet:\n{FENCE}python\nprint(1)\n{FENCE}",
])
def test_handoff_replies_become_calls_even_after_code_ran(content):
    assert _text_tool_calls(content)[0]["function"]["arguments"] == {"code": "print(1)"}


@pytest.mark.parametrize("content", [
    f"The rate is 26%. I computed it with:\n{FENCE}python\nprint(1)\n{FENCE}",
    "The model runs well: accuracy 0.68.",
])
def test_final_answers_with_example_code_are_not_rerun(content):
    assert _text_tool_calls(content) == []


@pytest.mark.parametrize("content", [
    # seen from qwen2.5-coder:7b: the call, then an explanation
    json.dumps(CALL) + "\n\nTo interpret the bar chart:\n- The x-axis represents the plans.",
    "Let me compute that. " + json.dumps(CALL) + " Then I'll summarise.",
])
def test_call_object_with_surrounding_prose(content):
    assert _text_tool_calls(content) == [{"function": {"name": "run_python", "arguments": {"code": CODE}}}]


def test_python_block_after_a_failed_step_is_run_as_the_fix(client, monkeypatch):
    # seen live: step errored, then "Let's update the code…" + a fixed block
    _fake_replies(monkeypatch, [
        {"role": "assistant", "content": json.dumps({"name": "run_python", "arguments": {"code": "1 / 0"}})},
        {"role": "assistant", "content": f"Let's update the code to avoid that:\n{FENCE}python\nprint(6 * 7)\n{FENCE}"},
        {"role": "assistant", "content": "It's 42."},
    ])
    res = client.post("/api/agent/", json={"message": "go"}, headers={"Origin": APP_ORIGIN})
    events = _events(res)
    assert [e for e, _ in events] == ["start", "code", "result", "code", "result", "final"]
    assert events[2][1]["error"] and events[4][1]["stdout"].strip() == "42"


def test_system_prompt_describes_values_from_the_data(client, monkeypatch):
    seen = []
    async def fake_llm(messages, tools=None):
        seen.append(messages[0]["content"])
        return {"role": "assistant", "content": "ok"}
    monkeypatch.setattr(agent_router, "chat_ollama", fake_llm)
    client.post("/api/agent/", json={"message": "hi", "dataset_id": _upload(client), "context": "churn(str)"},
                headers={"Origin": APP_ORIGIN})
    assert "- churn: text, 2 values: 'no', 'yes'" in seen[0]
    assert "- age: numeric" in seen[0]
