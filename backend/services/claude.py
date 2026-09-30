# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Optional cloud model: Claude via the Anthropic API.

The agent keeps its conversation in Ollama/OpenAI shape (role "system" /
"user" / "assistant" with tool_calls / "tool"). This module translates that to
the Messages API on the way out, and Claude's reply back on the way in, so the
agent loop doesn't care which model answers.
"""

import json

from services.llm import OllamaError

MAX_TOKENS = 16000
# Keep these reply blocks so they can be echoed back unchanged next turn
# (thinking blocks must be returned as-is when Claude continues a tool loop)
_ECHO_BLOCKS = ("text", "thinking", "redacted_thinking", "tool_use")


async def chat(messages: list[dict], tools: list[dict] | None, model: str, api_key: str) -> dict:
    """One model turn. Returns an Ollama-style assistant message."""
    import anthropic   # imported lazily: local-only users never load the SDK

    system, converted = to_claude_messages(messages)
    params = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": converted,
    }
    if tools:
        params["tools"] = [to_claude_tool(t) for t in tools]

    client = anthropic.AsyncAnthropic(api_key=api_key)
    try:
        if model == "claude-opus-5":
            # If Opus 5 declines a request, retry it server-side on the model
            # Anthropic recommends for that case instead of returning a refusal
            response = await client.beta.messages.create(
                **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            response = await client.messages.create(**params)
    except anthropic.AuthenticationError:
        raise OllamaError("The Anthropic API key was rejected. Check it in Model settings.") from None
    except anthropic.PermissionDeniedError:
        raise OllamaError(f"This API key doesn't have access to {model}. Pick another model in Model settings.") from None
    except anthropic.NotFoundError:
        raise OllamaError(f"The cloud model {model} wasn't found. Pick another model in Model settings.") from None
    except anthropic.RateLimitError:
        raise OllamaError("The cloud model is rate-limited right now (or the account is out of credit). "
                          "Wait a moment, or switch back to the local model.") from None
    except anthropic.APIStatusError as e:
        raise OllamaError(f"The cloud model returned an error ({e.status_code}). Try again, "
                          "or switch back to the local model.") from None
    except anthropic.APIConnectionError:
        raise OllamaError("Can't reach the Anthropic API. Check your internet connection, "
                          "or switch back to the local model.") from None
    finally:
        await client.close()

    return from_claude_response(response)


def to_claude_tool(tool: dict) -> dict:
    fn = tool["function"]
    return {"name": fn["name"], "description": fn["description"], "input_schema": fn["parameters"]}


def to_claude_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Ollama-style history → (system prompt, Messages API messages).

    Tool calls stored without ids (e.g. recovered from a local model's text)
    get generated ones; each "tool" message is paired, in order, with the
    pending call it answers, and consecutive results go back in ONE user message.
    """
    system_parts, out, pending = [], [], []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            system_parts.append(m.get("content") or "")
        elif role == "user":
            out.append({"role": "user", "content": m.get("content") or "(empty message)"})
        elif role == "assistant":
            raw = m.get("claude_content")
            if raw:
                content = raw
                pending = [b["id"] for b in raw if b.get("type") == "tool_use"]
            else:
                content, pending = [], []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for j, call in enumerate(m.get("tool_calls") or []):
                    fn = call.get("function", {})
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {"code": args}
                    call_id = f"toolu_df_{i}_{j}"
                    pending.append(call_id)
                    content.append({"type": "tool_use", "id": call_id, "name": fn.get("name", "run_python"),
                                    "input": args if isinstance(args, dict) else {"code": str(args)}})
                if not content:
                    content = [{"type": "text", "text": "(no reply)"}]
            out.append({"role": "assistant", "content": content})
        elif role == "tool" and pending:
            text = m.get("content") or ""
            result = {"type": "tool_result", "tool_use_id": pending.pop(0), "content": text}
            try:
                if "error" in json.loads(text):
                    result["is_error"] = True
            except (ValueError, TypeError):
                pass
            last = out[-1] if out else None
            if last and last["role"] == "user" and isinstance(last["content"], list) \
                    and last["content"] and last["content"][0].get("type") == "tool_result":
                last["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
    return "\n\n".join(p for p in system_parts if p), out


def from_claude_response(response) -> dict:
    """Messages API reply → Ollama-style assistant message (plus the raw blocks to echo back)."""
    if response.stop_reason == "refusal":
        return {"role": "assistant", "content": "The cloud model declined to answer this request. "
                                                "Try rephrasing it, or switch to the local model."}
    blocks = [b.model_dump(exclude_none=True) for b in response.content if b.type in _ECHO_BLOCKS]
    text = "\n".join(b["text"] for b in blocks if b["type"] == "text").strip()
    calls = [{"id": b["id"], "function": {"name": b["name"], "arguments": b["input"]}}
             for b in blocks if b["type"] == "tool_use"]
    if response.stop_reason == "max_tokens" and not calls:
        text += "\n\n(The answer was cut off because it hit the length limit.)"
    message = {"role": "assistant", "content": text, "claude_content": blocks}
    if calls:
        message["tool_calls"] = calls
    return message
