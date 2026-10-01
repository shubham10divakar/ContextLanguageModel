"""LLM clients.

``OpenAICompatLLM`` talks to any OpenAI-compatible chat endpoint (vLLM,
SGLang, llama.cpp server, Ollama, OpenRouter, ...). ``ScriptedLLM`` replays a
deterministic policy and is used for tests and oracle runs.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Protocol

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one bash command in the sandbox and return its stdout, stderr and exit code.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run."}},
            "required": ["command"],
        },
    },
}

_ids = itertools.count()


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class LLMResponse:
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None


class LLM(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int | None = None) -> LLMResponse: ...


def bash_call(command: str, content: str = "", reasoning: str | None = None) -> LLMResponse:
    """Convenience constructor for a response that runs one bash command."""
    return LLMResponse(content=content, reasoning=reasoning,
                       tool_calls=[ToolCall(f"call_{next(_ids)}", "bash", json.dumps({"command": command}))])


def to_api_messages(messages: list[dict], send_reasoning: bool = False,
                    no_trailing_assistant: bool = False) -> list[dict]:
    """Convert internal messages to the OpenAI chat format."""
    out = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for c in m["tool_calls"]]
            if send_reasoning and m.get("reasoning"):
                msg["reasoning_content"] = m["reasoning"]
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m.get("content") or ""})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    if no_trailing_assistant and out and out[-1]["role"] == "assistant":
        # Some APIs reject a trailing assistant turn (possible after an edit):
        # re-send its text as a user "context mirror" message.
        last = out.pop()
        out.append({"role": "user", "content": "[context mirror of your last turn]\n" + (last.get("content") or "")})
    return out


class OpenAICompatLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 4096,
                 extra_body: dict | None = None, timeout: float = 600.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                             api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    def complete(self, messages, tools=None, max_tokens=None) -> LLMResponse:
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature,
                      top_p=self.top_p, max_tokens=max_tokens or self.max_tokens)
        if tools:
            kwargs["tools"] = tools
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        r = self.client.chat.completions.create(**kwargs)
        choice = r.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        calls = [ToolCall(c.id or f"call_{next(_ids)}", c.function.name, c.function.arguments or "{}")
                 for c in (msg.tool_calls or [])]
        usage = r.usage
        cached = None
        if usage is not None and getattr(usage, "prompt_tokens_details", None) is not None:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", None)
        return LLMResponse(content=msg.content or "", reasoning=reasoning, tool_calls=calls,
                           prompt_tokens=getattr(usage, "prompt_tokens", None),
                           completion_tokens=getattr(usage, "completion_tokens", None),
                           cached_tokens=cached, finish_reason=choice.finish_reason)


class ScriptedLLM:
    """Deterministic policy: ``policy(messages) -> LLMResponse | str (bash command)``."""

    def __init__(self, policy: Callable[[list[dict]], "LLMResponse | str"] | list):
        if isinstance(policy, list):
            it = iter(policy)
            self.policy = lambda _m: next(it)
        else:
            self.policy = policy
        self.calls = 0

    def complete(self, messages, tools=None, max_tokens=None) -> LLMResponse:
        self.calls += 1
        r = self.policy(messages)
        return bash_call(r) if isinstance(r, str) else r
