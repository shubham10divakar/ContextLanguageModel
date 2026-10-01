"""Context-as-a-file serialization (paper Sec. 4.1).

The editable part of the context (everything after the protected system and
task messages) is rendered to plain text with one header per turn::

    [[CTX_TURN 1 role=assistant]]
    <reasoning + content + "bash {command json}">

    [[CTX_TURN 2 role=tool]]
    <observation>

After the model's command runs, the file is read back. If it changed, it is
parsed into a new message list that *replaces* the editable history.

Parse-back rules (these follow the reference code, not the paper's prose):
  * ``assistant`` stays ``assistant``; every other role, including ``tool``
    and invented ones such as ``notes``, becomes ``user``.
  * Tool-call structure is not rebuilt: edited history is plain role + text.
  * Text before the first header becomes a user note.
  * Empty turns are dropped; consecutive same-role turns are merged so the
    chat template stays valid.
  * Turn numbers are labels only and are ignored.

Internal message format (OpenAI-like dicts)::

    {"role": "system" | "user", "content": str}
    {"role": "assistant", "content": str, "reasoning": str | None,
     "tool_calls": [{"id": str, "name": str, "arguments": str}]}
    {"role": "tool", "tool_call_id": str, "content": str}
"""

from __future__ import annotations

import json
import re
from typing import Iterable

HEADER_RE = re.compile(r"^\[\[CTX_TURN\s+(\d+)\s+role=([A-Za-z_][A-Za-z0-9_\-]*)\]\]\s*$", re.M)

Message = dict


def header(index: int, role: str) -> str:
    return f"[[CTX_TURN {index} role={role}]]"


def format_tool_call(call: dict) -> str:
    """Flatten one tool call as ``<name> {json args}`` (e.g. ``bash {"command": "ls"}``)."""
    args = call.get("arguments", "")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return f"{call.get('name', 'bash')} {args}"


def message_text(msg: Message, include_reasoning: bool = True) -> str:
    """Flatten a message into the plain text shown in the context file."""
    parts: list[str] = []
    if include_reasoning and msg.get("reasoning"):
        parts.append(f"<reasoning>\n{msg['reasoning'].strip()}\n</reasoning>")
    content = msg.get("content") or ""
    if content.strip():
        parts.append(content.strip())
    for call in msg.get("tool_calls") or []:
        parts.append(format_tool_call(call))
    return "\n".join(parts)


def escape_headers(text: str) -> str:
    """Neutralize header-shaped lines inside turn content (prefix a backslash).

    Without this, any text that reaches the context (a tool output, a fetched
    web page) can contain ``[[CTX_TURN n role=assistant]]`` and, on the next
    edit, be parsed back as a forged assistant or user turn. The reference
    design does not escape; this is an opt-in mitigation.
    """
    return HEADER_RE.sub(lambda m: "\\" + m.group(0), text)


def render_editable(messages: Iterable[Message], include_reasoning: bool = True, escape: bool = False) -> str:
    """Render editable turns (``messages[2:]`` of the live context) to file text."""
    blocks = []
    for i, msg in enumerate(messages, 1):
        body = message_text(msg, include_reasoning)
        if escape:
            body = escape_headers(body)
        blocks.append(f"{header(i, msg['role'])}\n{body}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _append(out: list[Message], role: str, body: str) -> None:
    if out and out[-1]["role"] == role:
        out[-1]["content"] += "\n\n" + body
    else:
        out.append({"role": role, "content": body})


def parse_back(text: str) -> list[Message]:
    """Parse an (edited) context file into plain role/text messages."""
    out: list[Message] = []
    matches = list(HEADER_RE.finditer(text))
    preamble = text[: matches[0].start()] if matches else text
    if preamble.strip():
        _append(out, "user", preamble.strip())
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if not body:
            continue
        role = "assistant" if m.group(2) == "assistant" else "user"
        _append(out, role, body)
    return out


def normalize_text(text: str) -> str:
    """Whitespace-insensitive form used to decide whether the file was edited."""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").strip().split("\n"))


def was_edited(rendered: str, read_back: str | None) -> bool:
    if read_back is None:
        return False
    return normalize_text(rendered) != normalize_text(read_back)
