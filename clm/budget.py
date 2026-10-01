"""Budget controller: nudges, rollback-and-retry, ledger, output truncation.

The paper calls the agent "zero-shot", but the reference harness supplies a
fair amount of budget scaffolding. This module reproduces it:

* usable limit = budget - reserve (2,048 tokens by default);
* escalating nudges at 25/50/75% of the limit; each fires once and re-arms
  after the context drops back below that ratio;
* an urgent nudge on *every* turn near the limit: either past a fixed ratio
  (0.9) or, adaptively, when free room < max(10% of limit, 2x the largest of
  the last 3 tool outputs);
* rollback on overflow: the newest turns are dropped until ``reserve`` tokens
  are free, and the commands whose turns were dropped go into a ledger;
* head/tail truncation of long tool outputs;
* a "submit now" notice in the final turns.
"""

from __future__ import annotations

import json
from collections import deque

from .config import CLMConfig
from .tokens import TokenCounter

SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def command_of(msg: dict) -> str:
    for call in msg.get("tool_calls") or []:
        try:
            return json.loads(call.get("arguments") or "{}").get("command", "")
        except (json.JSONDecodeError, AttributeError):
            return str(call.get("arguments"))
    return (msg.get("content") or "")[:200]


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n... [{omitted} characters truncated] ...\n{text[-tail:]}"


class BudgetController:
    def __init__(self, cfg: CLMConfig, counter: TokenCounter, ctx_path: str = "the context file"):
        self.cfg = cfg
        self.counter = counter
        self.ctx_path = ctx_path
        self.fired: set[float] = set()
        self.recent_obs: deque[int] = deque(maxlen=3)
        self.retries = 0

    @property
    def limit(self) -> int:
        return self.cfg.limit

    # ------------------------------------------------------------------ nudges
    def record_observation(self, tokens: int) -> None:
        self.recent_obs.append(tokens)

    def urgent(self, used: int) -> bool:
        free = self.limit - used
        if self.cfg.urgent_mode == "ratio":
            return used >= self.cfg.persistent_nudge_ratio * self.limit
        if self.cfg.urgent_mode == "adaptive":
            biggest = max(self.recent_obs, default=0)
            return free < max(0.1 * self.limit, 2 * biggest)
        return False

    def nudges(self, used: int) -> list[str]:
        """Budget messages to append to this turn's observation footer."""
        out: list[str] = []
        for r in self.cfg.nudge_ratios:          # re-arm ratios we dropped below
            if used < r * self.limit:
                self.fired.discard(r)
        crossed = [r for r in self.cfg.nudge_ratios if used >= r * self.limit and r not in self.fired]
        if self.urgent(used):
            self.fired.update(crossed)
            out.append(
                f"URGENT: context is ~{used}/{self.limit} tokens; only ~{self.limit - used} free. "
                f"Before anything else, edit {self.ctx_path} to drop or summarize turns you no longer need."
            )
        elif crossed:
            self.fired.update(crossed)
            pct = int(100 * max(crossed))
            out.append(
                f"Note: context has passed {pct}% of the budget (~{used}/{self.limit} tokens). "
                f"Consider compacting {self.ctx_path}: keep what you still need, summarize or delete the rest."
            )
        return out

    def final_notice(self, steps_left: int) -> str | None:
        if steps_left <= self.cfg.final_turns_warning:
            return (f"Only {steps_left} step(s) left. Finish now: print your final answer and run "
                    f"`echo {SUBMIT_SENTINEL}` (final output goes after that line).")
        return None

    # ---------------------------------------------------------------- overflow
    def over_limit(self, messages: list[dict]) -> bool:
        return self.counter.count(messages) > self.limit

    def rollback(self, protected: list[dict], history: list[dict]) -> tuple[list[dict], list[str]]:
        """Drop the newest droppable turns until ``reserve`` tokens are free.

        Messages flagged ``env=True`` (operations pushed by the environment)
        are never dropped, so a task stream cannot silently lose an input.
        """
        h = list(history)
        dropped: list[str] = []
        target = self.limit - self.cfg.reserve_tokens

        def drop_newest() -> bool:
            for i in range(len(h) - 1, -1, -1):
                if not h[i].get("env"):
                    m = h.pop(i)
                    if m["role"] == "assistant":
                        dropped.append(command_of(m))
                    return True
            return False

        while self.counter.count(protected + h) > target:
            if not drop_newest():
                break
        dropped.extend(_repair_tool_pairs(h))
        self.retries += 1
        return h, dropped

    def truncate(self, text: str) -> str:
        return truncate_middle(text, self.cfg.observation_max_chars)


def _repair_tool_pairs(h: list[dict]) -> list[str]:
    """Remove tool calls whose results were dropped (and orphan results); return their commands."""
    answered = {m.get("tool_call_id") for m in h if m["role"] == "tool"}
    called = {c["id"] for m in h if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    removed: list[str] = []
    i = 0
    while i < len(h):
        m = h[i]
        if m["role"] == "tool" and m.get("tool_call_id") not in called:
            h.pop(i)
            continue
        if m["role"] == "assistant" and m.get("tool_calls"):
            if any(c["id"] not in answered for c in m["tool_calls"]):
                removed.append(command_of(h.pop(i)))
                continue
        i += 1
    return removed
