"""The CLM harness: one bash tool, context mirrored to an editable file.

Each turn (reference: ``clm_agent/harness.py`` -> ``ContextEnv.step``)::

    messages = [system, task] + history         # first two are protected
    loop:
      1. budget check (rollback-and-retry on overflow, ledger, final turn)
      2. response = LLM(messages)
      3. command  = response.tool_call.bash.command
      4. rendered = render_editable(history) -> write to the context file
      5. run the command in the sandbox      (the model may edit the file)
      6. read the file back
      7. if it changed: candidate = parse_back(file); if the edit gate allows,
         history = candidate                  # the context is REPLACED
      8. append assistant turn + tool result
         (output + "[context: ~N/limit tokens]" + edit receipt + nudges)
      9. edit-only turns (file changed, no stdout, exit 0) do not count as steps

``strategy`` selects the context policy so baselines share the same loop:
``clm`` (the method), ``base`` (append-only; overflow ends the run) and
``summary`` (Codex-style: summarize everything at 75% of the budget).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .budget import SUBMIT_SENTINEL, BudgetController
from .config import CLMConfig
from .context_file import parse_back, render_editable, was_edited
from .edit_gate import check_edit
from .llm import BASH_TOOL, LLM, LLMResponse, to_api_messages
from .prompts import FORMAT_ERROR, SUMMARY_PROMPT, system_prompt
from .sandbox import CommandResult, LocalSandbox
from .tasks import Task
from .tokens import TokenCounter
from .trace import TraceWriter

FENCE_RE = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```", re.S)


@dataclass
class RunResult:
    status: str
    final_output: str | None
    steps: int
    turns: int
    llm_calls: int
    peak_tokens: int
    stats: dict = field(default_factory=dict)
    grade: dict = field(default_factory=dict)
    run_dir: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ContextEnv:
    def __init__(self, cfg: CLMConfig, llm: LLM, task: Task, sandbox: LocalSandbox | None = None,
                 run_dir: str | Path | None = None, counter: TokenCounter | None = None):
        if cfg.strategy not in ("clm", "base", "summary"):
            raise ValueError(f"unknown strategy {cfg.strategy}")
        self.cfg = cfg
        self.llm = llm
        self.task = task
        self.sandbox = sandbox or LocalSandbox()
        self.counter = counter or TokenCounter(include_reasoning=cfg.send_reasoning)
        self.budget = BudgetController(cfg, self.counter, self.sandbox.ctx_path)
        self.trace = TraceWriter(run_dir)
        self.run_dir = str(run_dir) if run_dir else None
        self.system = system_prompt(cfg.strategy, self.sandbox.ctx_path)
        self.ledger: list[str] = []
        self.history: list[dict] = []
        self.final_output: str | None = None
        self.stats: Counter = Counter()
        self.turn = 0          # assistant turns taken
        self.steps = 0         # turns counted against max_steps
        self.llm_calls = 0
        self.peak_tokens = 0
        self._skill = Path(cfg.skill_path).read_text(encoding="utf-8") if cfg.skill_path else ""

    # ------------------------------------------------------------ messages
    def task_message(self) -> str:
        text = self.task.instruction
        if self._skill:
            text += "\n\n# Skill\n" + self._skill
        if self.cfg.extra_instructions:
            text += "\n\n" + self.cfg.extra_instructions
        if self.ledger:
            text += ("\n\n[LEDGER] These commands produced output that overflowed your context and their turns "
                     "were rolled back. Do not rerun them unchanged; narrow or redirect their output:\n"
                     + "\n".join(f"- {c[:300]}" for c in self.ledger))
        return text

    @property
    def protected(self) -> list[dict]:
        return [{"role": "system", "content": self.system}, {"role": "user", "content": self.task_message()}]

    def messages(self) -> list[dict]:
        return self.protected + self.history

    def used_tokens(self) -> int:
        return self.counter.count(self.messages())

    def context_text(self) -> str:
        """Everything the model will see next turn, as plain text (used by graders).

        Tool-call commands are decoded from their JSON arguments so text the
        model wrote inside a command counts the same as text in its message.
        """
        parts = []
        for m in self.messages():
            parts.append(m.get("content") or "")
            for c in m.get("tool_calls") or []:
                try:
                    parts.append(json.loads(c.get("arguments") or "{}").get("command", ""))
                except (json.JSONDecodeError, AttributeError):
                    parts.append(str(c.get("arguments", "")))
        return "\n\n".join(parts)

    def push_user(self, text: str, env: bool = False) -> None:
        if self.history and self.history[-1]["role"] == "user":
            self.history[-1]["content"] += "\n\n" + text
            self.history[-1]["env"] = self.history[-1].get("env", False) or env
        else:
            self.history.append({"role": "user", "content": text, "env": env})

    # ------------------------------------------------------------ one turn
    @staticmethod
    def extract_command(resp: LLMResponse) -> tuple[str | None, bool]:
        """Return (command, via_tool_call)."""
        for call in resp.tool_calls:
            if call.name != "bash":
                continue
            try:
                cmd = json.loads(call.arguments or "{}").get("command")
            except json.JSONDecodeError:
                cmd = None
            if cmd:
                return cmd, True
        m = FENCE_RE.search(resp.content or "")
        if m and m.group(1).strip():
            return m.group(1).strip(), False
        return None, False

    def step(self, resp: LLMResponse) -> None:
        self.turn += 1
        cmd, via_tool = self.extract_command(resp)
        assistant = {"role": "assistant", "content": resp.content or "", "reasoning": resp.reasoning,
                     "tool_calls": []}
        if cmd is None:
            self.history.append(assistant)
            self.push_user(FORMAT_ERROR)
            self.stats["format_errors"] += 1
            if not resp.tool_calls:
                self.stats["no_tool_call"] += 1
            self.steps += 1
            return
        if via_tool:
            first = next(c for c in resp.tool_calls if c.name == "bash")
            assistant["tool_calls"] = [first.as_dict()]
            if len(resp.tool_calls) > 1:
                self.stats["extra_tool_calls_ignored"] += len(resp.tool_calls) - 1

        clm = self.cfg.strategy == "clm"
        rendered = None
        if clm:
            rendered = render_editable(self.history, self.cfg.include_reasoning_in_file,
                                       escape=self.cfg.escape_headers)
            self.sandbox.write_context(rendered)
        res = self.sandbox.run(cmd, timeout=self.cfg.command_timeout)

        receipt, edited = "", False
        if clm:
            back = self.sandbox.read_context()
            if was_edited(rendered, back):
                candidate = parse_back(back)
                cur = self.used_tokens()
                new = self.counter.count(self.protected + candidate)
                decision = check_edit(self.cfg.edit_gate, cur, new, self.budget.limit)
                if decision.accepted:
                    self.history = candidate
                    edited = True
                    self.stats["edits_applied"] += 1
                    receipt = f"[context edit applied: ~{cur} -> ~{new} tokens]"
                else:
                    self.stats["edits_rejected"] += 1
                    receipt = decision.reason
                self.trace.event(type="edit", turn=self.turn, accepted=decision.accepted,
                                 tokens_before=cur, tokens_after=new)

        obs = self.budget.truncate(res.output).rstrip() or "(no output)"
        obs += f"\n(exit code {res.returncode})"
        self.history.append(assistant)
        if via_tool:
            self.history.append({"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": obs})
        else:
            self.push_user(obs)
        obs_msg = self.history[-1]
        self.budget.record_observation(self.counter.count_text(obs))

        edit_only = edited and not res.stdout.strip() and res.returncode == 0
        if edit_only:
            self.stats["edit_only_turns"] += 1
        else:
            self.steps += 1

        if SUBMIT_SENTINEL in res.stdout:
            after = res.stdout.split(SUBMIT_SENTINEL, 1)[1]
            self.final_output = after.lstrip("\n").strip()

        new_env = self.task.on_turn(self, cmd, res)

        used = self.used_tokens()
        footer = [f"[context: ~{used}/{self.budget.limit} tokens]"]
        if receipt:
            footer.append(receipt)
        if clm:
            footer += self.budget.nudges(used)
        notice = self.budget.final_notice(self.cfg.max_steps - self.steps)
        if notice and self.final_output is None:
            footer.append(notice)
        obs_msg["content"] += "\n" + "\n".join(footer)
        for text in new_env:
            self.push_user(text, env=True)
        self.peak_tokens = max(self.peak_tokens, self.used_tokens())
        self.trace.event(type="turn", turn=self.turn, command=cmd, returncode=res.returncode,
                         edited=edited, edit_only=edit_only, tokens=self.used_tokens(),
                         prompt_tokens=resp.prompt_tokens, cached_tokens=resp.cached_tokens,
                         completion_tokens=resp.completion_tokens)

    # ------------------------------------------------------------ LLM calls
    def call_llm(self, messages: list[dict], tools: list[dict] | None, kind: str = "turn") -> LLMResponse:
        api = to_api_messages(messages, self.cfg.send_reasoning, self.cfg.no_trailing_assistant)
        resp = self.llm.complete(api, tools=tools, max_tokens=self.cfg.max_tokens)
        self.llm_calls += 1
        self.trace.snapshot(self.llm_calls, api, {
            "kind": kind, "content": resp.content, "reasoning": resp.reasoning,
            "tool_calls": [c.as_dict() for c in resp.tool_calls], "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens, "cached_tokens": resp.cached_tokens})
        self.counter.calibrate(messages, resp.prompt_tokens)
        return resp

    def compact_summary(self) -> None:
        """Codex-style compaction: replace history by a model-written summary."""
        i = len(self.history)
        while i > 0 and self.history[i - 1]["role"] == "user" and self.history[i - 1].get("env"):
            i -= 1
        pending = self.history[i:]
        resp = self.call_llm(self.messages() + [{"role": "user", "content": SUMMARY_PROMPT}], None, kind="summary")
        self.history = [{"role": "user", "content": "[Summary of your earlier work]\n" + (resp.content or "")}]
        for m in pending:
            self.push_user(m["content"], env=True)
        self.stats["summaries"] += 1

    def rollback(self) -> None:
        before = self.used_tokens()
        self.history, dropped = self.budget.rollback(self.protected, self.history)
        self.ledger.extend(d for d in dropped if d)
        self.stats["rollbacks"] += 1
        self.push_user(f"[harness] Your next request would have exceeded the context limit (~{before}/"
                       f"{self.budget.limit} tokens), so your newest turn(s) were rolled back. Compact your "
                       f"context now by editing {self.sandbox.ctx_path}. See the LEDGER in the task message.")

    # ------------------------------------------------------------ main loop
    def reset(self) -> None:
        self.task.setup(self)
        for text in self.task.initial_messages():
            self.push_user(text, env=True)

    def run(self) -> RunResult:
        self.reset()
        status = None
        final_turn = False
        while status is None:
            if self.turn >= self.cfg.max_total_turns:
                status = "max_turns"
                break
            if self.steps >= self.cfg.max_steps:
                status = "max_steps"
                break
            if self.task.is_done():
                status = "task_done"
                break
            strategy = self.cfg.strategy
            if strategy == "summary" and self.used_tokens() >= self.cfg.summary_trigger_ratio * self.budget.limit:
                self.compact_summary()
            if self.budget.over_limit(self.messages()):
                if strategy == "clm":
                    if self.budget.retries < self.cfg.max_num_retry_on_limit:
                        self.rollback()
                    else:
                        self.rollback()
                        self.push_user("[harness] Context retries exhausted. This is your final turn: "
                                       f"submit now with `echo {SUBMIT_SENTINEL}` and your answer.")
                        final_turn = True
                    if self.budget.over_limit(self.messages()):
                        status = "context_overflow"
                        break
                elif strategy == "summary":
                    self.compact_summary()
                    if self.budget.over_limit(self.messages()):
                        status = "context_overflow"
                        break
                else:
                    status = "context_overflow"
                    break
            try:
                resp = self.call_llm(self.messages(), [BASH_TOOL])
            except Exception as e:  # noqa: BLE001 - surface any client failure as a run status
                self.trace.event(type="error", error=repr(e))
                status = "error"
                break
            self.step(resp)
            if self.final_output is not None:
                status = "submitted"
            elif final_turn:
                status = "context_limit"
        return self.finish(status)

    def finish(self, status: str) -> RunResult:
        try:
            grade = self.task.grade(self)
        finally:
            self.trace.close()
        result = RunResult(status=status, final_output=self.final_output, steps=self.steps, turns=self.turn,
                           llm_calls=self.llm_calls, peak_tokens=self.peak_tokens, stats=dict(self.stats),
                           grade=grade, run_dir=self.run_dir)
        self.trace.write_json("result.json", result.to_dict())
        return result


def run_task(cfg: CLMConfig, llm: LLM, task: Task, **kwargs) -> RunResult:
    return ContextEnv(cfg, llm, task, **kwargs).run()
