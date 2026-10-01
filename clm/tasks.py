"""Task interface used by the harness.

A task supplies the (protected) instruction, optional environment messages
streamed into the editable context, and a grader. Streaming tasks such as
ContextBench push their next operation whenever the agent prints
``READY_FOR_NEXT_OP``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import ContextEnv
    from .sandbox import CommandResult


class Task:
    instruction: str = ""

    def setup(self, env: "ContextEnv") -> None:
        """Prepare files in the sandbox before the first turn."""

    def initial_messages(self) -> list[str]:
        """Environment messages placed in the editable context before the first turn."""
        return []

    def on_turn(self, env: "ContextEnv", command: str, result: "CommandResult") -> list[str]:
        """Called after every command; returns new environment messages to append."""
        return []

    def is_done(self) -> bool:
        return False

    def grade(self, env: "ContextEnv") -> dict:
        return {}


class InstructionTask(Task):
    """A plain one-shot task: an instruction, graded by an optional callback on the final output."""

    def __init__(self, instruction: str, grader=None):
        self.instruction = instruction
        self.grader = grader

    def grade(self, env):
        if self.grader is None:
            return {}
        return self.grader(env.final_output)
