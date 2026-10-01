"""Shared machinery for ContextBench streaming tasks (paper Sec. 3, App. D).

The environment delivers the input as a sequence of operations. Each
operation arrives as a new user message, so it is in the agent's context
before the agent can act on it; no harness can truncate it on the way in.
The agent asks for the next operation by printing ``READY_FOR_NEXT_OP``.
Grading reads only the agent's context: answers held only in files are not
credited.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

from ..tasks import Task
from ..tokens import TokenCounter

READY = "READY_FOR_NEXT_OP"
CONTEXT_LIMIT = 32768

WORDS = ("amber anchor apple arrow aspen atlas basil beacon birch blaze bramble breeze cactus canyon cedar "
         "cinder citrus clover cobalt cobble comet copper coral crater crimson crystal delta drift dune ember "
         "falcon fern fjord flint forest frost garnet glacier granite harbor hazel heron indigo iris ivory jade "
         "juniper kelp lagoon lantern lattice lava lichen lotus lunar magnet maple marble meadow mesa mistral "
         "moss nectar nimbus oak obsidian ocean onyx orbit orchid pebble pepper pewter pine plume prairie "
         "prism quartz quill raven reef ridge river saffron sage sequoia shale sierra silver slate spruce "
         "summit tango thistle thunder topaz tundra umber valley velvet violet walnut willow zephyr").split()


def hex8(rng: random.Random) -> str:
    return f"{rng.getrandbits(32):08x}"


def words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


@dataclass
class Op:
    title: str
    body: str
    kind: str
    payload: dict = field(default_factory=dict)


def answer_blocks(text: str, attr: str) -> dict[str, str]:
    """Map ``<<<ANSWER attr=X>>> ... <<<ANSWER END>>>`` blocks to their (last) content."""
    out: dict[str, str] = {}
    pat = re.compile(r"<<<ANSWER\s+" + attr + r"=([^>\s]+)>>>\s*\n(.*?)\n?\s*<<<ANSWER END>>>", re.S)
    for m in pat.finditer(text):
        out[m.group(1)] = m.group(2).strip()
    return out


class StreamTask(Task):
    """A task delivered as a stream of operations, graded per operation from context."""

    name = "stream"
    header_instruction = ""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)
        self.ops: list[Op] = self.generate()
        self.idx = 0
        self.finished = False
        self.op_scores: dict[int, float] = {}
        self.last_context: str = ""   # context when the agent last acknowledged an operation

    # -- subclass API ---------------------------------------------------------
    def generate(self) -> list[Op]:
        raise NotImplementedError

    def score_op(self, i: int, context: str) -> float | None:
        """Score op ``i`` right after the agent acknowledges it; None = not graded per op."""
        return None

    def final_metrics(self, context: str) -> dict:
        graded = [i for i, op in enumerate(self.ops) if self.is_graded(op)]
        scores = [self.op_scores.get(i, 0.0) for i in graded]
        return {"accuracy": sum(scores) / len(scores) if scores else 0.0, "n_graded": len(graded)}

    def is_graded(self, op: Op) -> bool:
        return False

    # -- protocol ----------------------------------------------------------
    @property
    def instruction(self) -> str:
        return (f"{self.header_instruction}\n\n"
                f"Protocol: operations arrive one at a time as new messages. When you have fully handled the "
                f"current operation, run `echo {READY}` (you may append it to the same command that does the "
                f"work, e.g. `...; echo {READY}`). The next operation then arrives. Everything is graded from "
                f"what is in your context; answers held only in files are not credited. There are "
                f"{len(self.ops)} operations in total.")

    def render(self, i: int) -> str:
        op = self.ops[i]
        remaining = len(self.ops) - i - 1
        return f"=== {op.title} ({remaining} op(s) remaining after this) ===\n{op.body}"

    def initial_messages(self) -> list[str]:
        return [self.render(0)] if self.ops else []

    def on_turn(self, env, command, result) -> list[str]:
        if self.finished or READY not in result.stdout:
            return []
        context = env.context_text()
        self.last_context = context
        score = self.score_op(self.idx, context)
        if score is not None:
            self.op_scores[self.idx] = score
        self.idx += 1
        if self.idx >= len(self.ops):
            self.finished = True
            return []
        return [self.render(self.idx)]

    def is_done(self) -> bool:
        return self.finished

    def grade(self, env) -> dict:
        # Final-context metrics read the context as of the last acknowledged operation, so an
        # input pushed in right before an overflow does not count as retained.
        m = self.final_metrics(self.last_context)
        m.update(task=self.name, seed=self.seed, ops_total=len(self.ops),
                 ops_completed=self.idx if not self.finished else len(self.ops))
        return m

    # -- diagnostics ---------------------------------------------------------
    def total_input_tokens(self, counter: TokenCounter | None = None) -> int:
        counter = counter or TokenCounter()
        return counter.count_text(self.instruction) + sum(counter.count_text(self.render(i))
                                                          for i in range(len(self.ops)))

    def pressure(self, counter: TokenCounter | None = None) -> float:
        """Total input / 32,768 (paper Fig. 15)."""
        return self.total_input_tokens(counter) / CONTEXT_LIMIT
