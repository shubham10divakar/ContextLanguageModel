"""Needle Retention: selective verbatim retention.

Chunks of ~4K tokens, each with 2-8 needle lines and 140 filler lines. The
score is the fraction of all needle lines present verbatim in the final
context. The generator caps the needle count so that everything to retain
fits in half of the usable limit (paper App. D).
"""

from __future__ import annotations

from .common import CONTEXT_LIMIT, Op, StreamTask, hex8, words
from ..tokens import TokenCounter


class NeedleRetention(StreamTask):
    name = "needle_retention"
    header_instruction = (
        "TASK: Needle Retention. You will receive chunks of text. Each chunk starts with NEEDLES lines that you "
        "must keep verbatim in your context until the very end, followed by a FILLER-BLOCK that you do not need. "
        "Your final score is the fraction of all needle lines that appear verbatim in your context when the "
        "stream ends. No answer is required.")

    def __init__(self, n_chunks: int = 4, filler_lines: int = 140, needles_range=(2, 8), seed: int = 0,
                 usable_limit: int = CONTEXT_LIMIT - 2048):
        self.n_chunks = n_chunks
        self.filler_lines = filler_lines
        self.needles_range = needles_range
        self.usable_limit = usable_limit
        super().__init__(seed)

    def generate(self) -> list[Op]:
        rng = self.rng
        counts = [rng.randint(*self.needles_range) for _ in range(self.n_chunks)]
        needles = [[f"[n{c:05d}i{j:02d}#{hex8(rng)}] {words(rng, 10)}." for j in range(k)]
                   for c, k in enumerate(counts)]
        # cap: retained material must fit in half the usable limit with a 10% margin
        counter = TokenCounter()
        cap = int(0.5 * self.usable_limit / 1.1)
        per_line = counter.count_text(needles[0][0]) + 1
        while sum(len(n) for n in needles) * per_line > cap:
            c = max(range(len(needles)), key=lambda i: len(needles[i]))
            if len(needles[c]) <= 1:
                break
            needles[c].pop()
        self.needles = [line for chunk in needles for line in chunk]
        ops = []
        for c, chunk in enumerate(needles):
            tag = f"{c:05d}#{hex8(rng)}"
            filler = [f"[f{c:05d}x{j:03d}#{hex8(rng)}] {words(rng, 12)}" for j in range(self.filler_lines)]
            body = ("NEEDLES (keep these lines verbatim in your context):\n" + "\n".join(chunk) + "\n"
                    f"<<<FILLER-BLOCK {tag} START -- delete this entire block (through its END line) "
                    f"from your context this turn>>>\n" + "\n".join(filler) + f"\n<<<FILLER-BLOCK {tag} END>>>")
            ops.append(Op(title=f"chunk {c + 1}/{self.n_chunks}", body=body, kind="chunk"))
        return ops

    def final_metrics(self, context: str) -> dict:
        lines = set(line.strip() for line in context.splitlines())
        kept = sum(1 for n in self.needles if n in lines)
        return {"accuracy": kept / len(self.needles) if self.needles else 0.0,
                "needles_kept": kept, "needles_total": len(self.needles)}
