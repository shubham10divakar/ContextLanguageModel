"""Log Triage: lookup and count queries over a stream of service logs.

Batches of 14-54 log lines, then 24 queries mixing per-service level counts,
global level counts and request-id lookups. Graded like KV Store, with
ANSWER blocks keyed by ``qid``.
"""

from __future__ import annotations

from .common import Op, StreamTask, answer_blocks, hex8, words

SERVICES = ["billing", "auth", "db-proxy", "search", "cache", "gateway", "payments", "notify"]
LEVELS = ["INFO", "WARN", "ERROR"]
LEVEL_WEIGHTS = [0.6, 0.25, 0.15]


class LogTriage(StreamTask):
    name = "log_triage"
    header_instruction = (
        "TASK: Log Triage. You will receive batches of service log lines and later queries about ALL lines seen "
        "so far. Answer query <qid> by putting this block in your context:\n"
        "<<<ANSWER qid=<qid>>>>\n<answer>\n<<<ANSWER END>>>\n"
        "Counts are plain integers; service names are given exactly as they appear in the logs.")

    def __init__(self, n_lines: int = 160, n_queries: int = 24, batch_range=(14, 54), seed: int = 0):
        self.n_lines = n_lines
        self.n_queries = n_queries
        self.batch_range = batch_range
        super().__init__(seed)

    def generate(self) -> list[Op]:
        rng = self.rng
        t = 14 * 3600
        self.lines = []
        for _ in range(self.n_lines):
            t += rng.randint(1, 5)
            rec = {"level": rng.choices(LEVELS, LEVEL_WEIGHTS)[0], "service": rng.choice(SERVICES),
                   "req": hex8(rng)}
            hh, mm, ss = (t // 3600) % 24, (t // 60) % 60, t % 60
            rec["text"] = (f"2026-06-20T{hh:02d}:{mm:02d}:{ss:02d}Z [{rec['level']}] {rec['service']} "
                           f"req={rec['req']} {words(rng, 10)} #{hex8(rng)}")
            self.lines.append(rec)
        ops = []
        i, b = 0, 0
        while i < self.n_lines:
            n = min(rng.randint(*self.batch_range), self.n_lines - i)
            tag = f"{b:04d}#{hex8(rng)}"
            body = (f"<<<LOG-BATCH {tag} BEGIN -- offload this whole block this turn>>>\n"
                    + "\n".join(r["text"] for r in self.lines[i:i + n]) + f"\n<<<LOG-BATCH {tag} END>>>")
            ops.append(Op(title=f"log batch (lines {i}-{i + n - 1})", body=body, kind="batch"))
            i += n
            b += 1
        for q in range(self.n_queries):
            question, answer = self._question(rng)
            body = f"QUERY {q}: {question}\n(Answer with an ANSWER block for qid={q}.)"
            ops.append(Op(title=f"query {q}", body=body, kind="query", payload={"qid": str(q), "answer": answer}))
        return ops

    def _question(self, rng) -> tuple[str, str]:
        kind = rng.choice(["svc_count", "svc_count", "total_count", "lookup"])
        if kind == "svc_count":
            lvl, svc = rng.choice(LEVELS[1:]), rng.choice(SERVICES)
            n = sum(1 for r in self.lines if r["level"] == lvl and r["service"] == svc)
            return f'How many [{lvl}] log lines are from service "{svc}"?', str(n)
        if kind == "total_count":
            lvl = rng.choice(LEVELS[1:])
            return f"How many [{lvl}] log lines are there in total?", str(sum(r["level"] == lvl for r in self.lines))
        rec = rng.choice(self.lines)
        return f"Which service logged the line with request id req={rec['req']}?", rec["service"]

    def is_graded(self, op: Op) -> bool:
        return op.kind == "query"

    def score_op(self, i: int, context: str) -> float | None:
        op = self.ops[i]
        if op.kind != "query":
            return None
        return float(answer_blocks(context, "qid").get(op.payload["qid"]) == op.payload["answer"])
