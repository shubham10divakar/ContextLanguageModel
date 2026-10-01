"""KV Store: exact recall through offloading and retrieval.

Batches of 100 ``SET`` records with random 24-word values, then 24 ``GET``
queries over keys from any batch. Each GET is graded when the agent
acknowledges it: its ANSWER block for that key must be in context and match
the stored value exactly.
"""

from __future__ import annotations

from .common import Op, StreamTask, answer_blocks, hex8, words


class KVStore(StreamTask):
    name = "kv_store"
    header_instruction = (
        "TASK: KV Store. You will receive batches of SET records (`SET <key> = <value>`) and later GET queries. "
        "For each `GET <key>`, the value most recently SET for that key must appear in your context as\n"
        "<<<ANSWER key=<key>>>>\n<value>\n<<<ANSWER END>>>\n"
        "(for example print it with a command so it shows up in the output). Values must match exactly.")

    def __init__(self, n_records: int = 130, n_queries: int = 24, batch_size: int = 100,
                 value_words: int = 24, seed: int = 0):
        self.n_records = n_records
        self.n_queries = n_queries
        self.batch_size = batch_size
        self.value_words = value_words
        super().__init__(seed)

    def generate(self) -> list[Op]:
        rng = self.rng
        self.values = {f"K{i:05d}": f"{words(rng, self.value_words)} #{hex8(rng)}" for i in range(self.n_records)}
        keys = list(self.values)
        ops = []
        for b, lo in enumerate(range(0, self.n_records, self.batch_size)):
            hi = min(lo + self.batch_size, self.n_records) - 1
            tag = f"{b:04d}#{hex8(rng)}"
            lines = [f"SET {k} = {self.values[k]}" for k in keys[lo: hi + 1]]
            body = (f"<<<SET-BATCH {tag} BEGIN -- offload this whole block this turn>>>\n" + "\n".join(lines)
                    + f"\n<<<SET-BATCH {tag} END>>>")
            ops.append(Op(title=f"set batch (keys {lo}-{hi})", body=body, kind="set"))
        picks = rng.sample(keys, min(self.n_queries, len(keys)))
        for q, key in enumerate(picks):
            body = f"GET {key}\n(Report the value you stored for {key} as an ANSWER block for that key.)"
            ops.append(Op(title=f"get {q}", body=body, kind="get", payload={"key": key}))
        return ops

    def is_graded(self, op: Op) -> bool:
        return op.kind == "get"

    def score_op(self, i: int, context: str) -> float | None:
        op = self.ops[i]
        if op.kind != "get":
            return None
        key = op.payload["key"]
        return float(answer_blocks(context, "key").get(key) == self.values[key])
