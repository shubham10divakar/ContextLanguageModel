"""Run traces and context snapshots.

A snapshot of the exact message list is saved before every LLM call
(``context_snapshots/turn_XXXX.json``), together with the server usage that
came back. The FLOPs replay (``clm.flops``) reads these to rebuild prompt
lengths and prefix reuse turn by turn.
"""

from __future__ import annotations

import json
from pathlib import Path


class TraceWriter:
    def __init__(self, run_dir: str | Path | None):
        self.run_dir = Path(run_dir) if run_dir else None
        self.events: list[dict] = []
        if self.run_dir:
            (self.run_dir / "context_snapshots").mkdir(parents=True, exist_ok=True)
            self._trace = open(self.run_dir / "trace.jsonl", "w", encoding="utf-8")
        else:
            self._trace = None

    def snapshot(self, turn: int, messages: list[dict], response: dict) -> None:
        rec = {"turn": turn, "messages": messages, "response": response}
        if self.run_dir:
            path = self.run_dir / "context_snapshots" / f"turn_{turn:04d}.json"
            path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        else:
            self.events.append({"type": "snapshot", **rec})

    def event(self, **rec) -> None:
        if self._trace:
            self._trace.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._trace.flush()
        else:
            self.events.append(rec)

    def write_json(self, name: str, obj) -> None:
        if self.run_dir:
            (self.run_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                                             encoding="utf-8")

    def close(self) -> None:
        if self._trace:
            self._trace.close()
            self._trace = None


def load_snapshots(run_dir: str | Path) -> list[dict]:
    files = sorted((Path(run_dir) / "context_snapshots").glob("turn_*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]
