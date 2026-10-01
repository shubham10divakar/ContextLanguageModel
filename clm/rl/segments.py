"""Stepwise training segments from CLM traces.

Context edits break the append-only structure of a trajectory: the final
context may no longer contain the inputs under which earlier actions were
generated. Stepwise GRPO therefore trains every LLM call on its own
recorded prompt (the snapshot taken before that call) and assigns it the
trajectory's advantage. Edit turns are flagged so the dual-channel variant
can put efficiency credit on them only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..trace import load_snapshots
from .advantage import dual_channel_returns, efficiency_advantage, group_normalize, shaped_reward


@dataclass
class Segment:
    traj_id: str
    call: int
    prompt: list[dict]        # API messages sent for this call
    response: dict            # content / reasoning / tool_calls
    is_edit_turn: bool
    advantage: float = 0.0    # filled by assign_advantages (outcome + efficiency on edit turns)


@dataclass
class Trajectory:
    traj_id: str
    segments: list[Segment]
    reward: float
    success: bool
    cost: float               # prefix-reuse FLOPs


def load_trajectory(run_dir: str | Path, success_key: str = "accuracy", success_threshold: float = 1.0,
                    cost: float | None = None, edit_flag: str = "edit_only") -> Trajectory:
    """Build a trajectory from a harness run directory (snapshots + trace + result)."""
    run_dir = Path(run_dir)
    snaps = [s for s in load_snapshots(run_dir) if s["response"].get("kind", "turn") == "turn"]
    turns = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    turns = [t for t in turns if t.get("type") == "turn"]
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    edit_by_order = [bool(t.get(edit_flag)) for t in turns]
    segs = [Segment(run_dir.name, s["turn"], s["messages"], s["response"],
                    edit_by_order[i] if i < len(edit_by_order) else False) for i, s in enumerate(snaps)]
    score = result.get("grade", {}).get(success_key)
    success = score is not None and score >= success_threshold
    stats = result.get("stats", {})
    reward = shaped_reward(success, no_tool_call=stats.get("no_tool_call", 0) > 0,
                           format_failure=stats.get("format_errors", 0) > 0)
    if cost is None:
        from ..flops import replay_run

        cost = replay_run(run_dir)["prefix_reuse_flops"]
    return Trajectory(run_dir.name, segs, reward, success, cost)


def assign_advantages(group: list[Trajectory], w_eff: float = 0.25, mode: str = "dual_channel") -> list[Segment]:
    """Assign stepwise advantages to all segments of one rollout group."""
    a_out = group_normalize([t.reward for t in group])
    a_eff = efficiency_advantage([t.cost for t in group], [t.success for t in group])
    out = []
    for i, traj in enumerate(group):
        for seg in traj.segments:
            if mode == "paper":
                seg.advantage = float(a_out[i] + w_eff * a_eff[i])
            elif mode == "dual_channel":
                seg.advantage = float(dual_channel_returns(a_out[i], a_eff[i], np.array([seg.is_edit_turn]), w_eff)[0])
            elif mode == "outcome":
                seg.advantage = float(a_out[i])
            else:
                raise ValueError(mode)
            out.append(seg)
    return out
