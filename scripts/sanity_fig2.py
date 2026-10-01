"""Fig. 2-style sanity sweep with scripted policies (no LLM needed).

CLM oracle (edits its context file) vs keep-all reader under the append-only
``base`` harness. The oracle must stay at 1.0 at every pressure; base must
collapse once pressure exceeds 1x. Prints a markdown table.

    python scripts/sanity_fig2.py --out runs/sanity
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from clm.config import CLMConfig
from clm.contextbench import make_task
from clm.contextbench.oracles import clm_oracle, keep_all_oracle
from clm.flops import replay_run
from clm.harness import ContextEnv
from clm.llm import ScriptedLLM
from clm.sandbox import LocalSandbox

SWEEP = {
    "needle_retention": [4, 8, 24, 48],
    "sudoku_sketchpad": [4, 15, 30],
    "kv_store": [130, 520, 1000, 2000],
    "log_triage": [160, 640, 1600],
}


def run(task_name, level, strategy, policy, out: Path):
    task = make_task(task_name, level, seed=0)
    run_dir = out / f"{task_name}_{level}_{strategy}"
    sb = LocalSandbox(tempfile.mkdtemp(prefix="clm_sanity_"))
    try:
        res = ContextEnv(CLMConfig.load("contextbench", strategy=strategy), ScriptedLLM(policy(task_name)), task,
                         sandbox=sb, run_dir=run_dir).run()
    finally:
        sb.cleanup()
    pf = replay_run(run_dir, "qwen3.6-27b")["prefix_reuse_pflops"]
    return task.pressure(), res.grade["accuracy"], res.peak_tokens, pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/sanity"))
    args = ap.parse_args()
    print("| task | level | pressure | base acc | CLM-oracle acc | CLM peak tokens | base PFLOPs | CLM PFLOPs |")
    print("|---|---|---|---|---|---|---|---|")
    for t, levels in SWEEP.items():
        for lv in levels:
            pr, b_acc, _, b_pf = run(t, lv, "base", keep_all_oracle, args.out)
            _, c_acc, c_peak, c_pf = run(t, lv, "clm", clm_oracle, args.out)
            print(f"| {t} | {lv} | {pr:.2f}x | {b_acc:.2f} | {c_acc:.2f} | {c_peak} | {b_pf:.2f} | {c_pf:.2f} |",
                  flush=True)


if __name__ == "__main__":
    main()
