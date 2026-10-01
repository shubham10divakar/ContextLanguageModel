"""Run ContextBench levels with an LLM or a scripted oracle.

Examples::

    # sanity: the scripted CLM oracle should score 1.0 everywhere
    python -m clm.contextbench.run --task kv_store --levels 130 1000 --policy oracle

    # a real model behind any OpenAI-compatible server (vLLM, llama.cpp, Ollama, ...)
    python -m clm.contextbench.run --task kv_store --levels 520 --seeds 0 1 2 3 \\
        --policy llm --model qwen3-8b --base-url http://localhost:8000/v1 --strategy clm
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ..config import CLMConfig
from ..harness import ContextEnv
from ..llm import OpenAICompatLLM, ScriptedLLM
from ..sandbox import LocalSandbox
from . import LEVELS, TASKS, make_task
from .oracles import clm_oracle, keep_all_oracle


def build_llm(args, task_name: str, cfg: CLMConfig):
    if args.policy == "oracle":
        return ScriptedLLM(clm_oracle(task_name))
    if args.policy == "keep_all":
        return ScriptedLLM(keep_all_oracle(task_name))
    return OpenAICompatLLM(args.model, base_url=args.base_url, api_key=args.api_key,
                           temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_tokens)


def run_one(task_name: str, level: int, seed: int, args, out: Path | None) -> dict:
    cfg = CLMConfig.load(args.config, strategy=args.strategy, skill_path=args.skill,
                         extra_instructions=args.instructions)
    task = make_task(task_name, level, seed)
    run_dir = out / f"{task_name}_L{level}_s{seed}_{args.strategy}_{args.policy}" if out else None
    sandbox = LocalSandbox()
    try:
        env = ContextEnv(cfg, build_llm(args, task_name, cfg), task, sandbox=sandbox, run_dir=run_dir)
        res = env.run()
    finally:
        sandbox.cleanup()
    rec = {"task": task_name, "level": level, "seed": seed, "strategy": args.strategy, "policy": args.policy,
           "pressure": round(task.pressure(), 3), "status": res.status, "accuracy": res.grade.get("accuracy"),
           "ops_completed": res.grade.get("ops_completed"), "ops_total": res.grade.get("ops_total"),
           "turns": res.turns, "llm_calls": res.llm_calls, "peak_tokens": res.peak_tokens, "stats": res.stats}
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--levels", type=int, nargs="*", help="generator settings (default: all paper levels)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--strategy", choices=["clm", "base", "summary"], default="clm")
    ap.add_argument("--policy", choices=["llm", "oracle", "keep_all"], default="oracle")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--config", default="contextbench")
    ap.add_argument("--skill", help="path to a SKILL.md appended to the task (in-context steering)")
    ap.add_argument("--instructions", default="", help="extra instruction sentence (Sec. 5.2 steering)")
    ap.add_argument("--out", type=Path, default=Path("runs/contextbench"))
    args = ap.parse_args(argv)
    if args.policy == "llm" and not args.model:
        ap.error("--model is required with --policy llm")

    tasks = list(TASKS) if args.task == "all" else [args.task]
    records = []
    for t in tasks:
        for level in args.levels or LEVELS[t]:
            accs = []
            for seed in args.seeds:
                rec = run_one(t, level, seed, args, args.out)
                records.append(rec)
                accs.append(rec["accuracy"] or 0.0)
                print(json.dumps(rec))
            mean = sum(accs) / len(accs)
            se = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1) / len(accs)) if len(accs) > 1 else 0
            print(f"## {t} level={level} pressure={records[-1]['pressure']}x acc={mean:.3f} +/- {se:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "summary.jsonl", "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
