"""``clm`` command line.

    clm run "Find the largest file under /usr/share and report its size" \\
        --model qwen3-8b --base-url http://localhost:8000/v1 --config bcp --out runs/demo
    clm bench --task kv_store --levels 520 --policy oracle
    clm flops runs/demo --model qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import sys


def _run(argv):
    from .config import CLMConfig
    from .harness import ContextEnv
    from .llm import OpenAICompatLLM
    from .sandbox import LocalSandbox
    from .tasks import InstructionTask

    ap = argparse.ArgumentParser(prog="clm run")
    ap.add_argument("instruction", help="task text, or @path to read it from a file")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--config", default=None, help="config name (bcp, edgebench, contextbench) or YAML path")
    ap.add_argument("--strategy", choices=["clm", "base", "summary"])
    ap.add_argument("--workdir", help="sandbox working directory (default: fresh temp dir)")
    ap.add_argument("--skill")
    ap.add_argument("--instructions", default=None)
    ap.add_argument("--out", default=None, help="run directory for traces and snapshots")
    a = ap.parse_args(argv)
    text = open(a.instruction[1:], encoding="utf-8").read() if a.instruction.startswith("@") else a.instruction
    cfg = CLMConfig.load(a.config, strategy=a.strategy, skill_path=a.skill, extra_instructions=a.instructions)
    llm = OpenAICompatLLM(a.model, base_url=a.base_url, api_key=a.api_key, temperature=cfg.temperature,
                          top_p=cfg.top_p, max_tokens=cfg.max_tokens)
    res = ContextEnv(cfg, llm, InstructionTask(text), sandbox=LocalSandbox(a.workdir), run_dir=a.out).run()
    print(json.dumps(res.to_dict(), indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        _run(rest)
    elif cmd == "bench":
        from .contextbench.run import main as bench

        bench(rest)
    elif cmd == "flops":
        from .flops.prefix_reuse import main as flops

        flops(rest)
    else:
        raise SystemExit(f"unknown command {cmd!r}; expected run | bench | flops")


if __name__ == "__main__":
    main()
