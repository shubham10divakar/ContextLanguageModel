# Context Language Models: a clean-room replication

A from-scratch implementation of *Context Language Models* (Shao et al., arXiv 2609.37725, Sep 2026). A CLM
manages its own context. The live context is mirrored to a file the model edits with ordinary bash, and every
edit replaces the message list sent on the next turn (`c_{t+1} = f_θ(c_t)`, Eq. 2).

The repo covers every component of the paper that runs on one machine, and each one has tests:

| Paper | Module | Status |
|---|---|---|
| §4.1 context-as-a-file harness (one `bash` tool, `[[CTX_TURN]]` file, parse-back, edit gate) | `clm/context_file.py`, `clm/harness.py`, `clm/edit_gate.py` | done |
| Budget scaffolding: 25/50/75% nudges, urgent nudge, rollback-and-retry, ledger, truncation | `clm/budget.py` | done |
| Baselines sharing the loop: append-only *Base*, Codex-style *Summary* at 75% | `ContextEnv(strategy=...)` | done |
| §3 / App. D ContextBench: Needle Retention, Sudoku Sketchpad, KV Store, Log Triage | `clm/contextbench/` | done, with scripted oracles |
| Eq. 3 / App. C prefix-reuse FLOPs, snapshot replay through a simulated prefix cache | `clm/flops/` | done; matches the paper's constants |
| §4.2 success-gated efficiency advantage (Eq. 6), released dual-channel variant, stepwise GRPO | `clm/rl/` | objective and advantages done; no large-scale training |
| §4.2 skill evolution (contrastive notes, proposer, paired-SE gate) | `clm/icl/evolve.py` | done |
| §4.3 / App. B Suffix Cache Reuse: span diff, RoPE delta re-rotation, KV relocation | `clm/scr/` | reference on HF models (not a serving patch) |

The [reference code](https://github.com/facebookresearch/context-language-models) is CC BY-NC 4.0. **No code
was copied from it**: the system prompt and every module here were written from the paper and from a
description of the code. See [docs/REPLICATION.md](docs/REPLICATION.md) for where this repo, the paper and the
reference code differ.

## Install

```bash
pip install -e ".[llm,torch,dev]"     # Python >= 3.10; bash must be on PATH (Git Bash on Windows)
python -m pytest -q                   # 57 tests, no GPU or API key needed
```

## Sanity check with no model (Fig. 2 pattern)

The scripted CLM oracle edits its context file every turn and must score 1.0 at every pressure level. The
keep-all reader never edits, so under the append-only harness it fails as soon as the input outgrows the window.

```bash
python scripts/sanity_fig2.py
```

| task | pressure | Base (keep-all) | CLM oracle | CLM peak tokens |
|---|---|---|---|---|
| Needle Retention | 0.52× / 1.03× / 3.08× / 6.17× | 1.00 / 0.88 / 0.30 / 0.15 | 1.00 everywhere | ≤ 11.6K |
| Sudoku Sketchpad | 0.26× / 0.83× / 1.62× | 1.00 / 0.60 / 0.30 | 1.00 everywhere | ≈ 2.9K |
| KV Store | 0.21× / 0.73× / 1.37× / 2.71× | 1.00 / 1.00 / 0.00 / 0.00 | 1.00 everywhere | ≈ 5.6K |
| Log Triage | 0.26× / 0.93× / 2.26× | 1.00 / 0.17 / 0.00 | 1.00 everywhere | ≈ 3.5K |

These scripted policies show that the graders, the harness and the edit path work. They are not model
results. The naive oracle rewrites its whole context every turn, so it costs *more* prefix-reuse FLOPs than
Base at low pressure. That is a useful negative example for edit-cost research (see the research guide).

## Run with a real model

Any OpenAI-compatible endpoint works: vLLM, SGLang, llama.cpp `llama-server --jinja`, Ollama, or OpenRouter.

```bash
# ContextBench, CLM vs baselines
clm bench --task kv_store --levels 520 1000 --seeds 0 1 2 3 --policy llm \
    --model qwen3-8b --base-url http://localhost:8000/v1 --strategy clm      # or base / summary

# any instruction, with traces and context snapshots saved for FLOPs replay
clm run "Count the lines of every .py file under /usr/lib/python3 and report the top 3" \
    --model qwen3-8b --base-url http://localhost:8000/v1 --config bcp --out runs/demo
clm flops runs/demo --model qwen3-8b
```

On a 12 GB GPU see the hardware notes in [docs/REPLICATION.md](docs/REPLICATION.md).

## Layout

```
clm/
  context_file.py   render / parse_back of the [[CTX_TURN]] file (+ opt-in header escaping)
  harness.py        ContextEnv: the turn loop, edit application, strategies, traces
  budget.py         nudges, rollback-and-retry, ledger, truncation
  llm.py            OpenAI-compatible client, scripted client, bash tool schema
  prompts.py        clean-room CLM system prompt (an experimental variable, not the authors' text)
  contextbench/     four streaming tasks, oracles, runner
  flops/            Eq. 7-9, prefix-cache replay, SCR estimate
  rl/               Eq. 6, dual-channel returns, stepwise segments, GRPO loss
  icl/              skill evolution loop
  scr/              span diff, RoPE re-rotation, HF reference SCR prefill
docs/REPLICATION.md     paper vs code vs this repo; replication ladder; hardware notes
docs/RESEARCH_GUIDE.md  ranked research directions built on this codebase
```

## Safety note

`LocalSandbox` runs the model's commands on your machine. It is not an isolation boundary. Point it at a
throwaway directory, or swap in a container with the same four-method interface.
