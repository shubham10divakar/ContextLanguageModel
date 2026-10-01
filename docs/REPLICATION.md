# Replication notes

## 1. Paper vs reference code vs this repo

| Topic | Paper says | Reference code does (as described) | This repo |
|---|---|---|---|
| Edit mechanism | model edits a context file with bash | `[[CTX_TURN i role=r]]` file, `parse_back` regex, protect first 2 messages | same (`clm/context_file.py`) |
| New roles ("notes", Fig. 3b) | the model creates a new role | every non-assistant role becomes `user` | same; this has safety consequences (see §4) |
| Tool-call structure after an edit | n/a | not rebuilt; history becomes plain text | same |
| "Zero-shot" | no training | 25/50/75% nudges, 90% urgent nudge, rollback ×6 (BCP) / ×50 (EdgeBench), ledger, final-turn notice | same knobs in `CLMConfig`, so you can ablate them |
| Edit gate | not discussed | `fit` default, `shrink` on BCP | same |
| Efficiency advantage | Eq. 6 credit on the whole trajectory | dual channel: credit only on edit-turn tokens; `w_eff` defaults to 1.0 unless the env var is set | both (`mode="paper"` / `"dual_channel"`), `w_eff=0.25` default |
| SCR span tails | relocate surviving spans | last 16 tokens of each span re-prefilled | same (`min_extend=16`) |
| SCR linear-attention layers | restore the pre-edit recurrent state | `SSM_MODE=fork` | **not implemented** (the reference SCR here covers full-attention models only) |
| System prompt | — | `prompts.yaml` | **clean-room reconstruction**; expect behavioural drift |
| ContextBench | 4 tasks, levels in Fig. 15 | not released ("coming soon") | reimplemented from App. D; KV Store pressure comes out ≈0.7× the paper's at the same record count |
| Subagents / swarm | §5.1.2 | removed from release | not implemented |

Choices this repo makes where neither the paper nor the description pins things down:

* Rendering includes reasoning (`include_reasoning_in_file=True`), but reasoning is not re-sent to the API
  (`send_reasoning=False`), matching chat templates that strip old reasoning.
* Rollback never drops environment-pushed messages (ContextBench ops), so a stream cannot silently lose an input.
* ContextBench grading reads the context as of the agent's **last acknowledged operation**. An input pushed in
  just before an overflow is not credited. This reproduces Base ≈ 1/pressure on Needle Retention.
* Turn counting: edit-only turns (file changed, empty stdout, exit 0) do not count toward `max_steps`.
  `max_total_turns` is a hard cap.

## 2. Replication ladder

Each rung uses the output of the one before it. Rungs 0–3 fit on a single 12 GB GPU or a cheap API budget.

| Rung | What | Command / module | Expected |
|---|---|---|---|
| 0 | Harness, graders, FLOPs constants | `pytest -q`; `python scripts/sanity_fig2.py` | all pass; oracle 1.0, Base collapses above 1× |
| 1 | ContextBench with a real model: Base vs Summary vs CLM, 4 seeds | `clm bench --policy llm --strategy {base,summary,clm} ...` | CLM ≥ baselines, especially on KV Store and Sudoku above 1× (Fig. 2) |
| 2 | In-context steering (Fig. 7) | `--instructions "Monitor your context size ... compact ... about 4000 tokens"` with Y ∈ {16k, 24k, 32k} | first-compaction size tracks Y |
| 3 | Skill evolution on KV Store (Fig. 8) | `clm.icl.evolve.evolve(ContextBenchSource(...), LLMProposer(...))` | dev accuracy and/or cost improves over the empty skill |
| 4 | FLOPs and SCR analysis on your traces | `clm flops runs/...`; `clm.scr.reference.scr_divergence` | prefix hit rate falls on edit turns; SCR recovers part of it |
| 5 | RL (Table 2) | `clm.rl` advantages plugged into verl / slime / TRL | needs multi-GPU; the paper used 64 H200s |

Rungs that need infrastructure not included here: BrowseComp-Plus (search corpus and judge), TerminalBench 2.1
and TBLite (Harbor + Docker), EdgeBench, Software World. The harness is benchmark-agnostic. Implement a
`clm.tasks.Task` and a container-backed sandbox with the `LocalSandbox` interface to add them.

## 3. Hardware notes (RTX 3060, 12 GB)

* **Model size.** Qwen3-8B with 4-bit weights (~5 GB) plus a 32K-token KV cache (36 layers × 8 KV heads × 128 ×
  2 × 2 bytes ≈ 147 KB per token, ≈ 4.8 GB) just fits. Qwen3-4B leaves headroom. The paper's Qwen3.6-27B needs an
  API provider or a rented GPU.
* **vLLM** does not run natively on Windows; use WSL2. Enable tool parsing for your model family (see the vLLM
  docs for the right `--tool-call-parser`) and a reasoning parser if the model thinks.
* **llama.cpp**: `llama-server -m <model>.gguf -c 32768 -ngl 99 --jinja` serves `/v1` with tool calls.
* **Ollama**: the default context window is small. Set it to 32768 (for example `OLLAMA_CONTEXT_LENGTH=32768`),
  or the server silently truncates and every budget number is wrong.
* **Prefix-cache accounting.** Servers that do not report `cached_tokens` fall back to local tokenization in the
  FLOPs replay. Pass `--hf-tokenizer <model id>` to `clm flops` to use the model's own chat template.
* **Process spawning is slow on Windows** (Defender plus Python start-up). Expect 0.3–3 s per command.

## 4. Things the replication surfaced

1. **Header forgery.** Turn content is written to the file unescaped. Any text that reaches the context — a
   tool output, a web page, a log line — can contain `[[CTX_TURN n role=assistant]]`. After the *next* edit,
   for any reason, parse-back splits there and creates a forged assistant (or user) turn. No model cooperation
   is needed. Reproduced in `tests/test_harness.py::test_header_injection_forges_assistant_turn`. Mitigation:
   `escape_headers=True` (off by default, to stay faithful to the reference).
2. **Role laundering.** Because every non-assistant role becomes `user`, text the model writes under an invented
   role — or a summary of untrusted tool output — re-enters as user-authority text on later turns.
3. **Edits are expensive under prefix caching.** An oracle that rewrites its whole context every turn costs
   more FLOPs than append-only at low pressure (README table). Edit timing and position matter as much as
   edit content.
