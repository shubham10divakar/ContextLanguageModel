"""Skill evolution for CLMs (paper Eq. 5, Sec. 4.2, App. E "In-context evolution").

    s* = argmax_s E_x[R(tau(x; s))]

Each round:
  1. run the current SKILL.md on training tasks;
  2. build contrastive notes from paired runs: a success and a failure on the
     same task, or two successes where one was cheaper;
  3. a proposer (a stronger model = "assisted", or the agent itself =
     "self-evolution") writes N full rewrites, each citing the steps it
     targets and predicting its effect;
  4. drop malformed candidates;
  5. run every candidate on the dev split;
  6. gate: accept if the paired accuracy gain d > SE (per-task paired standard
     error), or if it ties (|d| <= SE) at lower cost.
A lineage stops after ``patience`` consecutive rounds without acceptance.

Plug in your own tasks by subclassing ``TaskSource``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Episode:
    reward: float
    cost: float
    notes: str = ""            # short trajectory description for the proposer

    @property
    def success(self) -> bool:
        return self.reward >= 1.0


class TaskSource:
    train_ids: list[str] = []
    dev_ids: list[str] = []
    test_ids: list[str] = []

    def run(self, task_id: str, skill_dir: Path, out_dir: Path, rep: int) -> Episode:
        raise NotImplementedError


@dataclass
class Candidate:
    skill: str
    rationale: str = ""


SKILL_RE = re.compile(r"<<<SKILL>>>\s*\n(.*?)\n\s*<<<END SKILL>>>", re.S)
RATIONALE_RE = re.compile(r"<<<RATIONALE>>>\s*\n(.*?)\n\s*<<<END RATIONALE>>>", re.S)

PROPOSER_PROMPT = """\
You are improving a SKILL.md that tells an agent how to manage its own context (the agent can edit a file that \
mirrors its live context). Below are the current skill and contrastive notes from paired runs of the agent on \
training tasks: successes vs failures on the same task, and cheap vs expensive successes (cost = prefix-reuse FLOPs).

# Current skill
{skill}

# Contrastive notes
{notes}

Write {n} different full rewrites of the skill. For each, output exactly:
<<<RATIONALE>>>
Which steps of the runs above this targets, and the predicted effect on accuracy and on cost.
<<<END RATIONALE>>>
<<<SKILL>>>
<the full new SKILL.md>
<<<END SKILL>>>
"""


def parse_candidates(text: str, max_chars: int = 12000) -> list[Candidate]:
    """Extract well-formed candidates; malformed ones (no rationale, empty, too long) are dropped."""
    out = []
    skills = list(SKILL_RE.finditer(text))
    rats = list(RATIONALE_RE.finditer(text))
    for m in skills:
        rationale = ""
        for r in rats:
            if r.end() <= m.start():
                rationale = r.group(1).strip()
        body = m.group(1).strip()
        if body and rationale and len(body) <= max_chars:
            out.append(Candidate(body, rationale))
    return out


class LLMProposer:
    """Uses any ``clm.llm`` client (no tools) to propose rewrites."""

    def __init__(self, llm, max_tokens: int = 8000):
        self.llm = llm
        self.max_tokens = max_tokens

    def propose(self, skill: str, notes: str, n: int) -> list[Candidate]:
        prompt = PROPOSER_PROMPT.format(skill=skill or "(empty: no context-management instructions yet)",
                                        notes=notes or "(no notes)", n=n)
        resp = self.llm.complete([{"role": "user", "content": prompt}], tools=None, max_tokens=self.max_tokens)
        return parse_candidates(resp.content or "")


def contrastive_notes(results: dict[str, list[Episode]], max_pairs: int = 8) -> str:
    notes = []
    for tid, eps in results.items():
        succ = sorted([e for e in eps if e.success], key=lambda e: e.cost)
        fail = [e for e in eps if not e.success]
        if succ and fail:
            notes.append(f"## {tid}: success vs failure\nSUCCESS (cost {succ[0].cost:.3g}): {succ[0].notes}\n"
                         f"FAILURE (reward {fail[0].reward:.2f}): {fail[0].notes}")
        elif len(succ) >= 2 and succ[-1].cost > succ[0].cost:
            notes.append(f"## {tid}: cheap vs expensive success\nCHEAP (cost {succ[0].cost:.3g}): {succ[0].notes}\n"
                         f"EXPENSIVE (cost {succ[-1].cost:.3g}): {succ[-1].notes}")
        if len(notes) >= max_pairs:
            break
    return "\n\n".join(notes)


def paired_gate(cand: dict[str, float], base: dict[str, float], cand_cost: float, base_cost: float):
    """Accept if d > SE, or |d| <= SE and cheaper. Returns (accept, d, se)."""
    ids = sorted(set(cand) & set(base))
    diffs = [cand[i] - base[i] for i in ids]
    n = len(diffs)
    if n == 0:
        return False, 0.0, 0.0
    d = sum(diffs) / n
    se = math.sqrt(sum((x - d) ** 2 for x in diffs) / (n - 1) / n) if n > 1 else 0.0
    accept = d > se or (abs(d) <= se and cand_cost < base_cost)
    return accept, d, se


@dataclass
class EvalResult:
    per_task: dict[str, float]
    mean_reward: float
    mean_cost: float
    episodes: dict[str, list[Episode]] = field(default_factory=dict)


def evaluate(source: TaskSource, skill: str, task_ids: list[str], out_dir: Path, reps: int = 1) -> EvalResult:
    skill_dir = out_dir / "skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill, encoding="utf-8")
    eps: dict[str, list[Episode]] = {}
    for tid in task_ids:
        for rep in range(reps):
            eps.setdefault(tid, []).append(source.run(tid, skill_dir, out_dir / f"{tid}_r{rep}".replace(":", "_"), rep))
    per_task = {t: sum(e.reward for e in v) / len(v) for t, v in eps.items()}
    costs = [e.cost for v in eps.values() for e in v]
    return EvalResult(per_task, sum(per_task.values()) / max(len(per_task), 1),
                      sum(costs) / max(len(costs), 1), eps)


def evolve(source: TaskSource, proposer, init_skill: str = "", rounds: int = 10, n_candidates: int = 4,
           train_reps: int = 2, dev_reps: int = 1, patience: int = 5, out_dir: str | Path = "runs/evolve") -> dict:
    out_dir = Path(out_dir)
    current = init_skill
    cur_dev = evaluate(source, current, source.dev_ids, out_dir / "round0_dev", dev_reps)
    history = [{"round": 0, "accepted": True, "dev_reward": cur_dev.mean_reward, "dev_cost": cur_dev.mean_cost,
                "skill": current}]
    stale = 0
    for rnd in range(1, rounds + 1):
        train = evaluate(source, current, source.train_ids, out_dir / f"round{rnd}_train", train_reps)
        cands = proposer.propose(current, contrastive_notes(train.episodes), n_candidates)
        best = None
        for k, cand in enumerate(cands):
            dev = evaluate(source, cand.skill, source.dev_ids, out_dir / f"round{rnd}_cand{k}", dev_reps)
            ok, d, se = paired_gate(dev.per_task, cur_dev.per_task, dev.mean_cost, cur_dev.mean_cost)
            rec = {"round": rnd, "cand": k, "accepted": False, "d": d, "se": se, "dev_reward": dev.mean_reward,
                   "dev_cost": dev.mean_cost, "rationale": cand.rationale, "skill": cand.skill}
            history.append(rec)
            if ok and (best is None or (dev.mean_reward, -dev.mean_cost) > (best[1].mean_reward, -best[1].mean_cost)):
                best = (rec, dev, cand)
        if best:
            best[0]["accepted"] = True
            current, cur_dev = best[2].skill, best[1]
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    result = {"final_skill": current, "dev_reward": cur_dev.mean_reward, "dev_cost": cur_dev.mean_cost,
              "history": history}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evolution.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "SKILL.final.md").write_text(current, encoding="utf-8")
    return result


class ContextBenchSource(TaskSource):
    """Runs ContextBench tasks with the CLM harness; reward = accuracy, cost = prefix-reuse FLOPs."""

    def __init__(self, llm_factory, train_ids, dev_ids, test_ids=(), config: str = "contextbench",
                 flops_model: str = "qwen3.6-27b"):
        self.llm_factory = llm_factory
        self.train_ids, self.dev_ids, self.test_ids = list(train_ids), list(dev_ids), list(test_ids)
        self.config = config
        self.flops_model = flops_model

    def run(self, task_id, skill_dir, out_dir, rep):
        from ..config import CLMConfig
        from ..contextbench import make_task
        from ..flops import replay_run
        from ..harness import ContextEnv
        from ..sandbox import LocalSandbox

        name, level, seed = task_id.split(":")
        task = make_task(name, int(level), int(seed) * 1000 + rep)
        cfg = CLMConfig.load(self.config, skill_path=str(Path(skill_dir) / "SKILL.md"))
        sb = LocalSandbox()
        try:
            res = ContextEnv(cfg, self.llm_factory(name), task, sandbox=sb, run_dir=out_dir).run()
        finally:
            sb.cleanup()
        cost = replay_run(out_dir, self.flops_model)["prefix_reuse_flops"]
        notes = (f"status={res.status} accuracy={res.grade.get('accuracy')} turns={res.turns} "
                 f"peak_tokens={res.peak_tokens} stats={json.dumps(res.stats)}")
        return Episode(float(res.grade.get("accuracy") or 0.0), cost, notes)


def episode_dict(e: Episode) -> dict:
    return asdict(e)
