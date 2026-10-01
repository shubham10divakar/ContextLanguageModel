import random

from clm.icl.evolve import (ContextBenchSource, Episode, TaskSource, contrastive_notes, evolve, paired_gate,
                            parse_candidates)
from clm.llm import ScriptedLLM
from clm.contextbench.oracles import clm_oracle


def test_parse_candidates_drops_malformed():
    text = ("<<<RATIONALE>>>\ntargets step 3\n<<<END RATIONALE>>>\n<<<SKILL>>>\ngood skill\n<<<END SKILL>>>\n"
            "<<<SKILL>>>\n\n<<<END SKILL>>>\n"                      # empty
            "<<<SKILL>>>\nunclosed")
    c = parse_candidates(text)
    assert len(c) == 1 and c[0].skill == "good skill" and c[0].rationale == "targets step 3"
    assert parse_candidates("<<<SKILL>>>\nno rationale\n<<<END SKILL>>>") == []


def test_paired_gate():
    base = {"a": 0.5, "b": 0.5, "c": 0.5}
    assert paired_gate({"a": 0.9, "b": 0.8, "c": 1.0}, base, 1, 1)[0]          # clear gain
    assert paired_gate({"a": 0.5, "b": 0.5, "c": 0.5}, base, 0.5, 1.0)[0]      # tie, cheaper
    assert not paired_gate({"a": 0.5, "b": 0.5, "c": 0.5}, base, 2.0, 1.0)[0]  # tie, pricier
    assert not paired_gate({"a": 0.1, "b": 0.2, "c": 0.0}, base, 0.1, 1.0)[0]  # worse


def test_contrastive_notes_pairs():
    n = contrastive_notes({"t1": [Episode(1, 5, "ok"), Episode(0, 3, "bad")],
                           "t2": [Episode(1, 2, "cheap"), Episode(1, 9, "pricey")]})
    assert "success vs failure" in n and "cheap vs expensive" in n


class ToySource(TaskSource):
    """Reward rises with the number of distinct good tips in the skill."""
    train_ids = ["t0", "t1", "t2"]
    dev_ids = ["d0", "d1", "d2", "d3"]
    TIPS = ["offload", "batch edits", "summarize"]

    def run(self, task_id, skill_dir, out_dir, rep):
        skill = (skill_dir / "SKILL.md").read_text()
        k = sum(t in skill for t in self.TIPS)
        rng = random.Random(hash((task_id, rep, k)) % 1000)
        return Episode(reward=min(1.0, 0.25 + 0.25 * k + rng.uniform(-0.02, 0.02)), cost=10 - k, notes=skill[:50])


class ToyProposer:
    def __init__(self):
        self.i = 0

    def propose(self, skill, notes, n):
        tip = ToySource.TIPS[min(self.i, 2)]
        self.i += 1
        return [type("C", (), {"skill": skill + "\n- " + tip, "rationale": "adds " + tip})(),
                type("C", (), {"skill": "unrelated text", "rationale": "noise"})()]


def test_evolve_improves_and_stops(tmp_path):
    r = evolve(ToySource(), ToyProposer(), init_skill="", rounds=6, n_candidates=2, patience=2,
               out_dir=tmp_path)
    assert r["dev_reward"] > 0.9
    assert all(t in r["final_skill"] for t in ToySource.TIPS)
    assert (tmp_path / "SKILL.final.md").exists()


def test_contextbench_source_runs_harness(tmp_path):
    src = ContextBenchSource(lambda name: ScriptedLLM(clm_oracle(name)), ["kv_store:130:0"], ["kv_store:130:1"])
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("offload batches to disk")
    ep = src.run("kv_store:130:0", skill_dir, tmp_path / "run", 0)
    assert ep.reward == 1.0 and ep.cost > 0
