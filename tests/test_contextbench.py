import pytest

from clm.config import CLMConfig
from clm.contextbench import make_task
from clm.contextbench.common import answer_blocks
from clm.contextbench.oracles import clm_oracle, keep_all_oracle
from clm.contextbench.sudoku import parse_last_board, render_board, solved_board
from clm.harness import ContextEnv
from clm.llm import ScriptedLLM
from clm.sandbox import LocalSandbox

SMALL = {"needle_retention": 4, "sudoku_sketchpad": 4, "kv_store": 130, "log_triage": 160}


def _run(task, policy, tmp_path, strategy="clm"):
    cfg = CLMConfig.load("contextbench", strategy=strategy)
    env = ContextEnv(cfg, ScriptedLLM(policy), task, sandbox=LocalSandbox(tmp_path / "sb"))
    return env.run()


@pytest.mark.parametrize("name", list(SMALL))
def test_clm_oracle_scores_perfectly(name, tmp_path):
    task = make_task(name, SMALL[name], seed=1)
    r = _run(task, clm_oracle(name), tmp_path)
    assert r.status == "task_done", r
    assert r.grade["accuracy"] == 1.0, r.grade
    assert r.stats.get("edits_applied", 0) >= len(task.ops)


@pytest.mark.parametrize("name", list(SMALL))
def test_keep_all_oracle_fine_at_low_pressure(name, tmp_path):
    task = make_task(name, SMALL[name], seed=2)
    r = _run(task, keep_all_oracle(name), tmp_path, strategy="base")
    assert r.status == "task_done" and r.grade["accuracy"] == 1.0, r.grade


def test_high_pressure_separates_base_from_clm(tmp_path):
    level = 1000  # ~1.4x context pressure
    base = _run(make_task("kv_store", level, seed=3), keep_all_oracle("kv_store"), tmp_path / "a", "base")
    clm = _run(make_task("kv_store", level, seed=3), clm_oracle("kv_store"), tmp_path / "b", "clm")
    assert base.status == "context_overflow" and base.grade["accuracy"] == 0.0
    assert clm.grade["accuracy"] == 1.0 and clm.peak_tokens < 8000


def test_sudoku_board_is_valid_and_roundtrips():
    import random

    b = solved_board(random.Random(0))
    for row in b:
        assert len(set(row)) == 16
    for c in range(16):
        assert len({b[r][c] for r in range(16)}) == 16
    version, cells = parse_last_board("junk\n" + render_board(b, 7))
    assert version == 7 and cells[(1, 1)] == b[0][0] and len(cells) == 256


def test_answer_blocks_take_last():
    text = "<<<ANSWER key=K1>>>\nold\n<<<ANSWER END>>>\n...\n<<<ANSWER key=K1>>>\nnew value\n<<<ANSWER END>>>"
    assert answer_blocks(text, "key") == {"K1": "new value"}


def test_generators_are_deterministic():
    a, b = make_task("log_triage", 160, seed=5), make_task("log_triage", 160, seed=5)
    assert [o.body for o in a.ops] == [o.body for o in b.ops]
    assert len([o for o in a.ops if o.kind == "query"]) == 24
