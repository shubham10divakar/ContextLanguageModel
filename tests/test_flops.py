import pytest

from clm.flops import SPECS, PrefixCacheSim, replay, turn_flops
from clm.scr.diff import plan_reuse

Q = SPECS["qwen3.6-27b"]


def test_qwen36_constants_match_paper():
    # App. C: C_token = 48.70e9 (MLP 34.23e9, attn proj 3.36e9, GDN 11.12e9), C_attn = 3.93e5
    assert Q.c_token == pytest.approx(48.70e9, rel=1e-3)
    assert Q.c_attn == pytest.approx(3.93e5, rel=1e-3)


@pytest.mark.parametrize("R,expected", [(18000, 1.41e14), (10000, 5.74e14), (0, 10.81e14)])
def test_fig14_turn_costs(R, expected):
    assert turn_flops(Q, P=20000, R=R, G=500) == pytest.approx(expected, rel=5e-3)


def test_prefix_cache_sim_block_granularity():
    c = PrefixCacheSim(block_size=4)
    a = list(range(10))
    c.insert(a)
    assert c.match(a) == 8                 # only full blocks
    assert c.match(a + [99, 98]) == 8
    assert c.match([0, 1, 2, 3, 7, 7, 7, 7]) == 4
    assert PrefixCacheSim(1).match(a) == 0


def _snap(turn, contents, G=10):
    msgs = [{"role": "system", "content": "sys"}] + [{"role": "user", "content": c} for c in contents]
    return {"turn": turn, "messages": msgs, "response": {"completion_tokens": G}}


def _toks(messages):
    return [ord(ch) for m in messages for ch in (m["role"] + ":" + m["content"] + "|")]


def test_replay_append_vs_edit():
    base = ["alpha " * 50, "beta " * 50, "gamma " * 50]
    append = [_snap(1, base[:2]), _snap(2, base)]
    edit = [_snap(1, base[:2]), _snap(2, ["EDITED"] + base[1:])]
    ra = replay(append, Q, tokenize=_toks, block_size=1)
    re_ = replay(edit, Q, tokenize=_toks, block_size=1)
    assert ra["per_turn"][1]["R"] == ra["per_turn"][0]["P"]
    assert re_["per_turn"][1]["R"] < 20
    assert re_["prefix_reuse_flops"] > ra["prefix_reuse_flops"]
    # SCR recovers most of the edited case: the unchanged suffix is relocated
    assert re_["per_turn"][1]["scr_reused"] > 200
    assert re_["scr_flops"] < re_["prefix_reuse_flops"]


def test_server_counts_take_priority():
    s = _snap(1, ["x" * 100])
    s["response"].update(prompt_tokens=5000, cached_tokens=4000)
    r = replay([s], Q, tokenize=_toks)
    t = r["per_turn"][0]
    assert (t["P"], t["R"], t["source"]) == (5000, 4000, "server")


def test_plan_reuse_relocates_k_longest_minus_tail():
    old = list(range(1000))
    new = list(range(100)) + [-1] * 5 + list(range(150, 1000))   # replace [100,150) by 5 tokens
    plan = plan_reuse(old, new, k=6, min_extend=16)
    assert plan.prefix == 100
    assert len(plan.relocated) == 1
    s = plan.relocated[0]
    assert (s.old_start, s.new_start, s.length) == (150, 105, 850 - 16)
    assert sum(plan.prefill_mask()) == 5 + 16
    assert plan_reuse(old, old + [5, 6]).relocated == []   # append-only
