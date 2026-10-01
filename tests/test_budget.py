from clm.budget import BudgetController, truncate_middle
from clm.config import CLMConfig
from clm.edit_gate import check_edit
from clm.tokens import TokenCounter


def _ctl(**kw):
    cfg = CLMConfig(context_budget_tokens=1000, reserve_tokens=100, **kw)
    return BudgetController(cfg, TokenCounter())


def test_escalating_nudges_fire_once_and_rearm():
    b = _ctl(urgent_mode="none")
    assert b.limit == 900
    assert b.nudges(100) == []
    assert len(b.nudges(250)) == 1          # 25% crosses
    assert b.nudges(260) == []               # already fired
    assert "75%" in b.nudges(700)[0]         # jumps to highest crossed
    assert b.nudges(710) == []
    b.nudges(100)                            # shrink re-arms
    assert len(b.nudges(500)) == 1


def test_urgent_ratio_fires_every_turn():
    b = _ctl(urgent_mode="ratio", persistent_nudge_ratio=0.9)
    assert "URGENT" in b.nudges(820)[0]
    assert "URGENT" in b.nudges(830)[0]


def test_urgent_adaptive_uses_recent_outputs():
    b = _ctl(urgent_mode="adaptive")
    b.record_observation(300)
    # free = 900-400 = 500 < 2*300 -> urgent
    assert any("URGENT" in n for n in b.nudges(400))
    b2 = _ctl(urgent_mode="adaptive")
    b2.record_observation(10)
    assert not any("URGENT" in n for n in b2.nudges(400))


def test_rollback_drops_newest_keeps_env_and_pairs():
    b = _ctl()
    protected = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    big = "word " * 500
    hist = [
        {"role": "user", "content": "op1", "env": True},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "bash", "arguments": '{"command": "cat a"}'}]},
        {"role": "tool", "tool_call_id": "a", "content": big},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "name": "bash", "arguments": '{"command": "cat b"}'}]},
        {"role": "tool", "tool_call_id": "b", "content": big},
        {"role": "user", "content": "op2", "env": True},
    ]
    new, dropped = b.rollback(protected, hist)
    assert b.counter.count(protected + new) <= 800
    assert [m["content"] for m in new if m.get("env")] == ["op1", "op2"]
    assert "cat b" in dropped
    # no dangling tool calls
    ids = {m.get("tool_call_id") for m in new if m["role"] == "tool"}
    for m in new:
        for c in m.get("tool_calls") or []:
            assert c["id"] in ids


def test_truncate_middle():
    s = "a" * 50 + "b" * 50
    out = truncate_middle(s, 20)
    assert out.startswith("a" * 10) and out.endswith("b" * 10) and "truncated" in out
    assert truncate_middle("short", 20) == "short"


def test_edit_gate_modes():
    assert check_edit("fit", 500, 800, 900).accepted
    assert not check_edit("fit", 500, 950, 900).accepted
    assert not check_edit("shrink", 500, 600, 900).accepted
    assert check_edit("shrink", 500, 400, 900).accepted


def test_config_load_named():
    cfg = CLMConfig.load("bcp")
    assert cfg.edit_gate == "shrink" and cfg.limit == 28672 - 2048
