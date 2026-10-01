import numpy as np
import pytest

from clm.config import CLMConfig
from clm.contextbench import make_task
from clm.contextbench.oracles import clm_oracle
from clm.harness import ContextEnv
from clm.llm import ScriptedLLM
from clm.rl.advantage import (dual_channel_returns, dynamic_sampling_keep, efficiency_advantage, group_normalize,
                              paper_advantage, shaped_reward)
from clm.rl.segments import Trajectory, assign_advantages, load_trajectory
from clm.sandbox import LocalSandbox


def test_efficiency_advantage_eq6():
    costs = [1.0, 2.0, 3.0, 100.0]
    succ = [True, True, True, False]
    a = efficiency_advantage(costs, succ)
    # c_bar = 2 over successes
    assert a.tolist() == pytest.approx([0.5, 0.0, -0.5, 0.0])
    assert efficiency_advantage([1, 2], [True, False]).tolist() == [0, 0]   # < 2 successes
    assert efficiency_advantage([1, 1, 100], [True] * 3)[2] == -1.0         # clipped


def test_paper_vs_dual_channel():
    r, s, c = [1, 1, 0, 0], [True, True, False, False], [1.0, 3.0, 5.0, 5.0]
    a = paper_advantage(r, s, c, w_eff=0.25)
    out = group_normalize(r)
    assert a[0] == pytest.approx(out[0] + 0.25 * 0.5)
    ret = dual_channel_returns(out[0], 0.5, [0, 1, 1, 0], w_eff=0.25)
    assert ret.tolist() == pytest.approx([out[0], out[0] + 0.125, out[0] + 0.125, out[0]])


def test_shaped_reward_and_dynamic_sampling():
    assert shaped_reward(True, True, True) == 1.0
    assert shaped_reward(False, True, True) == pytest.approx(-0.5)
    assert not dynamic_sampling_keep([1, 1, 1])
    assert dynamic_sampling_keep([1, 0, 1])
    assert np.all(group_normalize([3, 3]) == 0)


def test_segments_from_real_run(tmp_path):
    run = tmp_path / "run"
    env = ContextEnv(CLMConfig.load("contextbench"), ScriptedLLM(clm_oracle("needle_retention")),
                     make_task("needle_retention", 4, seed=0), sandbox=LocalSandbox(tmp_path / "sb"), run_dir=run)
    env.run()
    traj = load_trajectory(run, edit_flag="edited")
    assert traj.success and traj.reward == 1.0 and traj.cost > 0
    assert len(traj.segments) == 4 and all(s.is_edit_turn for s in traj.segments)
    other = Trajectory("b", [], 0.0, False, 1.0)
    segs = assign_advantages([traj, other], mode="dual_channel")
    assert all(s.advantage == pytest.approx(1.0, abs=1e-3) for s in segs)


def test_grpo_step_on_tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from clm.rl.grpo import GRPOConfig, grpo_step, grpo_token_loss

    # loss sanity: positive advantage, ratio 1 -> loss = -A
    z = torch.zeros(1, 3)
    loss, info = grpo_token_loss(z, z, torch.ones(1, 3), torch.ones(1, 3))
    assert loss.item() == pytest.approx(-1.0) and info["clip_frac"] == 0

    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    model = transformers.Qwen2ForCausalLM(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    good, bad = [5, 6, 7], [9, 9, 9]
    batch = [([1, 2, 3], good, 1.0), ([1, 2, 3], bad, -1.0)]
    from clm.rl.grpo import collate, response_logprobs

    def margin():
        ids, attn, resp, _ = collate(batch)
        with torch.no_grad():
            lp = response_logprobs(model, ids, attn)
        return ((lp[0] * resp[0]).sum() - (lp[1] * resp[1]).sum()).item()

    before = margin()
    for _ in range(5):
        grpo_step(model, opt, batch, cfg=GRPOConfig(kl_coef=0.0))
    assert margin() > before
