"""Advantages for CLM reinforcement learning (paper Sec. 4.2, Eq. 6).

Outcome channel: standard GRPO group-normalized reward.

Efficiency channel (success-gated): within a rollout group g, over the
successful trajectories G+ only,

    A_eff_i = clip((c_bar - c_i) / c_bar, -1, 1)   if i in G+,   else 0,

with c_bar the mean prefix-reuse FLOPs of G+; A_eff = 0 for everyone when
|G+| < 2.

Two ways to combine them:

* ``paper``:  A_i = A_out_i + w_eff * A_eff_i on every token of trajectory i (Eq. after 6).
* ``dual_channel`` (what the released patch does):
      return_i[t] = A_out_i + w_eff * A_eff_i * role_mask_i[t]
  where role_mask is 1 only on tokens of turns whose single action is a
  context edit, so efficiency credit lands on the edit tokens.

The reference patch defaults ``w_eff`` to 1.0 when the env var is missing; the
paper uses 0.25 (``POLAR_DUAL_CHANNEL_W_EFF=0.25``).
"""

from __future__ import annotations

import numpy as np

NO_TOOL_CALL_PENALTY = 0.3
FORMAT_FAILURE_PENALTY = 0.2


def shaped_reward(success: bool, no_tool_call: bool = False, format_failure: bool = False) -> float:
    """Binary task reward; failed trajectories pay for missing tool calls / malformed output."""
    if success:
        return 1.0
    r = 0.0
    if no_tool_call:
        r -= NO_TOOL_CALL_PENALTY
    if format_failure:
        r -= FORMAT_FAILURE_PENALTY
    return r


def group_normalize(rewards, eps: float = 1e-6) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    std = r.std()
    if std < eps:
        return np.zeros_like(r)
    return (r - r.mean()) / (std + eps)


def efficiency_advantage(costs, successes) -> np.ndarray:
    c = np.asarray(costs, dtype=np.float64)
    s = np.asarray(successes, dtype=bool)
    out = np.zeros_like(c)
    if s.sum() < 2:
        return out
    c_bar = c[s].mean()
    if c_bar <= 0:
        return out
    out[s] = np.clip((c_bar - c[s]) / c_bar, -1.0, 1.0)
    return out


def paper_advantage(rewards, successes, costs, w_eff: float = 0.25) -> np.ndarray:
    return group_normalize(rewards) + w_eff * efficiency_advantage(costs, successes)


def dual_channel_returns(outcome_adv: float, a_eff: float, role_mask, w_eff: float = 0.25) -> np.ndarray:
    m = np.asarray(role_mask, dtype=np.float64)
    return outcome_adv + w_eff * a_eff * m


def dynamic_sampling_keep(rewards, eps: float = 1e-9) -> bool:
    """DAPO dynamic sampling: drop groups with no reward variation (zero outcome signal)."""
    r = np.asarray(rewards, dtype=np.float64)
    return bool(r.max() - r.min() > eps)
