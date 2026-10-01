"""Minimal stepwise GRPO objective in PyTorch.

Paper recipe (App. E): GRPO with token-mean loss, truncated importance
sampling, low-variance KL (k3) with coefficient 0.01, DAPO dynamic sampling,
clip range [0.2, 0.28], Adam lr 1e-6, weight decay 0.1, 8 prompts x 32
rollouts per step.

This module gives the loss and a single optimisation step over tokenized
segments; any rollout engine (the harness above + vLLM/SGLang) can feed it.
For multi-GPU training use a full RL stack (verl, slime, TRL) and port
``clm.rl.advantage`` into its advantage hook.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    clip_low: float = 0.2
    clip_high: float = 0.28
    kl_coef: float = 0.01
    tis_cap: float | None = 2.0     # truncated importance sampling cap (rollout vs trainer logprobs)


def grpo_token_loss(logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
                    ref_logp: torch.Tensor | None = None, rollout_logp: torch.Tensor | None = None,
                    cfg: GRPOConfig = GRPOConfig()) -> tuple[torch.Tensor, dict]:
    """Clipped policy-gradient loss, token-mean over every response token in the batch.

    All tensors are [batch, seq]; ``adv`` is per token (broadcast a per-segment value).
    """
    mask = mask.to(logp.dtype)
    ratio = torch.exp(logp - old_logp)
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_low, 1 + cfg.clip_high)
    pg = torch.maximum(pg1, pg2)
    if rollout_logp is not None and cfg.tis_cap is not None:
        w = torch.exp(old_logp - rollout_logp).clamp(max=cfg.tis_cap).detach()
        pg = pg * w
    loss_tok = pg
    kl = torch.zeros_like(logp)
    if ref_logp is not None and cfg.kl_coef > 0:
        d = ref_logp - logp
        kl = torch.exp(d) - d - 1                       # k3 estimator, >= 0
        loss_tok = loss_tok + cfg.kl_coef * kl
    denom = mask.sum().clamp(min=1)
    loss = (loss_tok * mask).sum() / denom
    with torch.no_grad():
        clipped = ((ratio < 1 - cfg.clip_low) | (ratio > 1 + cfg.clip_high)).to(logp.dtype)
        info = {"loss": loss.item(), "kl": ((kl * mask).sum() / denom).item(),
                "clip_frac": ((clipped * mask).sum() / denom).item()}
    return loss, info


def response_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-token log p(x_t | x_<t), aligned to ``input_ids`` (position 0 gets 0)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    lp = torch.gather(F.log_softmax(logits, -1), 2, input_ids[:, 1:, None]).squeeze(-1)
    return F.pad(lp, (1, 0))


def collate(batch: list[tuple[list[int], list[int], float]], pad_id: int = 0):
    """(prompt_ids, response_ids, advantage) -> padded ids, attention mask, response mask, advantages."""
    L = max(len(p) + len(r) for p, r, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    resp = torch.zeros((len(batch), L), dtype=torch.float32)
    adv = torch.zeros((len(batch), L), dtype=torch.float32)
    for i, (p, r, a) in enumerate(batch):
        seq = p + r
        ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
        resp[i, len(p):len(seq)] = 1
        adv[i, :] = a
    return ids, attn, resp, adv


def grpo_step(model, optimizer, batch, ref_model=None, cfg: GRPOConfig = GRPOConfig(), pad_id: int = 0,
              old_logp: torch.Tensor | None = None) -> dict:
    """One on-policy optimisation step over ``[(prompt_ids, response_ids, advantage), ...]``."""
    device = next(model.parameters()).device
    ids, attn, resp, adv = (t.to(device) for t in collate(batch, pad_id))
    if old_logp is None:
        with torch.no_grad():
            old_logp = response_logprobs(model, ids, attn)
    ref = None
    if ref_model is not None:
        with torch.no_grad():
            ref = response_logprobs(ref_model, ids, attn)
    logp = response_logprobs(model, ids, attn)
    loss, info = grpo_token_loss(logp, old_logp, adv, resp, ref_logp=ref, cfg=cfg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return info
