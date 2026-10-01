"""Reference Suffix Cache Reuse on a Hugging Face decoder (full-attention layers).

Given the previous prompt's KV cache and a new (edited) prompt, build the new
cache left to right:

  * the shared prefix is copied;
  * each relocated span (from ``plan_reuse``) is spliced in from the old cache,
    keys re-rotated to their new positions, values copied;
  * every other position (the edit B', span tails, new suffix) is prefilled
    by running the model on those tokens over the cache built so far.

This is the semantics of Fig. 4/10 (left). It is an approximation: relocated
states were computed under the pre-edit context. ``scr_divergence`` measures
how far the next-token distribution moves versus an exact re-prefill, which
makes this a convenient testbed for studying when relocation is safe.

Scope: single sequence, standard (non-hybrid) RoPE models such as Qwen2/3 and
Llama. The linear-attention ``fork`` mode of the paper (restore the recurrent
state saved before the edit) is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .diff import ReusePlan, plan_reuse
from .rope import rotate_keys


def _inv_freq(model) -> torch.Tensor:
    rot = model.model.rotary_emb
    if getattr(rot, "attention_scaling", 1.0) not in (1.0, None):
        raise NotImplementedError("scaled RoPE variants are not supported by this reference")
    return rot.inv_freq.detach().cpu()


@torch.no_grad()
def full_prefill(model, ids: list[int]):
    out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=True)
    return out.logits[0, -1], out.past_key_values


@torch.no_grad()
def scr_prefill(model, old_cache, new_ids: list[int], plan: ReusePlan):
    """Build the cache for ``new_ids`` reusing ``old_cache`` according to ``plan``."""
    from transformers import DynamicCache

    dev = model.device
    inv_freq = _inv_freq(model)
    n_layers = len(old_cache.layers)
    cache = DynamicCache(config=model.config)
    reloc = {s.new_start: s for s in plan.relocated}
    logits = None

    def splice(old_lo: int, old_hi: int, new_lo: int):
        n = old_hi - old_lo
        old_pos = torch.arange(old_lo, old_hi)
        new_pos = torch.arange(new_lo, new_lo + n)
        for li in range(n_layers):
            k = old_cache.layers[li].keys[:, :, old_lo:old_hi]
            v = old_cache.layers[li].values[:, :, old_lo:old_hi]
            if old_lo != new_lo:
                k = rotate_keys(k.cpu(), old_pos, new_pos, inv_freq).to(dev, k.dtype)
            cache.update(k, v, li)

    def prefill(lo: int, hi: int):
        nonlocal logits
        out = model(input_ids=torch.tensor([new_ids[lo:hi]], device=dev), past_key_values=cache,
                    position_ids=torch.arange(lo, hi, device=dev)[None], use_cache=True)
        logits = out.logits[0, -1]

    if plan.prefix:
        splice(0, plan.prefix, 0)
    i, n = plan.prefix, len(new_ids)
    while i < n:
        if i in reloc:
            s = reloc[i]
            splice(s.old_start, s.old_start + s.length, s.new_start)
            i += s.length
            continue
        j = i
        while j < n and j not in reloc:
            j += 1
        prefill(i, j)
        i = j
    if logits is None:   # whole prompt reused: recompute the last token for its logits
        cache.crop(n - 1)
        prefill(n - 1, n)
    return logits, cache


@dataclass
class SCRReport:
    plan: ReusePlan
    reused_fraction: float
    kl: float              # KL(exact || scr) of the next-token distribution
    top1_agree: bool


@torch.no_grad()
def scr_divergence(model, old_ids: list[int], new_ids: list[int], k: int = 6, min_extend: int = 16) -> SCRReport:
    _, old_cache = full_prefill(model, old_ids)
    exact, _ = full_prefill(model, new_ids)
    plan = plan_reuse(old_ids, new_ids, k=k, min_extend=min_extend)
    approx, _ = scr_prefill(model, old_cache, new_ids, plan)
    p, q = F.log_softmax(exact.float(), -1), F.log_softmax(approx.float(), -1)
    kl = torch.sum(p.exp() * (p - q)).item()
    return SCRReport(plan, plan.reused_tokens / len(new_ids), kl, bool(exact.argmax() == approx.argmax()))
