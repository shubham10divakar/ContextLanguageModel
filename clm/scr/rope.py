"""RoPE delta re-rotation for relocated keys (the reference ``_rotate_keys``).

Cached keys are stored *after* RoPE, i.e. k_p = R(p) k. RoPE rotations of
each 2-D frequency pair compose, R(p') = R(p' - p) R(p), so moving a cached
key from position p to p' only needs a rotation by the difference
delta = p' - p (equivalently, cos/sin angle-difference formulas applied to
the stored rotation). Values carry no positional encoding and move as-is.

Convention: Hugging Face ``rotate_half`` layout, with the frequency vector
duplicated over the two halves of the rotary dimension.
"""

from __future__ import annotations

import torch


def rope_inv_freq(rotary_dim: int, base: float = 10000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))


def rope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape [n, rotary_dim] for (possibly negative) positions."""
    ang = positions.to(torch.float64)[:, None] * inv_freq.to(torch.float64)[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x[..., n, dh] at ``positions`` (partial rotary if inv_freq covers < dh)."""
    rd = 2 * inv_freq.shape[0]
    cos, sin = rope_cos_sin(positions, inv_freq)
    xr = x[..., :rd].to(torch.float64)
    out = xr * cos + rotate_half(xr) * sin
    return torch.cat([out.to(x.dtype), x[..., rd:]], dim=-1)


def rotate_keys(k: torch.Tensor, old_pos: torch.Tensor, new_pos: torch.Tensor,
                inv_freq: torch.Tensor) -> torch.Tensor:
    """Re-rotate post-RoPE keys k[..., n, dh] from ``old_pos`` to ``new_pos``."""
    return apply_rope(k, new_pos.to(torch.long) - old_pos.to(torch.long), inv_freq)
