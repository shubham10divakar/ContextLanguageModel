"""Per-token and per-pair FLOP constants (paper App. C, Eqs. 7-8).

C_token = 6 L d d_ff                                            (MLP, gated: 3 matmuls)
        + L_attn [2d(2 hq dh + 2 hkv dh) + 2 hq dh d]           (full-attention projections, with output gate)
        + L_lin  [2d(2 hk dk + 2 hv dv + 2 hv) + 2 hv dv d]     (Gated DeltaNet projections)
C_attn  = 4 L_attn hq dh                                        (per query-key pair)

The query projection is counted twice (2 hq dh) because Qwen3.6 full-attention
layers have an output gate of the same width. For a plain dense model set
``attn_output_gate=False``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    L: int                 # total layers
    d: int                 # hidden size
    d_ff: int              # MLP intermediate size
    L_attn: int            # full-attention layers
    hq: int
    hkv: int
    dh: int
    L_lin: int = 0         # Gated DeltaNet (linear-attention) layers
    hk: int = 0
    hv: int = 0
    dk: int = 0
    dv: int = 0
    attn_output_gate: bool = True

    @property
    def c_token(self) -> float:
        mlp = 6 * self.L * self.d * self.d_ff
        q_width = (2 if self.attn_output_gate else 1) * self.hq * self.dh
        attn = self.L_attn * (2 * self.d * (q_width + 2 * self.hkv * self.dh) + 2 * self.hq * self.dh * self.d)
        lin = self.L_lin * (2 * self.d * (2 * self.hk * self.dk + 2 * self.hv * self.dv + 2 * self.hv)
                            + 2 * self.hv * self.dv * self.d)
        return float(mlp + attn + lin)

    @property
    def c_attn(self) -> float:
        return float(4 * self.L_attn * self.hq * self.dh)


SPECS: dict[str, ModelSpec] = {
    # Paper App. C: 64 layers, 16 full attention + 48 Gated DeltaNet.
    "qwen3.6-27b": ModelSpec("qwen3.6-27b", L=64, d=5120, d_ff=17408, L_attn=16, hq=24, hkv=4, dh=256,
                             L_lin=48, hk=16, hv=48, dk=128, dv=128),
    # Dense models (no output gate, all layers full attention); from their public configs.
    "qwen3-8b": ModelSpec("qwen3-8b", L=36, d=4096, d_ff=12288, L_attn=36, hq=32, hkv=8, dh=128,
                          attn_output_gate=False),
    "qwen3-4b": ModelSpec("qwen3-4b", L=36, d=2560, d_ff=9728, L_attn=36, hq=32, hkv=8, dh=128,
                          attn_output_gate=False),
    "qwen3-1.7b": ModelSpec("qwen3-1.7b", L=28, d=2048, d_ff=6144, L_attn=28, hq=16, hkv=8, dh=128,
                            attn_output_gate=False),
    "qwen3-0.6b": ModelSpec("qwen3-0.6b", L=28, d=1024, d_ff=3072, L_attn=28, hq=16, hkv=8, dh=128,
                            attn_output_gate=False),
    "llama-3.1-8b": ModelSpec("llama-3.1-8b", L=32, d=4096, d_ff=14336, L_attn=32, hq=32, hkv=8, dh=128,
                              attn_output_gate=False),
}


def from_hf_config(cfg: dict, name: str = "custom") -> ModelSpec:
    """Build a dense-model spec from a Hugging Face ``config.json`` dict."""
    hq = cfg["num_attention_heads"]
    return ModelSpec(name, L=cfg["num_hidden_layers"], d=cfg["hidden_size"], d_ff=cfg["intermediate_size"],
                     L_attn=cfg["num_hidden_layers"], hq=hq, hkv=cfg.get("num_key_value_heads", hq),
                     dh=cfg.get("head_dim") or cfg["hidden_size"] // hq, attn_output_gate=False)
