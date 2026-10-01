import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from clm.scr.diff import plan_reuse  # noqa: E402
from clm.scr.reference import full_prefill, scr_divergence, scr_prefill  # noqa: E402
from clm.scr.rope import apply_rope, rope_inv_freq, rotate_keys  # noqa: E402


def test_delta_rotation_equals_direct_rope():
    torch.manual_seed(0)
    inv = rope_inv_freq(16)
    k = torch.randn(2, 10, 16, dtype=torch.float64)
    old = torch.arange(100, 110)
    new = torch.arange(37, 47)
    cached = apply_rope(k, old, inv)
    assert torch.allclose(rotate_keys(cached, old, new, inv), apply_rope(k, new, inv), atol=1e-10)


def test_partial_rotary_leaves_tail_untouched():
    inv = rope_inv_freq(8)
    k = torch.randn(3, 16, dtype=torch.float64)
    out = rotate_keys(k, torch.arange(3), torch.arange(5, 8), inv)
    assert torch.equal(out[..., 8:], k[..., 8:])


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return transformers.Qwen2ForCausalLM(cfg).eval().to(torch.float64)


def test_append_only_is_exact(model):
    old = list(range(1, 60))
    new = old + [7, 8, 9]
    _, cache = full_prefill(model, old)
    approx, _ = scr_prefill(model, cache, new, plan_reuse(old, new))
    exact, _ = full_prefill(model, new)
    assert torch.allclose(approx, exact, atol=1e-8)


def test_relocated_first_layer_keys_are_exact(model):
    # Layer-0 K/V depend only on token and position, so relocation must reproduce them exactly.
    old = [i % 100 + 1 for i in range(200)]
    new = old[:50] + [120, 121] + old[80:]        # replace 30 tokens by 2
    plan = plan_reuse(old, new, k=6, min_extend=16)
    assert plan.relocated and plan.relocated[0].shift == -28
    _, old_cache = full_prefill(model, old)
    _, scr_cache = scr_prefill(model, old_cache, new, plan)
    _, exact_cache = full_prefill(model, new)
    # HF computes RoPE cos/sin in float32, so keys agree to ~1e-6 (a wrong rotation gives O(1) errors)
    assert (scr_cache.layers[0].keys - exact_cache.layers[0].keys).abs().max() < 1e-5
    assert torch.allclose(scr_cache.layers[0].values, exact_cache.layers[0].values, atol=1e-12)
    # deeper layers differ (they saw the pre-edit context): SCR is an approximation
    assert (scr_cache.layers[2].keys - exact_cache.layers[2].keys).abs().max() > 1e-4


def test_divergence_report(model):
    old = [i % 100 + 1 for i in range(200)]
    new = old[:50] + [120, 121] + old[80:]
    rep = scr_divergence(model, old, new)
    assert 0.8 < rep.reused_fraction < 1.0
    assert rep.kl >= 0
