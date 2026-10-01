"""Prefix-reuse FLOPs (paper Eq. 3 and App. C, Eq. 9) and trace replay.

Per turn t with prompt length P_t, reused prefix R_t, prefill U_t = P_t - R_t
and G_t generated tokens::

    F_t = C_token (U_t + G_t) + C_attn [ 1/2 (P_t^2 - R_t^2) + G_t P_t + 1/2 G_t^2 ]

The replay walks the context snapshots saved before every LLM call through a
simulated prefix cache. Sources for (P_t, R_t), in priority order:
  1. the server's own ``prompt_tokens`` / ``cached_tokens`` report;
  2. a local tokenization of the exact prompt, matched against all earlier
     prompts at KV-block granularity (``block_size``; 1 = exact tokens).
Decoded tokens are not treated as cached: a response is prefilled again when
it first appears in a later prompt.

The same replay estimates Suffix Cache Reuse: tokens relocated from the
previous prompt are subtracted from the prefill and attention is charged
only for the positions actually prefilled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ..scr.diff import plan_reuse
from ..trace import load_snapshots
from .model_specs import SPECS, ModelSpec


def turn_flops(spec: ModelSpec, P: int, R: int, G: int) -> float:
    U = P - R
    return spec.c_token * (U + G) + spec.c_attn * (0.5 * (P * P - R * R) + G * P + 0.5 * G * G)


def turn_flops_from_mask(spec: ModelSpec, prefill_mask: list[bool], G: int) -> float:
    """Eq. 9 generalized to non-contiguous prefill (used for SCR)."""
    P = len(prefill_mask)
    positions = [i for i, m in enumerate(prefill_mask) if m]
    U = len(positions)
    pairs = sum(positions) + 0.5 * U          # equals (P^2 - R^2)/2 for a contiguous suffix
    return spec.c_token * (U + G) + spec.c_attn * (pairs + G * P + 0.5 * G * G)


def chatml(messages: list[dict]) -> str:
    """A model-agnostic chat serialization for local token counting."""
    out = []
    for m in messages:
        body = m.get("content") or ""
        for c in m.get("tool_calls") or []:
            fn = c.get("function", c)
            body += f"\n<tool_call>{fn.get('name')} {fn.get('arguments')}</tool_call>"
        out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
    return "".join(out)


def tiktoken_tokenizer(name: str = "o200k_base") -> Callable[[list[dict]], list[int]]:
    import tiktoken

    enc = tiktoken.get_encoding(name)
    return lambda messages: enc.encode(chatml(messages), disallowed_special=())


def hf_tokenizer(model_name: str) -> Callable[[list[dict]], list[int]]:
    """Use the served model's own chat template (most faithful local option)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)

    def encode(messages):
        msgs = []
        for m in messages:
            mm = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = [{"type": "function", "function": {
                    "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                    for c in m["tool_calls"]]
            if m.get("tool_call_id"):
                mm["tool_call_id"] = m["tool_call_id"]
            msgs.append(mm)
        return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

    return encode


class PrefixCacheSim:
    """Block-granular prefix cache: a prefix is cached if its chained block hashes were seen."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.seen: set[bytes] = set()

    def _chain(self, tokens: list[int]):
        h = b""
        bs = self.block_size
        for i in range(0, len(tokens) - len(tokens) % bs, bs):
            h = hashlib.blake2b(h + repr(tokens[i:i + bs]).encode(), digest_size=16).digest()
            yield h

    def match(self, tokens: list[int]) -> int:
        n = 0
        for h in self._chain(tokens):
            if h not in self.seen:
                break
            n += self.block_size
        return n

    def insert(self, tokens: list[int]) -> None:
        self.seen.update(self._chain(tokens))


@dataclass
class TurnCost:
    turn: int
    kind: str
    P: int
    R: int
    G: int
    flops: float
    source: str
    scr_reused: int = 0
    scr_flops: float = 0.0


def _gen_tokens(resp: dict, count_text: Callable[[str], int]) -> int:
    if resp.get("completion_tokens"):
        return int(resp["completion_tokens"])
    text = (resp.get("reasoning") or "") + (resp.get("content") or "")
    text += "".join(c.get("arguments", "") for c in resp.get("tool_calls") or [])
    return count_text(text)


def replay(snapshots: list[dict], spec: ModelSpec, tokenize=None, block_size: int = 16,
           use_server: bool = True, scr_k: int = 6, scr_min_extend: int = 16) -> dict:
    tokenize = tokenize or tiktoken_tokenizer()
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    count_text = lambda s: len(enc.encode(s, disallowed_special=()))  # noqa: E731
    cache = PrefixCacheSim(block_size)
    prev: list[int] | None = None
    turns: list[TurnCost] = []
    for snap in snapshots:
        tokens = tokenize(snap["messages"])
        resp = snap.get("response", {})
        G = _gen_tokens(resp, count_text)
        P, R, source = len(tokens), cache.match(tokens), "local"
        if use_server and resp.get("prompt_tokens") and resp.get("cached_tokens") is not None:
            P, R, source = int(resp["prompt_tokens"]), int(resp["cached_tokens"]), "server"
        R = min(R, P)
        cost = TurnCost(snap["turn"], resp.get("kind", "turn"), P, R, G, turn_flops(spec, P, R, G), source)
        # SCR estimate on local tokens against the session's previous prompt
        local_R = min(cache.match(tokens), len(tokens))
        if prev is not None:
            plan = plan_reuse(prev, tokens, k=scr_k, min_extend=scr_min_extend,
                              prefix=max(local_R, _lcp(prev, tokens)))
        else:
            plan = plan_reuse([], tokens, k=0, prefix=local_R)
        cost.scr_reused = plan.reused_tokens
        cost.scr_flops = turn_flops_from_mask(spec, plan.prefill_mask(), G)
        if source == "server":   # rescale the SCR estimate to the server's accounting
            cost.scr_flops = min(cost.scr_flops * cost.flops / max(turn_flops(spec, len(tokens), local_R, G), 1.0),
                                 cost.flops)
        turns.append(cost)
        cache.insert(tokens)
        prev = tokens
    total = sum(t.flops for t in turns)
    total_scr = sum(t.scr_flops for t in turns)
    prompt = sum(t.P for t in turns)
    return {
        "model": spec.name,
        "turns": len(turns),
        "prefix_reuse_flops": total,
        "prefix_reuse_pflops": total / 1e15,
        "scr_flops": total_scr,
        "scr_pflops": total_scr / 1e15,
        "prompt_tokens": prompt,
        "prefix_hit_tokens": sum(t.R for t in turns),
        "prefix_hit_rate": sum(t.R for t in turns) / prompt if prompt else 0.0,
        "generated_tokens": sum(t.G for t in turns),
        "per_turn": [asdict(t) for t in turns],
    }


def _lcp(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def replay_run(run_dir: str | Path, model: str = "qwen3.6-27b", **kw) -> dict:
    return replay(load_snapshots(run_dir), SPECS[model], **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prefix-reuse FLOPs for saved CLM runs.")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--model", default="qwen3.6-27b", choices=list(SPECS))
    ap.add_argument("--hf-tokenizer", help="HF model id whose chat template to tokenize with")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--no-server", action="store_true", help="ignore server-reported cached tokens")
    args = ap.parse_args(argv)
    tok = hf_tokenizer(args.hf_tokenizer) if args.hf_tokenizer else None
    for d in args.run_dirs:
        r = replay_run(d, args.model, tokenize=tok, block_size=args.block_size, use_server=not args.no_server)
        r.pop("per_turn")
        print(json.dumps({"run": str(d), **r}))


if __name__ == "__main__":
    main()
