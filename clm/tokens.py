"""Token counting with server calibration.

The harness counts with tiktoken (o200k_base, as in the paper's Appendix E)
and rescales its estimate by the ratio between the server's reported
``prompt_tokens`` and the local estimate for the same prompt, smoothed over
turns. This keeps the "[context: ~N/limit tokens]" readout close to what the
serving tokenizer actually sees.
"""

from __future__ import annotations

from functools import lru_cache

from .context_file import message_text

PER_MESSAGE_OVERHEAD = 4  # role markers / separators of a typical chat template


@lru_cache(maxsize=4)
def _encoding(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


class TokenCounter:
    def __init__(self, encoding: str = "o200k_base", include_reasoning: bool = False,
                 smoothing: float = 0.5):
        self.encoding_name = encoding
        self.include_reasoning = include_reasoning
        self.smoothing = smoothing
        self.scale = 1.0

    def count_text(self, text: str) -> int:
        return len(_encoding(self.encoding_name).encode(text, disallowed_special=()))

    def raw_count(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += PER_MESSAGE_OVERHEAD + self.count_text(message_text(m, self.include_reasoning))
        return total

    def count(self, messages: list[dict]) -> int:
        return int(round(self.raw_count(messages) * self.scale))

    def calibrate(self, messages: list[dict], server_prompt_tokens: int | None) -> None:
        """Update the scale from one (prompt, server-reported length) pair."""
        if not server_prompt_tokens:
            return
        raw = self.raw_count(messages)
        if raw <= 0:
            return
        ratio = server_prompt_tokens / raw
        self.scale = self.smoothing * self.scale + (1 - self.smoothing) * ratio
