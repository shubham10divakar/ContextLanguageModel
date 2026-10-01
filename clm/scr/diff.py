"""Token diff for Suffix Cache Reuse (paper Sec. 4.3, App. B).

Given the session's previous prompt and the new one, find the shared prefix
and the spans after it that survived the edit unchanged. SCR relocates the
K longest surviving spans (K = 6 in the paper) instead of re-prefilling
them. Following the reference code, the last ``min_extend`` (16) tokens of
each relocated span are re-prefilled rather than relocated, so the tokens
right after a span boundary are recomputed under the new context.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Span:
    old_start: int
    new_start: int
    length: int

    @property
    def shift(self) -> int:
        return self.new_start - self.old_start


@dataclass
class ReusePlan:
    n_new: int
    prefix: int                                   # reused as-is (standard prefix cache)
    relocated: list[Span] = field(default_factory=list)  # reused with RoPE re-rotation

    @property
    def relocated_tokens(self) -> int:
        return sum(s.length for s in self.relocated)

    @property
    def reused_tokens(self) -> int:
        return self.prefix + self.relocated_tokens

    def prefill_mask(self) -> list[bool]:
        """True for positions of the new prompt that must be (re-)prefilled."""
        mask = [True] * self.n_new
        for i in range(self.prefix):
            mask[i] = False
        for s in self.relocated:
            for i in range(s.new_start, s.new_start + s.length):
                mask[i] = False
        return mask


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def surviving_spans(old: Sequence[int], new: Sequence[int], prefix: int) -> list[Span]:
    """Unchanged spans of ``new`` (after ``prefix``) that also occur in ``old``, in order."""
    sm = difflib.SequenceMatcher(None, list(old), list(new), autojunk=False)
    spans = []
    for a, b, size in sm.get_matching_blocks():
        if size == 0:
            continue
        if b < prefix:                       # clip the part that overlaps the prefix
            cut = prefix - b
            if cut >= size:
                continue
            a, b, size = a + cut, b + cut, size - cut
        spans.append(Span(a, b, size))
    return spans


def plan_reuse(old: Sequence[int], new: Sequence[int], k: int = 6, min_extend: int = 16,
               prefix: int | None = None) -> ReusePlan:
    """Plan which tokens of ``new`` SCR reuses, relocates or prefills."""
    lcp = common_prefix(old, new)
    prefix = lcp if prefix is None else max(prefix, 0)
    plan = ReusePlan(n_new=len(new), prefix=prefix)
    if lcp >= len(old) or k <= 0:              # append-only: nothing after the prefix survives
        return plan
    spans = surviving_spans(old, new, prefix)
    spans.sort(key=lambda s: s.length, reverse=True)
    for s in spans[:k]:
        keep = s.length - min_extend
        if keep > 0:
            plan.relocated.append(Span(s.old_start, s.new_start, keep))
    plan.relocated.sort(key=lambda s: s.new_start)
    return plan
