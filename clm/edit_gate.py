"""Edit gate: decides whether a context-file edit is applied.

* ``fit`` (default): accept any edit whose result fits the usable limit.
* ``shrink`` (reference setting for BrowseComp-Plus): accept only edits that
  make the context smaller.

A rejected edit is not applied and the model receives a fixed message.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateDecision:
    accepted: bool
    reason: str = ""


def check_edit(mode: str, current_tokens: int, candidate_tokens: int, limit: int) -> GateDecision:
    if candidate_tokens > limit:
        return GateDecision(False, f"EDIT REJECTED: the edited context (~{candidate_tokens} tokens) exceeds "
                                   f"the limit ({limit}). Your previous context was kept unchanged.")
    if mode == "shrink" and candidate_tokens >= current_tokens:
        return GateDecision(False, f"EDIT REJECTED: edits must make the context smaller "
                                   f"(~{current_tokens} -> ~{candidate_tokens} tokens). "
                                   f"Your previous context was kept unchanged.")
    if mode not in ("fit", "shrink"):
        raise ValueError(f"unknown edit gate mode: {mode}")
    return GateDecision(True)
