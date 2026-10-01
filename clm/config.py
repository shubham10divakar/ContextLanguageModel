"""Harness configuration (mirrors the knobs of the reference ``configs/*.yaml``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass
class CLMConfig:
    # --- budget -----------------------------------------------------------
    context_budget_tokens: int = 28672
    reserve_tokens: int = 2048           # held back for the response
    max_tokens: int = 4096               # per LLM call
    # --- sampling ---------------------------------------------------------
    temperature: float = 0.7
    top_p: float = 0.95
    # --- context management ---------------------------------------------
    strategy: str = "clm"                # clm | base | summary
    edit_gate: str = "fit"               # fit | shrink
    max_num_retry_on_limit: int = 6      # rollback-and-retry budget
    nudge_ratios: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    urgent_mode: str = "ratio"           # ratio | adaptive | none
    persistent_nudge_ratio: float = 0.9
    observation_max_chars: int = 60000
    include_reasoning_in_file: bool = True
    send_reasoning: bool = False         # chat templates usually strip old reasoning
    summary_trigger_ratio: float = 0.75  # only for strategy=summary
    escape_headers: bool = False         # opt-in defense against forged [[CTX_TURN]] headers
    # --- loop -------------------------------------------------------------
    max_steps: int = 100                 # edit-only turns are not counted
    max_total_turns: int = 400           # hard cap including edit-only turns
    command_timeout: float = 180.0
    final_turns_warning: int = 3
    # --- steering / skills (Sec. 4.2) ------------------------------------
    extra_instructions: str = ""
    skill_path: str | None = None
    # --- misc -------------------------------------------------------------
    no_trailing_assistant: bool = False  # Anthropic-style APIs reject a trailing assistant turn

    @property
    def limit(self) -> int:
        return self.context_budget_tokens - self.reserve_tokens

    @classmethod
    def load(cls, name_or_path: str | None = None, **overrides) -> "CLMConfig":
        data: dict = {}
        if name_or_path:
            path = Path(name_or_path)
            if not path.exists():
                path = CONFIG_DIR / f"{name_or_path}.yaml"
            data = yaml.safe_load(path.read_text()) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
