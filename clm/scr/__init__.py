"""Suffix Cache Reuse (SCR): reuse KV states of tokens that survive a mid-context edit."""

from .diff import ReusePlan, Span, plan_reuse

__all__ = ["ReusePlan", "Span", "plan_reuse"]
