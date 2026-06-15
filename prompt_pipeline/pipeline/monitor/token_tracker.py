"""Thread-safe token usage tracker with budget alerting."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pipeline.config import TokenBudget


class AlertLevel(Enum):
    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenTracker:
    """Accumulates token usage across the pipeline, supports per-round and total budgets."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        self._budget = budget
        self._lock = threading.Lock()
        self._round_usage = TokenUsage()
        self._pipeline_usage = TokenUsage()
        self._rounds_history: list[TokenUsage] = []

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._round_usage.prompt_tokens += prompt_tokens
            self._round_usage.completion_tokens += completion_tokens
            self._pipeline_usage.prompt_tokens += prompt_tokens
            self._pipeline_usage.completion_tokens += completion_tokens

    def end_round(self) -> TokenUsage:
        """Finalize current round, start fresh round counter."""
        with self._lock:
            finished = TokenUsage(
                self._round_usage.prompt_tokens,
                self._round_usage.completion_tokens,
            )
            self._rounds_history.append(finished)
            self._round_usage = TokenUsage()
            return finished

    @property
    def round_total(self) -> int:
        with self._lock:
            return self._round_usage.total

    @property
    def pipeline_total(self) -> int:
        with self._lock:
            return self._pipeline_usage.total

    def check_round_budget(self) -> AlertLevel:
        if self._budget is None:
            return AlertLevel.OK
        with self._lock:
            used = self._round_usage.total
            limit = self._budget.total_per_round_max
        return _classify(used, limit, self._budget.alert_threshold)

    def check_pipeline_budget(self) -> AlertLevel:
        if self._budget is None:
            return AlertLevel.OK
        with self._lock:
            used = self._pipeline_usage.total
            limit = self._budget.total_pipeline_max
        return _classify(used, limit, self._budget.alert_threshold)

    def summary_line(self, scene: str = "", round_n: Optional[int] = None) -> str:
        """Single-line summary for console output."""
        with self._lock:
            rt = self._round_usage.total
            pt = self._pipeline_usage.total
        parts = []
        if scene:
            parts.append(f"场景={scene}")
        if round_n is not None:
            parts.append(f"第{round_n}轮")
        parts.append(f"本轮={rt:,} tokens")
        parts.append(f"累计={pt:,} tokens")
        if self._budget:
            pct = pt / self._budget.total_pipeline_max * 100
            parts.append(f"({pct:.1f}% of {self._budget.total_pipeline_max:,})")
        return "  ".join(parts)

    def table_row(self, scene: str, round_n: int) -> str:
        with self._lock:
            rt = self._round_usage.total
            in_tok = self._pipeline_usage.prompt_tokens
            out_tok = self._pipeline_usage.completion_tokens
        # qwen3.7-plus: input ¥2/M, output ¥8/M
        cost = in_tok / 1_000_000 * 2.0 + out_tok / 1_000_000 * 8.0
        return f"{scene:<20} | {round_n:>4} | {rt:>10,} | ¥{cost:.2f}"


def _classify(used: int, limit: int, threshold: float) -> AlertLevel:
    if used >= limit:
        return AlertLevel.EXCEEDED
    if used >= limit * threshold:
        return AlertLevel.WARNING
    return AlertLevel.OK
