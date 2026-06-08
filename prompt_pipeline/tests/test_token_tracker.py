"""Tests for pipeline/monitor/token_tracker.py."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import TokenBudget
from pipeline.monitor.token_tracker import AlertLevel, TokenTracker, TokenUsage


# ─────────────────────────────────────────────────────────────
def test_add_and_totals():
    t = TokenTracker()
    t.add(100, 50)
    t.add(200, 100)
    assert t.round_total == 450
    assert t.pipeline_total == 450


def test_end_round_resets_round_counter():
    t = TokenTracker()
    t.add(100, 50)
    finished = t.end_round()
    assert finished.total == 150
    assert t.round_total == 0          # reset
    assert t.pipeline_total == 150     # cumulative intact


def test_multiple_rounds_accumulate():
    t = TokenTracker()
    t.add(100, 50)
    t.end_round()
    t.add(200, 100)
    t.end_round()
    assert t.pipeline_total == 450


def test_no_budget_always_ok():
    t = TokenTracker()
    t.add(10_000_000, 0)
    assert t.check_round_budget() == AlertLevel.OK
    assert t.check_pipeline_budget() == AlertLevel.OK


def test_round_budget_warning():
    budget = TokenBudget(total_per_round_max=10_000, alert_threshold=0.8)
    t = TokenTracker(budget)
    t.add(8_500, 0)  # 85% > 80% threshold
    assert t.check_round_budget() == AlertLevel.WARNING


def test_round_budget_exceeded():
    budget = TokenBudget(total_per_round_max=10_000, alert_threshold=0.8)
    t = TokenTracker(budget)
    t.add(10_001, 0)
    assert t.check_round_budget() == AlertLevel.EXCEEDED


def test_pipeline_budget_ok():
    budget = TokenBudget(total_pipeline_max=30_000, alert_threshold=0.8)
    t = TokenTracker(budget)
    t.add(1_000, 0)
    assert t.check_pipeline_budget() == AlertLevel.OK


def test_pipeline_budget_exceeded():
    budget = TokenBudget(total_pipeline_max=30_000, alert_threshold=0.8)
    t = TokenTracker(budget)
    t.add(30_001, 0)
    assert t.check_pipeline_budget() == AlertLevel.EXCEEDED


def test_thread_safety():
    """Concurrent adds should not lose counts."""
    t = TokenTracker()
    def worker():
        for _ in range(100):
            t.add(1, 1)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert t.round_total == 2_000   # 10 threads × 100 × (1+1)
    assert t.pipeline_total == 2_000


def test_summary_line_contains_key_info():
    budget = TokenBudget(total_pipeline_max=30_000)
    t = TokenTracker(budget)
    t.add(1_000, 500)
    line = t.summary_line(scene="blind_curve", round_n=1)
    assert "blind_curve" in line
    assert "1,500" in line or "1500" in line


def test_token_usage_total():
    u = TokenUsage(prompt_tokens=300, completion_tokens=100)
    assert u.total == 400
