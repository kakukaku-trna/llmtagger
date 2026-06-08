"""Tests for pipeline/evaluate/metrics.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import make_result
from pipeline.evaluate.metrics import (
    Metrics,
    calc_metrics,
    load_all_versions,
    save_version_metrics,
)


# ─────────────────────────────────────────────────────────────
def _results(tp=0, fp=0, fn=0, tn=0):
    items = []
    for i in range(tp):
        items.append(make_result(f"tp_{i}.mp4", label="是", result="是"))
    for i in range(fp):
        items.append(make_result(f"fp_{i}.mp4", label="否", result="是"))
    for i in range(fn):
        items.append(make_result(f"fn_{i}.mp4", label="是", result="否"))
    for i in range(tn):
        items.append(make_result(f"tn_{i}.mp4", label="否", result="否"))
    return items


# ─────────────────────────────────────────────────────────────
def test_all_tp():
    m = calc_metrics(_results(tp=5))
    assert m.TP == 5
    assert m.FP == 0
    assert m.precision == 100.0
    assert m.recall == 100.0
    assert m.f1 == 100.0


def test_all_fp():
    m = calc_metrics(_results(fp=5))
    assert m.FP == 5
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_all_fn():
    m = calc_metrics(_results(fn=5))
    assert m.FN == 5
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_all_tn():
    m = calc_metrics(_results(tn=5))
    assert m.TN == 5
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_empty_results():
    m = calc_metrics([])
    assert m.total == 0
    assert m.f1 == 0.0


def test_mixed_typical():
    # tp=3, fp=1, fn=1, tn=5
    m = calc_metrics(_results(tp=3, fp=1, fn=1, tn=5))
    expected_p = 3 / 4 * 100
    expected_r = 3 / 4 * 100
    assert abs(m.precision - round(expected_p, 1)) < 0.01
    assert abs(m.recall - round(expected_r, 1)) < 0.01
    assert m.success == 10


def test_failed_results_excluded():
    results = _results(tp=3, fp=1)
    failed = make_result("fail.mp4", "是", "", status="failed")
    results.append(failed)
    m = calc_metrics(results)
    assert m.total == 5
    assert m.failed == 1
    assert m.success == 4


def test_tokens_summed_from_results():
    results = _results(tp=2)
    m = calc_metrics(results)
    assert m.tokens_used == 2 * 150   # each result: 100+50=150


def test_tokens_explicit_override():
    results = _results(tp=2)
    m = calc_metrics(results, tokens_used=9999)
    assert m.tokens_used == 9999


def test_meets_target():
    m = Metrics(precision=85.0, recall=80.0)
    assert m.meets_target(80.0, 75.0) is True
    assert m.meets_target(90.0, 75.0) is False


def test_better_than():
    a = Metrics(f1=70.0)
    b = Metrics(f1=60.0)
    assert a.better_than(b) is True
    assert b.better_than(a) is False


def test_save_and_load_version_metrics(tmp_path):
    m = calc_metrics(_results(tp=3, fp=1, fn=1, tn=4))
    metrics_path = tmp_path / "metrics.json"
    save_version_metrics("test", "v1", m, metrics_path)

    loaded = load_all_versions(metrics_path)
    assert "v1" in loaded
    assert abs(loaded["v1"].f1 - m.f1) < 0.01


def test_load_all_versions_multiple(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    for ver, tp in [("v1", 2), ("v2", 4)]:
        m = calc_metrics(_results(tp=tp, tn=tp))
        save_version_metrics("test", ver, m, metrics_path)

    all_v = load_all_versions(metrics_path)
    assert set(all_v.keys()) == {"v1", "v2"}


def test_load_all_versions_empty(tmp_path):
    result = load_all_versions(tmp_path / "nonexistent.json")
    assert result == {}
