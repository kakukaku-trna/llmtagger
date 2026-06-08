"""Tests for pipeline/inference/batch_runner.py – uses mock infer_fn."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import make_result, make_sample
from pipeline.config import InferenceConfig
from pipeline.inference.batch_runner import BatchRunner, _dict_to_result, _result_to_dict
from pipeline.inference.dashscope_client import InferResult
from pipeline.monitor.token_tracker import TokenTracker


# ─────────────────────────────────────────────────────────────
def _fast_infer_fn(sample):
    return make_result(sample.video_path, sample.label, sample.label)  # perfect accuracy


def _failing_infer_fn(sample):
    raise RuntimeError("API error")


# ─────────────────────────────────────────────────────────────
def test_run_returns_all_results(minimal_scene_cfg):
    samples = [make_sample(f"v{i}.mp4", "是") for i in range(3)]
    runner = BatchRunner(config=minimal_scene_cfg.inference)
    results = runner.run(samples, "prompt", infer_fn=_fast_infer_fn)
    assert len(results) == 3


def test_run_correct_labels(minimal_scene_cfg):
    samples = [
        make_sample("a.mp4", "是"),
        make_sample("b.mp4", "否"),
    ]
    runner = BatchRunner(config=minimal_scene_cfg.inference)
    results = runner.run(samples, "prompt", infer_fn=_fast_infer_fn)
    by_path = {r.video_path: r for r in results}
    assert by_path["a.mp4"].result == "是"
    assert by_path["b.mp4"].result == "否"


def test_run_handles_exceptions(minimal_scene_cfg):
    samples = [make_sample("err.mp4", "是")]
    runner = BatchRunner(config=minimal_scene_cfg.inference)
    results = runner.run(samples, "prompt", infer_fn=_failing_infer_fn)
    assert len(results) == 1
    assert results[0].status == "failed"
    assert "API error" in results[0].error


def test_cache_hit_skips_infer(minimal_scene_cfg, tmp_path):
    cache_path = str(tmp_path / "cache.json")
    samples = [make_sample("cached.mp4", "是")]
    runner = BatchRunner(config=minimal_scene_cfg.inference, cache_path=cache_path)

    calls = []
    def counting_infer(sample):
        calls.append(sample)
        return make_result(sample.video_path, sample.label, "是")

    # First run – fills cache
    runner.run(samples, "prompt", infer_fn=counting_infer)
    assert len(calls) == 1

    # Second run – cache hit, no new call
    runner2 = BatchRunner(config=minimal_scene_cfg.inference, cache_path=cache_path)
    results = runner2.run(samples, "prompt", infer_fn=counting_infer)
    assert len(calls) == 1   # unchanged
    assert results[0].video_path == "cached.mp4"


def test_cache_persisted_to_json(minimal_scene_cfg, tmp_path):
    cache_path = str(tmp_path / "cache.json")
    samples = [make_sample("v.mp4", "否")]
    runner = BatchRunner(config=minimal_scene_cfg.inference, cache_path=cache_path)
    runner.run(samples, "prompt", infer_fn=_fast_infer_fn)

    with open(cache_path) as f:
        data = json.load(f)
    assert "v.mp4" in data


def test_token_tracker_updated(minimal_scene_cfg):
    tracker = TokenTracker()
    samples = [make_sample(f"x{i}.mp4", "是") for i in range(3)]
    runner = BatchRunner(config=minimal_scene_cfg.inference, tracker=tracker)
    runner.run(samples, "prompt", infer_fn=_fast_infer_fn)
    # each make_result has 100+50=150 tokens
    assert tracker.round_total == 3 * 150


def test_result_serialization_roundtrip():
    r = make_result("test.mp4", "是", "否", status="success")
    d = _result_to_dict(r)
    r2 = _dict_to_result(d)
    assert r2.video_path == "test.mp4"
    assert r2.label == "是"
    assert r2.result == "否"
    assert r2.status == "success"
