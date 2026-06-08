"""Integration tests: full pipeline run with mocked DashScope API."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import make_result, make_sample
from pipeline.config import load_scene
from pipeline.data.local_loader import load_samples
from pipeline.evaluate.metrics import calc_metrics, load_all_versions, save_version_metrics
from pipeline.improve.failure_analyzer import FailureReport, analyze_failures
from pipeline.inference.batch_runner import BatchRunner
from pipeline.monitor.token_tracker import AlertLevel, TokenTracker


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _make_video_files(directory: Path, count: int, label: str):
    """Create dummy .mp4 files (0-byte placeholders)."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"{label}_{i:02d}.mp4").touch()


def _mock_infer(sample):
    """Perfect-accuracy mock: always returns the ground-truth label."""
    return make_result(sample.video_path, sample.label, sample.label)


def _mock_infer_all_wrong(sample):
    """Worst-case mock: always inverts the label."""
    wrong = "否" if sample.label == "是" else "是"
    return make_result(sample.video_path, sample.label, wrong)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def scene_with_videos(tmp_path, scene_yaml, monkeypatch):
    """Load test_scene config with actual (empty) video files on disk."""
    pos_dir = tmp_path / "pos"
    neg_dir = tmp_path / "neg"
    _make_video_files(pos_dir, 3, "pos")
    _make_video_files(neg_dir, 3, "neg")

    # Load scene from the fixture yaml, override data dirs
    cfg = load_scene("test_scene", scenes_dir=tmp_path)
    cfg.data.positive_dir = str(pos_dir)
    cfg.data.negative_dir = str(neg_dir)
    return cfg


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────
def test_load_samples_returns_correct_count(scene_with_videos):
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=None)
    assert len(samples) == 6


def test_load_samples_labels(scene_with_videos):
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=None)
    labels = {s.label for s in samples}
    assert "是" in labels
    assert "否" in labels


def test_load_samples_with_limit(scene_with_videos):
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=4)
    assert len(samples) == 4


def test_full_pipeline_perfect_accuracy(scene_with_videos, tmp_path):
    """Run entire pipeline; verify metrics saved to metrics.json."""
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=None)

    tracker = TokenTracker(cfg.token_budget)
    runner = BatchRunner(config=cfg.inference, tracker=tracker)
    results = runner.run(samples, "test prompt", infer_fn=_mock_infer)

    metrics = calc_metrics(results)
    assert metrics.success == 6
    assert metrics.precision == 100.0
    assert metrics.recall == 100.0
    assert metrics.f1 == 100.0

    # Save and reload
    metrics_path = tmp_path / "metrics.json"
    save_version_metrics("test_scene", "v1", metrics, metrics_path)
    loaded = load_all_versions(metrics_path)
    assert "v1" in loaded
    assert loaded["v1"].f1 == 100.0


def test_full_pipeline_all_wrong(scene_with_videos, tmp_path):
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=None)

    runner = BatchRunner(config=cfg.inference)
    results = runner.run(samples, "test prompt", infer_fn=_mock_infer_all_wrong)

    metrics = calc_metrics(results)
    assert metrics.f1 == 0.0


def test_failure_analyzer_finds_fp_fn(scene_with_videos):
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=None)

    runner = BatchRunner(config=cfg.inference)
    results = runner.run(samples, "test prompt", infer_fn=_mock_infer_all_wrong)

    report = analyze_failures(results)
    assert report.total_failures == 6
    # All pos samples predicted "否" → FN; all neg predicted "是" → FP
    assert len(report.fn_cases) == 3
    assert len(report.fp_cases) == 3


def test_token_tracker_accumulates(scene_with_videos):
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=None)

    tracker = TokenTracker(cfg.token_budget)
    runner = BatchRunner(config=cfg.inference, tracker=tracker)
    runner.run(samples, "test prompt", infer_fn=_mock_infer)

    # each make_result: 150 tokens; 6 samples total
    assert tracker.round_total == 6 * 150


def test_metrics_json_written(scene_with_videos, tmp_path):
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=6)

    runner = BatchRunner(config=cfg.inference)
    results = runner.run(samples, "prompt", infer_fn=_mock_infer)
    m = calc_metrics(results)

    metrics_path = tmp_path / "metrics.json"
    save_version_metrics("test_scene", "v1", m, metrics_path)

    assert metrics_path.exists()
    with open(metrics_path) as f:
        raw = json.load(f)
    assert "v1" in raw
    assert raw["v1"]["TP"] > 0 or raw["v1"]["TN"] > 0


def test_meets_target_after_perfect_run(scene_with_videos):
    cfg = scene_with_videos
    samples = load_samples(cfg.data, sample_size=6)
    runner = BatchRunner(config=cfg.inference)
    results = runner.run(samples, "prompt", infer_fn=_mock_infer)
    m = calc_metrics(results)
    assert m.meets_target(cfg.target.precision, cfg.target.recall)


def test_budget_check_ok(scene_with_videos):
    cfg = scene_with_videos
    tracker = TokenTracker(cfg.token_budget)
    tracker.add(100, 50)
    assert tracker.check_pipeline_budget() == AlertLevel.OK


def test_budget_exceeded_detection(scene_with_videos):
    cfg = scene_with_videos
    tracker = TokenTracker(cfg.token_budget)
    tracker.add(cfg.token_budget.total_pipeline_max + 1, 0)
    assert tracker.check_pipeline_budget() == AlertLevel.EXCEEDED
