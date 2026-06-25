"""Tests for pipeline/data/local_loader.py unlabeled input loading."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.data.local_loader import load_unlabeled_samples


def test_load_unlabeled_samples_from_file(tmp_path):
    video_path = tmp_path / "demo.mp4"
    video_path.touch()

    samples = load_unlabeled_samples([str(video_path)])

    assert len(samples) == 1
    assert samples[0].video_path == str(video_path)
    assert samples[0].label == ""


def test_load_unlabeled_samples_from_frame_dir(tmp_path):
    frame_dir = tmp_path / "clip_a"
    frame_dir.mkdir()
    (frame_dir / "frame_01.jpg").touch()
    (frame_dir / "frame_02.jpg").touch()

    samples = load_unlabeled_samples([str(tmp_path)])

    assert len(samples) == 1
    assert samples[0].video_path == str(frame_dir)
    assert samples[0].uuid == "clip_a"
