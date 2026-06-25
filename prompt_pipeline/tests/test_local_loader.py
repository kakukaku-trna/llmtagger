"""Tests for pipeline/data/local_loader.py camera filtering and unlabeled loading."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import DataConfig
from pipeline.data.local_loader import (
    _discover_media,
    _load_dir,
    load_samples,
    load_unlabeled_samples,
)


def _build_mixed_dir(tmp_path: Path) -> Path:
    """Create a directory with mixed camera videos and one frame directory."""
    root = tmp_path / "mixed"
    root.mkdir()

    for name in (
        "video_front.mp4",
        "video_back.mp4",
        "video_left.mp4",
        "other_clip.mp4",
    ):
        (root / name).write_text("fake mp4")

    front_dir = root / "scene_front"
    front_dir.mkdir()
    (front_dir / "frame_001.jpg").write_text("fake frame")

    return root


class TestDiscoverMediaCameraFilter:
    def test_filter_by_suffix_front_only(self, tmp_path: Path):
        root = _build_mixed_dir(tmp_path)
        result = _discover_media(str(root), camera_suffix="front")
        names = [
            Path(p).stem if Path(p).suffix == ".mp4" else Path(p).name for p in result
        ]
        assert sorted(names) == ["scene_front", "video_front"]

    def test_filter_by_suffix_back_only(self, tmp_path: Path):
        root = _build_mixed_dir(tmp_path)
        result = _discover_media(str(root), camera_suffix="back")
        names = [
            Path(p).stem if Path(p).suffix == ".mp4" else Path(p).name for p in result
        ]
        assert sorted(names) == ["video_back"]

    def test_no_filter_returns_all(self, tmp_path: Path):
        root = _build_mixed_dir(tmp_path)
        result = _discover_media(str(root), camera_suffix=None)
        names = [
            Path(p).stem if Path(p).suffix == ".mp4" else Path(p).name for p in result
        ]
        assert sorted(names) == [
            "other_clip",
            "scene_front",
            "video_back",
            "video_front",
            "video_left",
        ]

    def test_filter_no_match_returns_empty(self, tmp_path: Path):
        root = _build_mixed_dir(tmp_path)
        result = _discover_media(str(root), camera_suffix="nonexistent")
        assert result == []

    def test_filter_empty_string_no_effect(self, tmp_path: Path):
        root = _build_mixed_dir(tmp_path)
        result = _discover_media(str(root), camera_suffix="")
        names = [
            Path(p).stem if Path(p).suffix == ".mp4" else Path(p).name for p in result
        ]
        assert sorted(names) == [
            "other_clip",
            "scene_front",
            "video_back",
            "video_front",
            "video_left",
        ]


class TestLoadDirCameraFilter:
    def test_load_dir_with_suffix(self, tmp_path: Path):
        root = _build_mixed_dir(tmp_path)
        samples = _load_dir(str(root), label="是", camera_suffix="front")
        uuids = [s.uuid for s in samples]
        assert sorted(uuids) == ["scene_front", "video_front"]
        assert all(s.label == "是" for s in samples)

    def test_load_dir_without_suffix_returns_all(self, tmp_path: Path):
        root = _build_mixed_dir(tmp_path)
        samples = _load_dir(str(root), label="是", camera_suffix=None)
        assert len(samples) == 5


class TestLoadSamplesCameraFilter:
    def test_load_samples_balanced_with_suffix(self, tmp_path: Path):
        pos_dir = tmp_path / "pos"
        neg_dir = tmp_path / "neg"
        pos_dir.mkdir()
        neg_dir.mkdir()

        for name in ("video_front.mp4", "video_back.mp4", "video_left.mp4"):
            (pos_dir / name).write_text("fake mp4")
            (neg_dir / name).write_text("fake mp4")

        cfg = DataConfig(positive_dir=str(pos_dir), negative_dir=str(neg_dir))
        samples = load_samples(cfg, camera_suffix="front")
        assert len(samples) == 2
        assert all(s.uuid == "video_front" for s in samples)

    def test_load_samples_no_suffix_returns_all(self, tmp_path: Path):
        pos_dir = tmp_path / "pos"
        neg_dir = tmp_path / "neg"
        pos_dir.mkdir()
        neg_dir.mkdir()

        for name in ("video_front.mp4", "video_back.mp4"):
            (pos_dir / name).write_text("fake mp4")
            (neg_dir / name).write_text("fake mp4")

        cfg = DataConfig(positive_dir=str(pos_dir), negative_dir=str(neg_dir))
        samples = load_samples(cfg)
        assert len(samples) == 4

    def test_load_samples_sample_size_with_suffix(self, tmp_path: Path):
        pos_dir = tmp_path / "pos"
        neg_dir = tmp_path / "neg"
        pos_dir.mkdir()
        neg_dir.mkdir()

        for name in ("video_front.mp4", "video_back.mp4", "video_left.mp4"):
            (pos_dir / name).write_text("fake mp4")
            (neg_dir / name).write_text("fake mp4")

        cfg = DataConfig(positive_dir=str(pos_dir), negative_dir=str(neg_dir))
        samples = load_samples(cfg, sample_size=4, camera_suffix="front")
        assert len(samples) == 2


class TestLoadUnlabeledSamples:
    def test_load_unlabeled_samples_from_file(self, tmp_path: Path):
        video_path = tmp_path / "demo.mp4"
        video_path.touch()

        samples = load_unlabeled_samples([str(video_path)])

        assert len(samples) == 1
        assert samples[0].video_path == str(video_path)
        assert samples[0].label == ""
        assert samples[0].uuid == "demo"

    def test_load_unlabeled_samples_from_frame_dir(self, tmp_path: Path):
        frame_dir = tmp_path / "clip_a"
        frame_dir.mkdir()
        (frame_dir / "frame_01.jpg").touch()
        (frame_dir / "frame_02.jpg").touch()

        samples = load_unlabeled_samples([str(tmp_path)])

        assert len(samples) == 1
        assert samples[0].video_path == str(frame_dir)
        assert samples[0].uuid == "clip_a"

    def test_load_unlabeled_samples_with_camera_suffix(self, tmp_path: Path):
        root = _build_mixed_dir(tmp_path)

        samples = load_unlabeled_samples([str(root)], camera_suffix="front")

        names = [
            Path(s.video_path).stem
            if Path(s.video_path).suffix == ".mp4"
            else Path(s.video_path).name
            for s in samples
        ]
        assert sorted(names) == ["scene_front", "video_front"]
