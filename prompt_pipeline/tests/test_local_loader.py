"""Tests for camera suffix filter in local_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import DataConfig
from pipeline.data.local_loader import _discover_media, _load_dir, load_samples


# ─────────────────────────────────────────
# Helper: build a temp directory with mixed files
# ─────────────────────────────────────────
def _build_mixed_dir(tmp_path: Path) -> Path:
    """Create a directory with:
    - video_front.mp4
    - video_back.mp4
    - video_left.mp4
    - other_clip.mp4
    - scene_front/ (dir with frame_*.jpg -> should also match camera filter)
    """
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


# ─────────────────────────────────────────
# RED – Tests should fail before implementation
# ─────────────────────────────────────────
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
        """Empty string suffix should act like no filter."""
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
        assert len(samples) == 2  # one positive, one negative
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
        # camera_suffix="front" 只会留下 1 正 1 负，sample_size=4 应该返回 2 个
        samples = load_samples(cfg, sample_size=4, camera_suffix="front")
        assert len(samples) == 2
