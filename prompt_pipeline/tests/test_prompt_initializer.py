"""Tests for pipeline/improve/prompt_initializer.py helper context builders."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.improve.prompt_initializer import (
    _build_scene_context,
    _infer_modality_hints,
)


def test_build_scene_context_includes_label_semantics(minimal_scene_cfg, tmp_path):
    pos_dir = tmp_path / "multicam_animal_pos"
    neg_dir = tmp_path / "multicam_animal_neg"
    pos_dir.mkdir()
    neg_dir.mkdir()
    (pos_dir / "clip_a.mp4").touch()
    (neg_dir / "clip_b.mp4").touch()

    minimal_scene_cfg.name = "animal_multicam"
    minimal_scene_cfg.display_name = "天气检测-多摄"
    minimal_scene_cfg.data.mode = "local"
    minimal_scene_cfg.data.positive_dir = str(pos_dir)
    minimal_scene_cfg.data.negative_dir = str(neg_dir)

    context = _build_scene_context(minimal_scene_cfg, "检测视频中是否出现动物")

    assert "positive_dir 下样本应判“是”" in context
    assert "negative_dir 下样本应判“否”" in context
    assert "multicam_animal_pos" in context
    assert "multicam_animal_neg" in context
    assert "clip_a.mp4" in context
    assert "clip_b.mp4" in context


def test_infer_modality_hints_from_scene_metadata(minimal_scene_cfg):
    minimal_scene_cfg.name = "animal_multicam"
    minimal_scene_cfg.display_name = "天气检测-多摄"
    minimal_scene_cfg.data.positive_dir = "/tmp/front120_multicam_pos"
    minimal_scene_cfg.data.negative_dir = "/tmp/front120_multicam_neg"

    hints = _infer_modality_hints(minimal_scene_cfg, "检测视频中是否出现动物")

    assert "多摄或拼接视频输入" in hints
    assert "主视角可能是 Front120" in hints
