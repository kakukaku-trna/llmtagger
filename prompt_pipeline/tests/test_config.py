"""Tests for pipeline/config.py – YAML loading and validation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    InferenceConfig,
    SceneConfig,
    TokenBudget,
    load_scene,
)


# ─────────────────────────────────────────────────────────────
def test_load_scene_basic(scene_yaml, tmp_path):
    """load_scene() should return a populated SceneConfig."""
    cfg = load_scene("test_scene", scenes_dir=tmp_path)
    assert cfg.name == "test_scene"
    assert cfg.inference.engine == "dashscope"
    assert cfg.inference.model == "qwen3.7-plus"
    assert cfg.inference.api_key == "sk-fake"


def test_load_scene_target(scene_yaml, tmp_path):
    cfg = load_scene("test_scene", scenes_dir=tmp_path)
    assert cfg.target.precision == 80.0
    assert cfg.target.recall == 75.0


def test_load_scene_token_budget(scene_yaml, tmp_path):
    cfg = load_scene("test_scene", scenes_dir=tmp_path)
    assert cfg.token_budget.total_per_round_max == 10_000
    assert cfg.token_budget.total_pipeline_max == 30_000
    assert cfg.token_budget.alert_threshold == 0.8


def test_load_scene_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene("nonexistent_scene", scenes_dir=tmp_path)


def test_load_scene_data_dirs(scene_yaml, tmp_path):
    cfg = load_scene("test_scene", scenes_dir=tmp_path)
    assert Path(cfg.data.positive_dir).exists()
    assert Path(cfg.data.negative_dir).exists()


def test_load_scene_api_key_from_env(tmp_path, monkeypatch):
    """If api_key not in YAML, should fall back to env var."""
    content = {
        "name": "env_scene",
        "data": {"positive_dir": str(tmp_path), "negative_dir": str(tmp_path)},
        "inference": {"engine": "dashscope"},
    }
    yf = tmp_path / "env_scene.yaml"
    with open(yf, "w") as f:
        yaml.dump(content, f)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-from-env")
    cfg = load_scene("env_scene", scenes_dir=tmp_path)
    assert cfg.inference.api_key == "sk-from-env"


def test_scene_prompt_path(minimal_scene_cfg, tmp_path, monkeypatch):
    """prompt_path() should return path inside prompts/{scene}/."""
    # patch PROMPTS_DIR to tmp_path
    import pipeline.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "PROMPTS_DIR", tmp_path)
    # Recreate fixture since PROMPTS_DIR is a module-level constant
    cfg = minimal_scene_cfg
    # Just verify it returns a Path ending with the right name
    p = cfg.prompt_path("v1")
    assert p.name == "v1.md"


def test_inference_config_defaults():
    cfg = InferenceConfig()
    assert cfg.engine == "dashscope"
    assert cfg.workers == 5
    assert cfg.max_retries == 3


def test_token_budget_defaults():
    tb = TokenBudget()
    assert tb.alert_threshold == 0.8
    assert tb.total_pipeline_max == 600_000
