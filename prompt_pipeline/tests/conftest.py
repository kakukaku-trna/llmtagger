"""Shared pytest fixtures for LLMTagger tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Ensure llmtagger/ root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    AlertConfig,
    DataConfig,
    EarlyStop,
    InferenceConfig,
    IterationConfig,
    SceneConfig,
    TargetConfig,
    TokenBudget,
)
from pipeline.data.local_loader import Sample
from pipeline.inference.dashscope_client import InferResult


# ─────────────────────────────────────────────────────────────
# Minimal scene config (no filesystem access required)
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def minimal_scene_cfg(tmp_path) -> SceneConfig:
    cfg = SceneConfig(name="test_scene")
    cfg.data = DataConfig(
        positive_dir=str(tmp_path / "pos"),
        negative_dir=str(tmp_path / "neg"),
    )
    cfg.inference = InferenceConfig(
        engine="dashscope",
        model="qwen3.7-plus",
        api_key="sk-fake",
        workers=2,
        timeout_per_sample=10,
    )
    cfg.target = TargetConfig(precision=80.0, recall=75.0)
    cfg.iteration = IterationConfig(max_rounds=2, eval_sample_size=10, improve_topk=2)
    cfg.token_budget = TokenBudget(
        per_sample_max=1000,
        total_per_round_max=10_000,
        total_pipeline_max=30_000,
        alert_threshold=0.8,
    )
    cfg.alerts = AlertConfig()
    return cfg


# ─────────────────────────────────────────────────────────────
# A minimal YAML scene file on disk
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def scene_yaml(tmp_path) -> Path:
    pos_dir = tmp_path / "pos"
    neg_dir = tmp_path / "neg"
    pos_dir.mkdir()
    neg_dir.mkdir()

    content = {
        "name": "test_scene",
        "data": {
            "mode": "local",
            "positive_dir": str(pos_dir),
            "negative_dir": str(neg_dir),
        },
        "inference": {
            "engine": "dashscope",
            "model": "qwen3.7-plus",
            "api_key": "sk-fake",
            "workers": 2,
            "timeout_per_sample": 10,
        },
        "target": {"precision": 80.0, "recall": 75.0},
        "iteration": {"max_rounds": 2, "eval_sample_size": 10, "improve_topk": 2},
        "token_budget": {
            "per_sample_max": 1000,
            "total_per_round_max": 10000,
            "total_pipeline_max": 30000,
            "alert_threshold": 0.8,
        },
        "alerts": {"email": None},
    }
    yaml_file = tmp_path / "test_scene.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(content, f)
    return yaml_file


# ─────────────────────────────────────────────────────────────
# Sample factory helpers
# ─────────────────────────────────────────────────────────────
def make_sample(path: str, label: str) -> Sample:
    return Sample(video_path=path, label=label, uuid=path)


def make_result(
    path: str,
    label: str,
    result: str,
    status: str = "success",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> InferResult:
    return InferResult(
        video_path=path,
        label=label,
        result=result,
        reason="test reason",
        status=status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        elapsed=0.5,
        error="",
    )
