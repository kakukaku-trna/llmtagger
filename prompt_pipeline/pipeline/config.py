"""Scene configuration loader. Reads scenes/{name}.yaml into typed dataclasses."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

SCENES_DIR = Path(__file__).parent.parent / "scenes"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass
class DataConfig:
    mode: str = "local"          # local | stream
    positive_dir: str = ""
    negative_dir: str = ""
    dataset: Optional[str] = None


@dataclass
class InferenceConfig:
    engine: str = "dashscope"    # dashscope | vllm
    model: str = "qwen3.7-plus"
    workers: int = 5
    timeout_per_sample: int = 60
    fps: int = 2
    thinking_budget: int = 512
    max_retries: int = 3
    api_key: str = ""
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vllm_url: str = "http://localhost:8000/v1/chat/completions"


@dataclass
class TargetConfig:
    precision: float = 80.0
    recall: float = 75.0
    accuracy: float = 0.0  # 0 = not enforced


@dataclass
class EarlyStop:
    no_improvement_rounds: int = 2
    token_budget_exceeded: bool = True


@dataclass
class IterationConfig:
    max_rounds: int = 3
    eval_sample_size: int = 200
    full_eval_on_final: bool = True
    improve_topk: int = 3
    early_stop: EarlyStop = field(default_factory=EarlyStop)


@dataclass
class TokenBudget:
    per_sample_max: int = 3000
    total_per_round_max: int = 200_000
    total_pipeline_max: int = 600_000
    alert_threshold: float = 0.8


@dataclass
class AlertConfig:
    email: Optional[str] = None


@dataclass
class SceneConfig:
    name: str
    display_name: str = ""
    version: str = "1.0"
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    iteration: IterationConfig = field(default_factory=IterationConfig)
    token_budget: TokenBudget = field(default_factory=TokenBudget)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    def prompts_dir(self) -> Path:
        return PROMPTS_DIR / self.name

    def metrics_path(self) -> Path:
        return self.prompts_dir() / "metrics.json"

    def history_path(self) -> Path:
        return self.prompts_dir() / "history.json"

    def prompt_path(self, version: str) -> Path:
        return self.prompts_dir() / f"{version}.md"

    def list_prompt_versions(self) -> list[str]:
        d = self.prompts_dir()
        if not d.exists():
            return []
        return sorted(
            p.stem for p in d.glob("v*.md")
        )

    def latest_prompt_version(self) -> str:
        versions = self.list_prompt_versions()
        return versions[-1] if versions else "v1"


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert nested dicts to dataclasses."""
    if data is None:
        return cls()
    kwargs = {}
    import dataclasses
    for f in dataclasses.fields(cls):
        val = data.get(f.name)
        if val is None:
            kwargs[f.name] = f.default if f.default is not dataclasses.MISSING else f.default_factory()
            continue
        if dataclasses.is_dataclass(f.type) or (isinstance(f.type, str) and f.type in globals()):
            field_cls = f.type if dataclasses.is_dataclass(f.type) else globals()[f.type]
            kwargs[f.name] = _dict_to_dataclass(field_cls, val) if isinstance(val, dict) else val
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_scene(name: str, scenes_dir: Optional[Path] = None) -> SceneConfig:
    """Load and validate a scene config from scenes/{name}.yaml."""
    base = scenes_dir or SCENES_DIR
    path = base / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Scene config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid scene config: {path}")

    cfg = SceneConfig(name=raw.get("name", name))
    cfg.display_name = raw.get("display_name", name)
    cfg.version = str(raw.get("version", "1.0"))

    cfg.data = _load_data(raw.get("data", {}))
    cfg.inference = _load_inference(raw.get("inference", {}))
    cfg.target = _load_target(raw.get("target", {}))
    cfg.iteration = _load_iteration(raw.get("iteration", {}))
    cfg.token_budget = _load_token_budget(raw.get("token_budget", {}))
    cfg.alerts = _load_alerts(raw.get("alerts", {}))

    # Resolve ${ENV_VAR} placeholders and fall back to env var if key not set
    cfg.inference.api_key = _resolve_env(cfg.inference.api_key) or os.environ.get("DASHSCOPE_API_KEY", "")

    return cfg


def _load_data(d: dict) -> DataConfig:
    return DataConfig(
        mode=d.get("mode", "local"),
        positive_dir=d.get("positive_dir", ""),
        negative_dir=d.get("negative_dir", ""),
        dataset=d.get("dataset"),
    )


def _load_inference(d: dict) -> InferenceConfig:
    return InferenceConfig(
        engine=d.get("engine", "dashscope"),
        model=d.get("model", "qwen3.7-plus"),
        workers=int(d.get("workers", 5)),
        timeout_per_sample=int(d.get("timeout_per_sample", 60)),
        fps=int(d.get("fps", 2)),
        thinking_budget=int(d.get("thinking_budget", 512)),
        max_retries=int(d.get("max_retries", 3)),
        api_key=d.get("api_key", ""),
        api_base=d.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        vllm_url=d.get("vllm_url", "http://localhost:8000/v1/chat/completions"),
    )


def _load_target(d: dict) -> TargetConfig:
    return TargetConfig(
        precision=float(d.get("precision", 80.0)),
        recall=float(d.get("recall", 75.0)),
        accuracy=float(d.get("accuracy", 0.0)),
    )


def _load_iteration(d: dict) -> IterationConfig:
    es = d.get("early_stop", {})
    return IterationConfig(
        max_rounds=int(d.get("max_rounds", 3)),
        eval_sample_size=int(d.get("eval_sample_size", 200)),
        full_eval_on_final=bool(d.get("full_eval_on_final", True)),
        improve_topk=int(d.get("improve_topk", 3)),
        early_stop=EarlyStop(
            no_improvement_rounds=int(es.get("no_improvement_rounds", 2)),
            token_budget_exceeded=bool(es.get("token_budget_exceeded", True)),
        ),
    )


def _load_token_budget(d: dict) -> TokenBudget:
    return TokenBudget(
        per_sample_max=int(d.get("per_sample_max", 3000)),
        total_per_round_max=int(d.get("total_per_round_max", 200_000)),
        total_pipeline_max=int(d.get("total_pipeline_max", 600_000)),
        alert_threshold=float(d.get("alert_threshold", 0.8)),
    )


def _load_alerts(d: dict) -> AlertConfig:
    return AlertConfig(
        email=_resolve_env(d.get("email")),
    )


def _resolve_env(value: Optional[str]) -> Optional[str]:
    """Expand ${VAR_NAME} placeholders using os.environ. Returns None if value is None."""
    if not value:
        return value
    import re
    def _sub(m):
        return os.environ.get(m.group(1), m.group(0))
    return re.sub(r"\$\{([^}]+)\}", _sub, value)
