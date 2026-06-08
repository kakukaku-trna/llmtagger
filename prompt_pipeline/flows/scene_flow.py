"""Single-scene evaluation and iteration flow.

Uses Prefect @flow / @task decorators when Prefect is installed.
Falls back to plain function calls otherwise, so the pipeline works
without Prefect in CI or local dev environments.

Usage (with Prefect):
    from flows.scene_flow import run_scene_flow
    result = run_scene_flow(scene="blind_curve", sample_size=50, iterate=True)

Usage (without Prefect / standalone):
    python flows/scene_flow.py --scene blind_curve --sample 50
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Allow running standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Optional Prefect import ──────────────────────────────────
try:
    from prefect import flow, task
    _PREFECT = True
except ImportError:
    _PREFECT = False

    def flow(fn=None, **kw):
        return fn if fn else (lambda f: f)

    def task(fn=None, **kw):
        return fn if fn else (lambda f: f)

# ─────────────────────────────────────────────────────────────
from pipeline.config import load_scene
from pipeline.data.local_loader import load_samples
from pipeline.evaluate.metrics import (
    Metrics,
    calc_metrics,
    print_metrics_report,
    save_version_metrics,
)
from pipeline.improve.failure_analyzer import analyze_failures
from pipeline.improve.topk_modifier import (
    generate_next_prompt,
    next_version_name,
    save_new_version,
)
from pipeline.inference.batch_runner import BatchRunner
from pipeline.monitor.alert import AlertLevel, emit_alert
from pipeline.monitor.token_tracker import TokenTracker


# ─────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────

@task(name="load-samples", retries=1)
def task_load_samples(scene_name: str, sample_size: Optional[int]):
    cfg = load_scene(scene_name)
    return load_samples(cfg.data, sample_size=sample_size), cfg


@task(name="run-inference")
def task_run_inference(scene_name: str, version: str, sample_size: Optional[int]):
    cfg = load_scene(scene_name)
    samples = load_samples(cfg.data, sample_size=sample_size)
    prompt_path = cfg.prompt_path(version)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")

    tracker = TokenTracker(cfg.token_budget)
    runner = BatchRunner(config=cfg.inference, tracker=tracker)
    results = runner.run(samples, prompt)
    return results, tracker


@task(name="evaluate")
def task_evaluate(scene_name: str, version: str, results, tracker: TokenTracker):
    cfg = load_scene(scene_name)
    metrics = calc_metrics(results)
    save_version_metrics(scene_name, version, metrics, cfg.metrics_path())
    print_metrics_report(metrics, version=version, scene=scene_name)
    return metrics


@task(name="improve-prompt")
def task_improve_prompt(scene_name: str, current_version: str, results, metrics: Metrics):
    cfg = load_scene(scene_name)
    failures = analyze_failures(results)
    prompt = cfg.prompt_path(current_version).read_text(encoding="utf-8")
    new_prompt = generate_next_prompt(
        current_prompt=prompt,
        failures=failures,
        metrics=metrics,
        current_version=current_version,
        topk=cfg.iteration.improve_topk,
        config=cfg.inference,
    )
    next_ver = next_version_name(current_version)
    save_new_version(
        scene_cfg=cfg,
        new_prompt=new_prompt,
        from_version=current_version,
        new_version=next_ver,
        failures=failures,
        metrics_delta={"f1": metrics.f1},
    )
    return next_ver


# ─────────────────────────────────────────────────────────────
# Flow
# ─────────────────────────────────────────────────────────────

@flow(name="scene-eval-flow", log_prints=True)
def run_scene_flow(
    scene: str,
    version: Optional[str] = None,
    sample_size: Optional[int] = None,
    iterate: bool = False,
    max_rounds: int = 3,
) -> Metrics:
    """Evaluate (and optionally iterate) a single scene's prompt.

    Returns the final Metrics from the last evaluation round.
    """
    cfg = load_scene(scene)
    current_version = version or cfg.latest_prompt_version()
    no_improvement = 0
    best_f1 = -1.0
    last_metrics = None

    for round_n in range(1, max_rounds + 1):
        print(f"\n── 第 {round_n}/{max_rounds} 轮  版本={current_version} ──")

        results, tracker = task_run_inference(scene, current_version, sample_size)
        metrics = task_evaluate(scene, current_version, results, tracker)
        last_metrics = metrics

        # Budget check
        if tracker.check_pipeline_budget().value == "exceeded":
            emit_alert(AlertLevel.ERROR, "Token 预算耗尽", "停止迭代", scene=scene)
            break

        # Target reached
        if metrics.meets_target(cfg.target.precision, cfg.target.recall):
            emit_alert(AlertLevel.INFO, "达到评估目标",
                       f"P={metrics.precision:.1f}%  R={metrics.recall:.1f}%",
                       scene=scene, progress=f"{round_n}/{max_rounds}")
            break

        if not iterate or round_n == max_rounds:
            break

        # Early stop
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= cfg.iteration.early_stop.no_improvement_rounds:
                print("  早停：连续无改善")
                break

        current_version = task_improve_prompt(scene, current_version, results, metrics)

    return last_metrics


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--iterate", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    run_scene_flow(
        scene=args.scene,
        version=args.version,
        sample_size=args.sample,
        iterate=args.iterate,
        max_rounds=args.max_rounds,
    )
