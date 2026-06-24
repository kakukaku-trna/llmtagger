"""LLMTagger – unified entry point.

Usage:
    python run.py --scene blind_curve [--version v1] [--sample 10] [--iterate] [--max-rounds 3]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from the llmtagger/ directory without installing
sys.path.insert(0, str(Path(__file__).parent))

# Auto-load .env from the project root (keys already in env take precedence)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from pipeline.config import load_scene
from pipeline.data.local_loader import load_samples
from pipeline.evaluate.metrics import (
    best_version,
    calc_metrics,
    load_all_versions,
    print_metrics_report,
    print_versions_table,
    save_version_metrics,
)
from pipeline.improve.failure_analyzer import analyze_failures
from pipeline.improve.prompt_initializer import generate_initial_prompt
from pipeline.improve.topk_modifier import (
    generate_next_prompt,
    next_version_name,
    save_new_version,
)
from pipeline.inference.batch_runner import BatchRunner
from pipeline.monitor.alert import AlertLevel, emit_alert
from pipeline.monitor.raw_log import RawLogWriter
from pipeline.monitor.token_tracker import TokenTracker


# ─────────────────────────────────────────────────────────────
def load_prompt(scene_cfg, version: str) -> str:
    p = scene_cfg.prompt_path(version)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


# 飞书告警 webhook — 硬编码，无需配置
# 原始响应日志根目录
LOGS_DIR = Path(__file__).parent / "logs"
FEISHU_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/2e1e6058-0122-442b-9dfb-6e97506c8014"
FEISHU_MENTION_ID = os.environ.get("FEISHU_MENTION_ID", "all")


def _alert(scene_cfg, level, event, detail, **kw):
    """Wrapper that always injects feishu_webhook, mention_id and email from scene config."""
    emit_alert(
        level,
        event,
        detail,
        feishu_webhook=FEISHU_WEBHOOK,
        email=scene_cfg.alerts.email,
        mention_id=FEISHU_MENTION_ID,
        **kw,
    )


def _check_budget(tracker: TokenTracker, scene_cfg, scene: str, round_n: int) -> bool:
    """Return True if pipeline budget exceeded (should stop)."""
    level = tracker.check_pipeline_budget()
    if level.value == "exceeded":
        _alert(
            scene_cfg,
            AlertLevel.ERROR,
            "Token 预算耗尽",
            "已超出 pipeline 最大 token 上限，停止迭代",
            scene=scene,
            tokens_used=tracker.pipeline_total,
            token_budget=scene_cfg.token_budget.total_pipeline_max,
        )
        return True
    if level.value == "warning":
        _alert(
            scene_cfg,
            AlertLevel.WARNING,
            "Token 预算告警",
            f"已超过 {scene_cfg.token_budget.alert_threshold * 100:.0f}% 预算阈值",
            scene=scene,
            tokens_used=tracker.pipeline_total,
            token_budget=scene_cfg.token_budget.total_pipeline_max,
        )
    return False


def _target_desc(scene_cfg) -> str:
    """Human-readable target string."""
    t = scene_cfg.target
    parts = [
        f"Accuracy>={t.accuracy}%" if t.accuracy > 0 else None,
        f"Recall>={t.recall}%",
        f"Precision>={t.precision}%",
    ]
    return "  ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────
def run_init_prompt(args: argparse.Namespace) -> None:
    """Generate an initial prompt for a scene via LLM."""
    scene_cfg = load_scene(args.scene)

    version = args.version or "v1"
    prompt_path = scene_cfg.prompt_path(version)

    if prompt_path.exists() and not args.force:
        print(f"  ✗ {prompt_path} 已存在，使用 --force 覆盖")
        return

    desc = args.description or scene_cfg.display_name or args.scene
    print(f"\n  生成初始 Prompt: 场景={args.scene}  版本={version}")
    print(f"  描述: {desc}\n")

    prompt_text = generate_initial_prompt(
        scene_cfg=scene_cfg,
        description=args.description or "",
    )

    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")
    print(f"  ✓ 已保存 → {prompt_path}\n")
    print("  可以运行以下命令开始评估:")
    print(f"    python3 run.py --scene {args.scene} --sample 10")
    print(f"  或开启迭代优化:")
    print(f"    python3 run.py --scene {args.scene} --iterate")


# ─────────────────────────────────────────────────────────────
def run_single(args: argparse.Namespace) -> None:
    """Run a single evaluation round without iteration."""
    scene_cfg = load_scene(args.scene)
    version = args.version or scene_cfg.latest_prompt_version()
    prompt = load_prompt(scene_cfg, version)
    samples = load_samples(scene_cfg.data, sample_size=args.sample)

    print(f"\n LLMTagger  场景={args.scene}  版本={version}  样本数={len(samples)}")
    print(
        f"  推理引擎: {scene_cfg.inference.engine}  模型: {scene_cfg.inference.model}\n"
    )

    tracker = TokenTracker(scene_cfg.token_budget)
    log_path = RawLogWriter.make_log_path(LOGS_DIR, args.scene)
    log_writer = RawLogWriter(log_path)
    print(f"  原始响应日志: {log_path}\n")
    runner = BatchRunner(
        config=scene_cfg.inference,
        tracker=tracker,
        feishu_webhook=FEISHU_WEBHOOK,
        alert_email=scene_cfg.alerts.email,
        log_writer=log_writer,
    )

    results = runner.run(samples, prompt, scene=args.scene, version=version, round_n=1)

    metrics = calc_metrics(results)
    save_version_metrics(args.scene, version, metrics, scene_cfg.metrics_path())
    print_metrics_report(metrics, version=version, scene=args.scene)

    tracker.end_round()
    print(tracker.summary_line(args.scene, round_n=1))

    all_versions = load_all_versions(scene_cfg.metrics_path())
    if len(all_versions) > 1:
        print_versions_table(all_versions)


def _print_best_prompt(scene_cfg, version: str, metrics) -> None:
    """Print the best-performing prompt when the iteration target is not reached."""
    sep = "═" * 58
    acc_str = (
        f"  Accuracy>={scene_cfg.target.accuracy}%"
        if scene_cfg.target.accuracy > 0
        else ""
    )
    print(f"\n{sep}")
    print(f"  未达到目标 ({_target_desc(scene_cfg)})")
    print(
        f"  最佳版本: {version}"
        f"  —  Accuracy={metrics.accuracy:.1f}%  Recall={metrics.recall:.1f}%"
        f"  Precision={metrics.precision:.1f}%"
    )
    print(f"{sep}")
    prompt_path = scene_cfg.prompt_path(version)
    if prompt_path.exists():
        print(f"\n  >>> 推荐使用 {version} (路径: {prompt_path})\n")
        print(prompt_path.read_text(encoding="utf-8"))
    print(sep)


# ─────────────────────────────────────────────────────────────
def run_iterate(args: argparse.Namespace) -> None:
    """Run iterative prompt improvement loop."""
    scene_cfg = load_scene(args.scene)
    version = args.version or scene_cfg.latest_prompt_version()
    prompt = load_prompt(scene_cfg, version)
    max_rounds = args.max_rounds or scene_cfg.iteration.max_rounds
    # CLI --sample takes precedence; fall back to yaml eval_sample_size
    sample_size = args.sample or scene_cfg.iteration.eval_sample_size
    samples = load_samples(scene_cfg.data, sample_size=sample_size)

    print(
        f"\n LLMTagger (迭代模式)  场景={args.scene}  版本={version}  最大轮次={max_rounds}"
    )
    print(f"  样本数={len(samples)}  目标: {_target_desc(scene_cfg)}\n")

    tracker = TokenTracker(scene_cfg.token_budget)
    log_path = RawLogWriter.make_log_path(LOGS_DIR, args.scene)
    log_writer = RawLogWriter(log_path)
    print(f"  原始响应日志: {log_path}\n")
    runner = BatchRunner(
        config=scene_cfg.inference,
        tracker=tracker,
        feishu_webhook=FEISHU_WEBHOOK,
        alert_email=scene_cfg.alerts.email,
        log_writer=log_writer,
    )

    no_improvement = 0
    best_score = -1.0      # recall + precision
    best_version_name = version
    current_version = version
    current_prompt = prompt
    target_reached = False

    for round_n in range(1, max_rounds + 1):
        print(f"\n{'─' * 50}")
        print(f"  第 {round_n}/{max_rounds} 轮  版本={current_version}")
        print(f"{'─' * 50}")

        # Clear cache per round so re-evaluation uses current prompt
        runner._cache = {}

        results = runner.run(
            samples,
            current_prompt,
            scene=args.scene,
            version=current_version,
            round_n=round_n,
        )
        metrics = calc_metrics(results)
        save_version_metrics(
            args.scene, current_version, metrics, scene_cfg.metrics_path()
        )
        print_metrics_report(metrics, version=current_version, scene=args.scene)

        tracker.end_round()
        print(tracker.summary_line(args.scene, round_n=round_n))

        # Budget check
        if _check_budget(tracker, scene_cfg, args.scene, round_n):
            break

        # Target reached
        if metrics.meets_target(
            scene_cfg.target.precision,
            scene_cfg.target.recall,
            scene_cfg.target.accuracy,
        ):
            target_reached = True
            _alert(
                scene_cfg,
                AlertLevel.INFO,
                "达到评估目标",
                f"Accuracy={metrics.accuracy:.1f}%  Recall={metrics.recall:.1f}%  Precision={metrics.precision:.1f}%",
                scene=args.scene,
                progress=f"{round_n}/{max_rounds}",
            )
            break

        # Track best version by recall + precision (primary optimisation goal)
        score = metrics.opt_score()
        if score > best_score:
            best_score = score
            best_version_name = current_version
            no_improvement = 0
        else:
            no_improvement += 1
            print(
                f"  无改善轮次: {no_improvement}/{scene_cfg.iteration.early_stop.no_improvement_rounds}"
            )
            if no_improvement >= scene_cfg.iteration.early_stop.no_improvement_rounds:
                print("  触发早停：连续无改善，停止迭代")
                break

        # Last round – don't generate next version
        if round_n == max_rounds:
            break

        # Generate improved prompt
        print(
            f"\n  分析失败案例，生成 Top-{scene_cfg.iteration.improve_topk} 改进 Prompt…"
        )
        failures = analyze_failures(results)
        print(failures.summary())

        try:
            new_prompt = generate_next_prompt(
                current_prompt=current_prompt,
                failures=failures,
                metrics=metrics,
                current_version=current_version,
                topk=scene_cfg.iteration.improve_topk,
                config=scene_cfg.inference,
            )
            next_ver = next_version_name(current_version)
            save_new_version(
                scene_cfg=scene_cfg,
                new_prompt=new_prompt,
                from_version=current_version,
                new_version=next_ver,
                failures=failures,
                metrics_delta={
                    "accuracy": metrics.accuracy,
                    "recall": metrics.recall,
                    "precision": metrics.precision,
                    "f1": metrics.f1,
                },
            )
            print(f"  新版本 {next_ver} 已保存 → {scene_cfg.prompt_path(next_ver)}")
            current_version = next_ver
            current_prompt = new_prompt
        except Exception as exc:
            print(f"  ⚠  Prompt 生成失败: {exc}，保持当前版本继续")

    # Final comparison table
    all_versions = load_all_versions(scene_cfg.metrics_path())
    if len(all_versions) > 1:
        if target_reached:
            print_versions_table(all_versions)
        else:
            best_ver, best_m = best_version(all_versions)
            print_versions_table(all_versions, highlight=best_ver)
            _print_best_prompt(scene_cfg, best_ver, best_m)


# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLMTagger – multimodal video annotation pipeline"
    )
    parser.add_argument("--scene", required=True, help="Scene name (e.g. blind_curve)")
    parser.add_argument(
        "--version", default=None, help="Prompt version to use (e.g. v1)"
    )
    parser.add_argument("--sample", type=int, default=None, help="Max samples to use")
    parser.add_argument(
        "--iterate", action="store_true", help="Enable prompt iteration loop"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None, help="Max iteration rounds"
    )
    parser.add_argument(
        "--init-prompt", action="store_true", help="Generate initial prompt via LLM"
    )
    parser.add_argument(
        "--description", default=None, help="Scene description for --init-prompt"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing prompt (for --init-prompt)"
    )

    args = parser.parse_args()

    if args.init_prompt:
        run_init_prompt(args)
    elif args.iterate:
        run_iterate(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
