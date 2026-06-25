"""LLMTagger 统一入口脚本。

这个文件负责把 scene 配置、prompt、数据加载、模型推理、评估统计和
caption 导出串成一个可直接执行的 CLI 入口。脚本启动时会自动读取项目
根目录下的 `.env`，再根据 `scenes/{scene}.yaml` 中的配置选择模型、
数据目录和运行模式。

目前主要支持三类用法：
1. 单轮评估：读取正负样本，跑一次推理并输出 precision / recall / F1。
2. 迭代优化：在评估结果基础上分析失败案例，自动生成下一版 prompt。
3. Caption 导出：对显式传入的视频或帧目录生成纯文本描述，不走二分类评估。

常用示例：
    python run.py --scene blind_curve --sample 10
    python run.py --scene blind_curve --iterate --max-rounds 3
    python run.py --scene multicam_caption --caption --input /path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
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
from pipeline.data.local_loader import load_samples, load_unlabeled_samples
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
from pipeline.inference.batch_runner import BatchRunner, build_client
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
CAPTIONS_DIR = Path(__file__).parent / "captions"
FEISHU_WEBHOOK = (
    "https://open.feishu.cn/open-apis/bot/v2/hook/2e1e6058-0122-442b-9dfb-6e97506c8014"
)
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
    samples = load_samples(
        scene_cfg.data, sample_size=args.sample, camera_suffix=args.camera
    )

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


def _caption_output_dir(scene: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CAPTIONS_DIR / scene / f"caption_{now}"


def _result_uuid(video_path: str) -> str:
    path = Path(video_path)
    if path.is_dir():
        return path.name
    return path.parent.name or path.stem


def _caption_len(text: str) -> int:
    return len(text.strip())


def _caption_valid(result, scene_cfg) -> bool:
    if result.status != "success" or not result.result.strip():
        return False
    min_chars = scene_cfg.caption.min_chars
    max_chars = scene_cfg.caption.max_chars
    text_len = _caption_len(result.result)
    if min_chars and text_len < min_chars:
        return False
    if max_chars and text_len > max_chars:
        return False
    return True


def _build_caption_retry_prompt(
    base_prompt: str,
    *,
    previous_result: str,
    previous_status: str,
    min_chars: int,
    max_chars: int,
) -> str:
    text_len = _caption_len(previous_result) if previous_result else 0
    note = (
        f"上一版输出状态为 {previous_status}，长度为 {text_len} 字，"
        f"未满足 {min_chars}-{max_chars} 字的要求。"
    )
    prev = previous_result or "上一版没有产出有效描述。"
    return (
        f"{base_prompt}\n\n"
        f"补充要求：{note}"
        "请在不编造视频中不存在内容的前提下重写整段描述，优先补充时间顺序、本车速度变化、"
        "交互车辆位置与状态、交通信号灯、车道与路况细节。"
        f"最终输出必须严格在 {min_chars}-{max_chars} 字之间，且仍然只输出一段纯中文文本。\n"
        f"上一版输出：{prev}"
    )


def _enforce_caption_constraints(
    runner: BatchRunner,
    scene_cfg,
    samples,
    prompt: str,
    *,
    scene: str,
    version: str,
    results,
):
    if scene_cfg.caption.max_attempts <= 1:
        return results

    ordered_paths = [sample.video_path for sample in samples]
    result_map = {result.video_path: result for result in results}
    client = build_client(scene_cfg.inference)

    for attempt in range(2, scene_cfg.caption.max_attempts + 1):
        pending = [
            sample
            for sample in samples
            if not _caption_valid(result_map[sample.video_path], scene_cfg)
        ]
        if not pending:
            break

        print(
            f"\n  Caption 长度校正: 第 {attempt}/{scene_cfg.caption.max_attempts} 次尝试"
            f"  待重试={len(pending)}"
        )

        previous_map = {sample.video_path: result_map[sample.video_path] for sample in pending}

        def _retry_infer(sample):
            previous = previous_map[sample.video_path]
            retry_prompt = _build_caption_retry_prompt(
                prompt,
                previous_result=previous.result,
                previous_status=previous.status,
                min_chars=scene_cfg.caption.min_chars,
                max_chars=scene_cfg.caption.max_chars,
            )
            return client.infer(sample, retry_prompt)

        retried = runner.run(
            pending,
            prompt,
            infer_fn=_retry_infer,
            scene=scene,
            version=f"{version}_retry{attempt}",
            round_n=attempt,
        )
        for result in retried:
            result_map[result.video_path] = result

    return [result_map[path] for path in ordered_paths]


def _write_caption_results(
    output_dir: Path,
    *,
    prompt: str,
    results,
    scene: str,
    version: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_used.md").write_text(prompt, encoding="utf-8")

    summary_path = output_dir / "results.jsonl"
    used_names: dict[str, int] = {}

    with summary_path.open("w", encoding="utf-8") as f:
        for result in results:
            base_name = _result_uuid(result.video_path)
            count = used_names.get(base_name, 0)
            used_names[base_name] = count + 1
            file_name = f"{base_name}_{count + 1}" if count else base_name

            caption_file = ""
            if result.status == "success" and result.result:
                caption_file = f"{file_name}.txt"
                (output_dir / caption_file).write_text(result.result, encoding="utf-8")

            record = {
                "scene": scene,
                "version": version,
                "uuid": base_name,
                "video_path": result.video_path,
                "status": result.status,
                "caption_file": caption_file,
                "caption": result.result,
                "error": result.error,
                "elapsed": result.elapsed,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return summary_path


def run_caption(args: argparse.Namespace) -> None:
    """Run caption generation for explicit video or frame inputs."""
    if not args.input:
        raise ValueError("--caption 模式必须提供 --input")

    scene_cfg = load_scene(args.scene)
    version = args.version or scene_cfg.latest_prompt_version()
    prompt = load_prompt(scene_cfg, version)
    samples = load_unlabeled_samples(
        args.input,
        sample_size=args.sample,
        camera_suffix=args.camera,
    )

    if not samples:
        raise ValueError("未找到可用的视频或帧目录")

    print(f"\n LLMTagger (Caption 模式)  场景={args.scene}  版本={version}  样本数={len(samples)}")
    print(
        f"  推理引擎: {scene_cfg.inference.engine}  模型: {scene_cfg.inference.model}"
        f"  输出模式: {scene_cfg.inference.response_mode}\n"
    )

    tracker = TokenTracker(scene_cfg.token_budget)
    log_path = RawLogWriter.make_log_path(LOGS_DIR, args.scene)
    log_writer = RawLogWriter(log_path)
    runner = BatchRunner(
        config=scene_cfg.inference,
        tracker=tracker,
        feishu_webhook=FEISHU_WEBHOOK,
        alert_email=scene_cfg.alerts.email,
        log_writer=log_writer,
    )

    results = runner.run(samples, prompt, scene=args.scene, version=version, round_n=1)
    results = _enforce_caption_constraints(
        runner,
        scene_cfg,
        samples,
        prompt,
        scene=args.scene,
        version=version,
        results=results,
    )
    tracker.end_round()

    output_dir = _caption_output_dir(args.scene, args.output_dir)
    summary_path = _write_caption_results(
        output_dir,
        prompt=prompt,
        results=results,
        scene=args.scene,
        version=version,
    )

    success = sum(1 for result in results if result.status == "success")
    print(f"\n  成功: {success}/{len(results)}")
    print(f"  Caption 输出目录: {output_dir}")
    print(f"  汇总文件: {summary_path}")
    print(f"  原始响应日志: {log_path}")
    print(tracker.summary_line(args.scene, round_n=1))


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
    samples = load_samples(
        scene_cfg.data, sample_size=sample_size, camera_suffix=args.camera
    )

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
    best_score = -1.0  # recall + precision
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
    parser.add_argument(
        "--caption", action="store_true", help="Generate free-form captions instead of evaluation"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="One or more input video files or directories for --caption",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for caption outputs (default: captions/{scene}/caption_*/)",
    )

    parser.add_argument(
        "--camera",
        default=None,
        help="Only process files ending with this camera name (suffix match)",
    )

    args = parser.parse_args()

    if args.caption and (args.iterate or args.init_prompt):
        raise ValueError("--caption 不能与 --iterate 或 --init-prompt 同时使用")

    if args.init_prompt:
        run_init_prompt(args)
    elif args.caption:
        run_caption(args)
    elif args.iterate:
        run_iterate(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
