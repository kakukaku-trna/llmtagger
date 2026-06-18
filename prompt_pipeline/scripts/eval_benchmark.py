#!/usr/bin/env python3
"""Multi-label eval script for eval_benchmark.csv.

Usage:
    cd /home/huajiang.sun/llmtagger/prompt_pipeline

    # DashScope (qwen3.7-plus)
    python3 scripts/eval_benchmark.py --sample 3
    python3 scripts/eval_benchmark.py --version v2 --workers 5

    # vLLM (Qwen2.5-VL-7B, local)
    python3 scripts/eval_benchmark.py --engine vllm --version v3 --workers 10 --sample 3
    python3 scripts/eval_benchmark.py --engine vllm --version v3 --workers 10

    # vLLM (Qwen3.5-9B, local)
    python3 scripts/eval_benchmark.py --engine qwen35 --version v9 --workers 4 --sample 3
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from pipeline.config import InferenceConfig
from pipeline.data.local_loader import Sample
from pipeline.inference.batch_runner import BatchRunner
from pipeline.inference.dashscope_client import (
    DashScopeClient,
    InferResult,
    _encode_video,
    _find_frames,
)
from pipeline.monitor.token_tracker import TokenTracker

# ── paths ─────────────────────────────────────────────────────────────────────
CSV_PATH   = Path("/home/huajiang.sun/data/eval_benchmark.csv")
VIDEOS_DIR = Path("/home/huajiang.sun/data/eval_videos")
PROMPTS_DIR = ROOT / "prompts/eval_benchmark"

# ── labels ────────────────────────────────────────────────────────────────────
BINARY_LABELS = [
    "公交车道", "非机动车道", "路面积水", "双向单车道",
    "前方左转待转区", "逆光", "炫光", "强反射", "光影", "道路积雪",
]
ALL_LABELS = ["天气", "时段"] + BINARY_LABELS

# ── qwen3.7-plus pricing (DashScope, 2025) ────────────────────────────────────
INPUT_PRICE_PER_1K  = 0.002   # ¥/1K tokens
OUTPUT_PRICE_PER_1K = 0.008   # ¥/1K tokens


# ─────────────────────────────────────────────────────────────────────────────
# Multi-label client: stores the full JSON string in result instead of just
# the "result" field that the default _parse_response() extracts.
# ─────────────────────────────────────────────────────────────────────────────
class MultiLabelClient(DashScopeClient):
    def infer(self, sample: Sample, prompt: str) -> InferResult:
        t0 = datetime.datetime.now()
        out = InferResult(
            video_path=sample.video_path,
            label=sample.label,
            result="",
            reason="",
            status="failed",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            elapsed=0.0,
        )
        try:
            frame_paths = _find_frames(sample.video_path)
            if frame_paths:
                raw, pt, ct = self._call_api_with_frames(frame_paths, prompt)
            else:
                b64 = _encode_video(sample.video_path)
                raw, pt, ct = self._call_api(b64, prompt)

            # Strip markdown fences, extract JSON object
            text = raw.strip()
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                out.result = m.group(0)        # full JSON string
                out.reason = parsed.get("reason", "")
                out.status = "success"
            else:
                out.status = "parse_error"
                out.error = f"No JSON found in response: {raw[:300]}"

            out.prompt_tokens = pt
            out.completion_tokens = ct
            out.total_tokens = pt + ct
        except Exception as e:
            out.error = str(e)

        out.elapsed = (datetime.datetime.now() - t0).total_seconds()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# vLLM helpers: two modes
#   jpeg: extract N JPEG frames → data:video/jpeg;base64,{f1},{f2},...
#         Bypasses vLLM 0.8.5 MP4 bug (assert i==num_frames fails for <16s clips)
#   mp4:  compress to 480×270 2fps MP4, pad to 16s → data:video/mp4;base64,...
#         Mirrors batch_infer_full.py; needs tpad to guarantee exactly 32 frames
# ─────────────────────────────────────────────────────────────────────────────
VLLM_W, VLLM_H    = 480, 270
VLLM_N_FRAMES     = 16    # jpeg mode: extract this many frames
VLLM_FPS          = 2.0   # mp4 mode: output fps
VLLM_DURATION     = 16.0  # mp4 mode: target duration (→ 32 frames at 2fps)
_VLLM_TMP_DIR     = "/tmp/eval_vllm_tmp"
QWEN35_MODEL_PATH = "/data-algorithm/model_cache/Qwen/Qwen3.5-9B"
QWEN35_VLLM_URL   = "http://localhost:8001/v1/chat/completions"

import os as _os
_os.makedirs(_VLLM_TMP_DIR, exist_ok=True)


def _extract_jpeg_frames_b64(video_path: str, n_frames: int = VLLM_N_FRAMES) -> list[str]:
    """Extract n evenly-spaced JPEG frames at 480×270 → list of base64 strings."""
    import subprocess, base64, threading
    tid = threading.get_ident()
    out_pat = f"{_VLLM_TMP_DIR}/f_{tid}_%03d.jpg"
    dur_r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=10,
    )
    try:
        duration = float(dur_r.stdout.strip())
    except Exception:
        duration = 10.0
    fps_val = n_frames / max(duration, 0.1)
    vf = (f"scale={VLLM_W}:{VLLM_H}:force_original_aspect_ratio=decrease,"
          f"pad={VLLM_W}:{VLLM_H}:(ow-iw)/2:(oh-ih)/2,fps={fps_val:.4f}")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", vf,
           "-vframes", str(n_frames), "-q:v", "4", out_pat]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    result = []
    for i in range(1, n_frames + 1):
        p = f"{_VLLM_TMP_DIR}/f_{tid}_{i:03d}.jpg"
        if _os.path.exists(p):
            with open(p, "rb") as f:
                result.append(base64.b64encode(f.read()).decode())
            _os.remove(p)
    if not result:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
    return result


def _compress_video_mp4_b64(video_path: str) -> str:
    """Compress to 480×270 2fps MP4, pad to 16s (→ exactly 32 frames).
    Mirrors batch_infer_full.py; the tpad ensures clips shorter than 16s
    are padded with the last frame so vLLM's num_frames==32 assertion holds.
    """
    import subprocess, base64, threading
    tid = threading.get_ident()
    tmp = f"{_VLLM_TMP_DIR}/clip_{tid}.mp4"
    # tpad=stop_duration=30 pads up to 30s of cloned last-frame after video ends;
    # -t 16 caps output at 16s → always exactly 32 frames at 2fps.
    vf = (f"scale={VLLM_W}:{VLLM_H}:force_original_aspect_ratio=decrease,"
          f"pad={VLLM_W}:{VLLM_H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,"
          f"fps={VLLM_FPS},"
          f"tpad=stop_duration=30:stop_mode=clone")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-t", str(VLLM_DURATION),
        "-c:v", "libx264", "-crf", "30", "-preset", "fast", "-an",
        tmp,
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    if r.returncode != 0 or not _os.path.exists(tmp):
        raise RuntimeError(f"ffmpeg compress failed for {video_path}")
    with open(tmp, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    _os.remove(tmp)
    return b64


class MultiLabelVLLMClient:
    def __init__(self, config: InferenceConfig, mode: str = "jpeg"):
        self._config = config
        self._mode = mode   # "jpeg" or "mp4"

    def infer(self, sample: Sample, prompt: str) -> InferResult:
        import datetime
        import time
        import requests as _req
        t0 = datetime.datetime.now()
        out = InferResult(
            video_path=sample.video_path,
            label=sample.label,
            result="",
            reason="",
            status="failed",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            elapsed=0.0,
        )
        try:
            cfg = self._config
            is_qwen35 = "Qwen3.5" in str(cfg.model)
            if self._mode == "mp4":
                b64 = _compress_video_mp4_b64(sample.video_path)
                video_url = f"data:video/mp4;base64,{b64}"
                content = [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url, "fps": cfg.fps},
                    },
                    {"type": "text", "text": prompt},
                ]
            else:
                frames = _extract_jpeg_frames_b64(sample.video_path, VLLM_N_FRAMES)
                if is_qwen35:
                    content = [{
                        "type": "text",
                        "text": f"以下是从同一段行车视频中提取的 {len(frames)} 帧图像，请综合分析它们来判断。",
                    }]
                    content.extend(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                        }
                        for frame_b64 in frames
                    )
                    content.append({"type": "text", "text": prompt})
                else:
                    video_url = f"data:video/jpeg;base64,{','.join(frames)}"
                    content = [
                        {"type": "video_url", "video_url": {"url": video_url}},
                        {"type": "text", "text": prompt},
                    ]

            payload = {
                "model": cfg.model,
                "temperature": 0.0,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": content}],
            }
            if "Qwen3.5" in str(cfg.model):
                payload["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": 20,
                    "mm_processor_kwargs": {"fps": 1, "do_sample_frames": True},
                }
            last_err = ""
            raw = ""
            pt = ct = 0
            for attempt in range(1, cfg.max_retries + 1):
                try:
                    resp = _req.post(cfg.vllm_url, json=payload,
                                     timeout=cfg.timeout_per_sample)
                    if (
                        resp.status_code == 422
                        and "extra_body" in payload
                        and "extra_body" in resp.text
                    ):
                        payload.pop("extra_body", None)
                        last_err = f"server rejected extra_body: {resp.text[:200]}"
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    raw = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    pt = usage.get("prompt_tokens", 0)
                    ct = usage.get("completion_tokens", 0)
                    break
                except Exception as e:
                    if 'resp' in locals():
                        last_err = f"{e} | body={resp.text[:300]}"
                    else:
                        last_err = str(e)
                    if attempt < cfg.max_retries:
                        time.sleep(2)
            else:
                raise RuntimeError(f"vLLM failed after {cfg.max_retries} retries: {last_err}")

            text = raw.strip()
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                out.result = m.group(0)
                out.reason = parsed.get("reason", "")
                out.status = "success"
            else:
                out.status = "parse_error"
                out.error = f"No JSON found: {raw[:300]}"

            out.prompt_tokens = pt
            out.completion_tokens = ct
            out.total_tokens = pt + ct
        except Exception as e:
            out.error = str(e)

        out.elapsed = (datetime.datetime.now() - t0).total_seconds()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_ground_truth() -> dict[str, dict]:
    gt: dict[str, dict] = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["clip_id"]
            labels = {}
            for col in ALL_LABELS:
                v = row.get(col, "")
                labels[col] = v if v and v != "null" else None
            gt[cid] = labels
    return gt


def build_samples(gt: dict, sample_n: int | None = None) -> list[Sample]:
    clip_ids = list(gt.keys())
    if sample_n:
        clip_ids = clip_ids[:sample_n]

    samples, missing = [], []
    for cid in clip_ids:
        clip_dir = VIDEOS_DIR / cid
        mp4s = sorted(clip_dir.glob("*.mp4")) if clip_dir.exists() else []
        if mp4s:
            samples.append(Sample(
                video_path=str(mp4s[0]),
                label=json.dumps(gt[cid], ensure_ascii=False),
                uuid=cid,
            ))
        else:
            missing.append(cid)

    if missing:
        print(f"[WARN] {len(missing)} clips missing MP4: {missing[:3]}...")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def _clip_id(r: InferResult) -> str:
    """Extract clip_id from video_path (parent directory name)."""
    return Path(r.video_path).parent.name


def compute_metrics(results: list[InferResult], gt: dict) -> dict:
    ok, failed = [], []
    for r in results:
        cid = _clip_id(r)
        if r.status != "success":
            failed.append(cid)
            continue
        try:
            pred = json.loads(r.result)
        except (json.JSONDecodeError, TypeError):
            failed.append(cid)
            continue
        ok.append((cid, pred, gt.get(cid, {})))

    if failed:
        print(f"[WARN] {len(failed)} clips skipped (failed/parse_error)")

    metrics: dict = {}

    # Binary labels: only evaluate on GT=是 clips (null = unannotated, not negative)
    # → only Recall is meaningful; Precision/F1 excluded
    for label in BINARY_LABELS:
        tp = fn = 0
        for _, pred, truth in ok:
            if truth.get(label) != "是":   # skip unannotated clips
                continue
            pred_pos = (str(pred.get(label, "")).strip() == "是")
            if pred_pos: tp += 1
            else:        fn += 1

        total_pos = tp + fn
        rec = tp / total_pos * 100 if total_pos > 0 else float("nan")
        metrics[label] = {
            "TP": tp, "FN": fn,
            "positives_in_gt": total_pos,
            "recall": round(rec, 1) if rec == rec else None,
        }

    # 天气: multi-class accuracy (only clips with non-null GT)
    correct = total = 0
    confusion: dict[str, int] = {}
    for _, pred, truth in ok:
        gt_val = truth.get("天气")
        if gt_val is None:
            continue
        total += 1
        pv = str(pred.get("天气", "")).strip()
        if gt_val == pv:
            correct += 1
        key = f"{gt_val}→{pv}"
        confusion[key] = confusion.get(key, 0) + 1
    weather_classes = sorted({truth.get("天气") for _, _, truth in ok if truth.get("天气")})
    metrics["天气"] = {
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0.0,
        "correct": correct, "total": total,
        "n_classes": len(weather_classes),
        "confusion": dict(sorted(confusion.items(), key=lambda x: -x[1])[:20]),
    }

    # 时段: binary accuracy (only clips with non-null GT)
    correct = total = 0
    for _, pred, truth in ok:
        gt_val = truth.get("时段")
        if gt_val is None:
            continue
        total += 1
        if gt_val == str(pred.get("时段", "")).strip():
            correct += 1
    metrics["时段"] = {
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0.0,
        "correct": correct, "total": total,
    }

    metrics["_summary"] = {
        "total_clips": len(results),
        "parsed_ok": len(ok),
        "failed_or_parse_error": len(failed),
    }
    return metrics


def estimate_cost(results: list[InferResult]) -> dict:
    i = sum(r.prompt_tokens for r in results)
    o = sum(r.completion_tokens for r in results)
    return {
        "input_tokens":  i,
        "output_tokens": o,
        "total_tokens":  i + o,
        "cost_cny":      round(i / 1000 * INPUT_PRICE_PER_1K + o / 1000 * OUTPUT_PRICE_PER_1K, 2),
        "clips_ok":      sum(1 for r in results if r.status == "success"),
        "clips_failed":  sum(1 for r in results if r.status != "success"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────
def save_predictions_csv(results: list[InferResult], out_path: Path):
    """Write a CSV with clip_id + all predicted labels."""
    rows = []
    for r in results:
        row = {"clip_id": _clip_id(r), "status": r.status}
        if r.status == "success":
            try:
                pred = json.loads(r.result)
                for label in ALL_LABELS:
                    row[f"pred_{label}"] = pred.get(label, "")
                row["reason"] = pred.get("reason", "")
            except (json.JSONDecodeError, TypeError):
                for label in ALL_LABELS:
                    row[f"pred_{label}"] = ""
                row["reason"] = ""
        else:
            for label in ALL_LABELS:
                row[f"pred_{label}"] = ""
            row["reason"] = r.error
        rows.append(row)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Predictions CSV → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(v) -> str:
    if v is None or v != v:   # None or nan
        return "  N/A "
    return f"{v:5.1f}%"


def print_report(metrics: dict, cost: dict, version: str):
    print()
    print("=" * 70)
    print(f"  eval_benchmark {version}  —  推理结果报告")
    print("=" * 70)
    print(f"\n  {'标签':<16} {'GT正例':>6} {'TP':>5} {'FN':>5}  {'Recall':>8}")
    print("  " + "-" * 42)
    for label in BINARY_LABELS:
        m = metrics[label]
        print(f"  {label:<16} {m['positives_in_gt']:>6} {m['TP']:>5} {m['FN']:>5}  "
              f"{_fmt(m['recall']):>8}")

    wm, tm = metrics["天气"], metrics["时段"]
    print()
    n_classes = metrics["天气"].get("n_classes", 6)
    print(f"  天气  ({n_classes}-way acc)  : {wm['accuracy']:5.1f}%  ({wm['correct']}/{wm['total']})")
    print(f"  时段  (binary acc) : {tm['accuracy']:5.1f}%  ({tm['correct']}/{tm['total']})")

    sm = metrics["_summary"]
    print()
    print(f"  样本  : {sm['total_clips']} 总  /  {sm['parsed_ok']} 成功  /  {sm['failed_or_parse_error']} 失败")
    print()
    print(f"  Token : 输入 {cost['input_tokens']:,}  +  输出 {cost['output_tokens']:,}  =  {cost['total_tokens']:,}")
    print(f"  成本  : ¥{cost['cost_cny']:.2f} CNY  "
          f"(input×¥{INPUT_PRICE_PER_1K}/1K + output×¥{OUTPUT_PRICE_PER_1K}/1K)")
    print("=" * 70)
    print()

    if wm["confusion"]:
        print("  天气混淆矩阵 (前10条):")
        for k, v in list(wm["confusion"].items())[:10]:
            mark = "  " if "→" not in k or k.split("→")[0] == k.split("→")[1] else "✗ "
            print(f"    {mark}{k}: {v}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Multi-label eval for eval_benchmark.csv")
    ap.add_argument("--sample",   type=int, default=None, help="only run first N clips (for testing)")
    ap.add_argument("--workers",  type=int, default=5)
    ap.add_argument("--version",  default="v10")
    ap.add_argument("--engine",     default="dashscope", choices=["dashscope", "vllm", "qwen35"])
    ap.add_argument("--vllm-url",   default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--vllm-model", default="qwen2.5-vl-7b-instruct")
    ap.add_argument("--vllm-mode",  default="jpeg", choices=["jpeg", "mp4"],
                    help="jpeg: extract frames as JPEG (default); mp4: compress+pad to 16s MP4")
    ap.add_argument("--qwen35-url",   default=QWEN35_VLLM_URL)
    ap.add_argument("--qwen35-model", default=QWEN35_MODEL_PATH)
    ap.add_argument("--qwen35-mode",  default="jpeg", choices=["jpeg", "mp4"],
                    help="Qwen3.5-9B local mode: jpeg frames (default) or mp4")
    args = ap.parse_args()

    version      = args.version
    cache_path   = PROMPTS_DIR / f"cache_{version}.json"
    results_path = PROMPTS_DIR / f"results_{version}.json"
    metrics_path = PROMPTS_DIR / f"eval_metrics_{version}.json"
    prompt_path  = PROMPTS_DIR / f"{version}.md"

    print(f"Loading ground truth from {CSV_PATH} ...")
    gt = load_ground_truth()
    print(f"  {len(gt)} clips")

    prompt = prompt_path.read_text(encoding="utf-8")
    print(f"Loaded prompt  : {len(prompt)} chars  ({prompt_path.name})")

    samples = build_samples(gt, sample_n=args.sample)
    print(f"Built {len(samples)} samples")

    if args.engine == "vllm":
        cfg = InferenceConfig(
            engine="vllm",
            model=args.vllm_model,
            workers=args.workers,
            fps=1,
            timeout_per_sample=120,
            vllm_url=args.vllm_url,
        )
        client = MultiLabelVLLMClient(cfg, mode=args.vllm_mode)
        print(f"vLLM mode      : {args.vllm_mode}")
    elif args.engine == "qwen35":
        cfg = InferenceConfig(
            engine="vllm",
            model=args.qwen35_model,
            workers=args.workers,
            fps=1,
            timeout_per_sample=180,
            vllm_url=args.qwen35_url,
        )
        client = MultiLabelVLLMClient(cfg, mode=args.qwen35_mode)
        print(f"Qwen3.5 model  : {args.qwen35_model}")
        print(f"Qwen3.5 URL    : {args.qwen35_url}")
        print(f"Qwen3.5 mode   : {args.qwen35_mode}")
    else:
        cfg = InferenceConfig(
            model="qwen3.7-plus",
            workers=args.workers,
            fps=2,
            thinking_budget=512,
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        )
        client = MultiLabelClient(cfg)

    runner = BatchRunner(
        config=cfg,
        cache_path=str(cache_path),
        tracker=TokenTracker(),
    )

    print(f"\nRunning inference  ({len(samples)} clips, workers={args.workers}) ...")
    results = runner.run(
        samples, prompt,
        infer_fn=lambda s: client.infer(s, prompt),
    )

    # Save raw results
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_out = [
        {
            "clip_id":    _clip_id(r),
            "video_path": r.video_path,
            "gt":         r.label,
            "prediction": r.result,
            "reason":     r.reason,
            "status":     r.status,
            "tokens":     r.total_tokens,
            "elapsed_s":  round(r.elapsed, 2),
            "error":      r.error,
        }
        for r in results
    ]
    results_path.write_text(json.dumps(raw_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Raw results  → {results_path}")

    metrics = compute_metrics(results, gt)
    cost    = estimate_cost(results)
    metrics_path.write_text(
        json.dumps({**metrics, "cost": cost}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictions_csv = PROMPTS_DIR / f"predictions_{version}.csv"
    save_predictions_csv(results, predictions_csv)
    print(f"Metrics      → {metrics_path}")

    print_report(metrics, cost, version)


if __name__ == "__main__":
    main()
