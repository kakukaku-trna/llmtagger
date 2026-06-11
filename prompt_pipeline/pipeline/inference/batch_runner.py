"""Batch inference runner. Adapted from detect_blind_curve.py two-stage parallel pattern."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pipeline.config import InferenceConfig
from pipeline.data.local_loader import Sample
from pipeline.inference.dashscope_client import (
    DashScopeClient,
    InferResult,
    _encode_video,
)
from pipeline.monitor.alert import AlertLevel, emit_alert
from pipeline.monitor.raw_log import RawLogWriter
from pipeline.monitor.token_tracker import TokenTracker


def _encode_worker(video_path: str):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    return video_path, _encode_video(video_path)


def build_client(config: InferenceConfig):
    """Factory: return the appropriate inference client."""
    if config.engine == "vllm":
        from pipeline.inference.vllm_client import VLLMClient

        return VLLMClient(config)
    return DashScopeClient(config)


class BatchRunner:
    """Runs batch inference with caching, timeout detection, token tracking, and raw-response logging."""

    def __init__(
        self,
        config: InferenceConfig,
        cache_path: Optional[str] = None,
        tracker: Optional[TokenTracker] = None,
        feishu_webhook: Optional[str] = None,
        alert_email: Optional[str] = None,
        log_writer: Optional[RawLogWriter] = None,
    ):
        self._config = config
        self._cache_path = cache_path
        self._tracker = tracker or TokenTracker()
        self._feishu_webhook = feishu_webhook
        self._alert_email = alert_email
        self._log_writer = log_writer
        self._cache: Dict[str, InferResult] = {}
        self._cache_lock = threading.Lock()

        if cache_path and Path(cache_path).exists():
            self._load_cache()

    # ─────────────────────────────────────────────────────────
    def run(
        self,
        samples: List[Sample],
        prompt: str,
        infer_fn: Optional[Callable] = None,
        *,
        scene: str = "",
        version: str = "",
        round_n: int = 0,
    ) -> List[InferResult]:
        """Run inference on all samples. Returns list of InferResult."""
        self._scene = scene
        self._version = version
        self._round_n = round_n

        if infer_fn is None:
            client = build_client(self._config)
            infer_fn = lambda s: client.infer(s, prompt)  # noqa: E731

        # Split into cached and pending
        cached = [
            self._cache[s.video_path] for s in samples if s.video_path in self._cache
        ]
        pending = [s for s in samples if s.video_path not in self._cache]

        if cached:
            print(f"  缓存命中: {len(cached)} 条")
        if not pending:
            return cached

        print(f"  待处理: {len(pending)} 条  并发: {self._config.workers}")

        new_results = self._run_parallel(pending, infer_fn)
        all_results = cached + new_results

        # Update token tracker
        for r in new_results:
            self._tracker.add(r.prompt_tokens, r.completion_tokens)

        return all_results

    # ─────────────────────────────────────────────────────────
    def _run_parallel(
        self, samples: List[Sample], infer_fn: Callable
    ) -> List[InferResult]:
        results: List[InferResult] = []
        consecutive_timeouts = 0

        with ThreadPoolExecutor(max_workers=self._config.workers) as pool:
            future_map = {pool.submit(infer_fn, s): s for s in samples}
            for future in as_completed(future_map):
                try:
                    r: InferResult = future.result(
                        timeout=self._config.timeout_per_sample + 5
                    )
                    if r.elapsed > self._config.timeout_per_sample:
                        consecutive_timeouts += 1
                    else:
                        consecutive_timeouts = 0

                    if consecutive_timeouts >= 5:
                        emit_alert(
                            AlertLevel.WARNING,
                            "推理卡死",
                            f"连续 {consecutive_timeouts} 条超时（>{self._config.timeout_per_sample}s），已暂停",
                            feishu_webhook=self._feishu_webhook,
                            email=self._alert_email,
                        )
                        break

                    self._save_to_cache(r)
                    results.append(r)
                    self._print_result(r)
                    if self._log_writer:
                        self._log_writer.write(
                            r,
                            scene=self._scene,
                            version=self._version,
                            round_n=self._round_n,
                        )
                except Exception as e:
                    s = future_map[future]
                    r = InferResult(
                        video_path=s.video_path,
                        label=s.label,
                        result="",
                        reason="",
                        status="failed",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        elapsed=0.0,
                        error=str(e),
                    )
                    results.append(r)
                    if self._log_writer:
                        self._log_writer.write(
                            r,
                            scene=self._scene,
                            version=self._version,
                            round_n=self._round_n,
                        )

        return results

    # ─────────────────────────────────────────────────────────
    def _save_to_cache(self, result: InferResult):
        if not self._cache_path:
            return
        with self._cache_lock:
            self._cache[result.video_path] = result
            data = {k: _result_to_dict(v) for k, v in self._cache.items()}
            Path(self._cache_path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _load_cache(self):
        with open(self._cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            self._cache[k] = _dict_to_result(v)

    @staticmethod
    def _print_result(r: InferResult):
        correct = (
            ("✓" if r.result == r.label else "✗") if r.status == "success" else " "
        )
        name = Path(r.video_path).name
        print(f"  [{correct}] {name}")
        print(
            f"       标签={r.label}  预测={r.result}  "
            f"耗时={r.elapsed:.1f}s  tok={r.total_tokens}"
        )
        if r.status != "success":
            print(f"       错误: {r.error[:100]}")


def _result_to_dict(r: InferResult) -> dict:
    return {
        "video_path": r.video_path,
        "label": r.label,
        "result": r.result,
        "reason": r.reason,
        "status": r.status,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "total_tokens": r.total_tokens,
        "elapsed": r.elapsed,
        "error": r.error,
    }


def _dict_to_result(d: dict) -> InferResult:
    return InferResult(
        video_path=d.get("video_path", ""),
        label=d.get("label", ""),
        result=d.get("result", ""),
        reason=d.get("reason", ""),
        status=d.get("status", "failed"),
        prompt_tokens=d.get("prompt_tokens", 0),
        completion_tokens=d.get("completion_tokens", 0),
        total_tokens=d.get("total_tokens", 0),
        elapsed=d.get("elapsed", 0.0),
        error=d.get("error", ""),
    )
