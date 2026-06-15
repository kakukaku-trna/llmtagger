"""Evaluation metrics: precision, recall, F1, confusion matrix. Persists to metrics.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline.inference.dashscope_client import InferResult


@dataclass
class Metrics:
    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    total: int = 0
    success: int = 0
    failed: int = 0
    tokens_used: int = 0

    def meets_target(
        self,
        precision_target: float,
        recall_target: float,
        accuracy_target: float = 0.0,
    ) -> bool:
        """Return True when all configured targets are met.

        accuracy_target = 0 means the accuracy target is not enforced.
        """
        ok = self.recall >= recall_target and self.precision >= precision_target
        if accuracy_target > 0:
            ok = ok and self.accuracy >= accuracy_target
        return ok

    def opt_score(self) -> float:
        """Primary optimisation score: recall + precision (equal weight)."""
        return self.recall + self.precision

    def better_than(self, other: "Metrics") -> bool:
        return self.f1 > other.f1


def calc_metrics(results: List[InferResult], tokens_used: int = 0) -> Metrics:
    """Compute confusion matrix and derived metrics from inference results."""
    tp = fp = tn = fn = 0
    success = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]

    for r in success:
        label, pred = r.label, r.result
        if label == "是" and pred == "是":
            tp += 1
        elif label == "否" and pred == "是":
            fp += 1
        elif label == "否" and pred == "否":
            tn += 1
        elif label == "是" and pred == "否":
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total * 100 if total > 0 else 0.0

    # Sum tokens from results if not provided
    if tokens_used == 0:
        tokens_used = sum(r.total_tokens for r in results)

    return Metrics(
        TP=tp,
        FP=fp,
        TN=tn,
        FN=fn,
        precision=round(precision, 1),
        recall=round(recall, 1),
        f1=round(f1, 1),
        accuracy=round(accuracy, 1),
        total=len(results),
        success=len(success),
        failed=len(failed),
        tokens_used=tokens_used,
    )


def save_version_metrics(
    scene: str,
    version: str,
    metrics: Metrics,
    metrics_path: Path,
) -> None:
    """Append/update version metrics in prompts/{scene}/metrics.json."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict = {}
    if metrics_path.exists() and metrics_path.stat().st_size > 2:
        with open(metrics_path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    existing[version] = asdict(metrics)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_all_versions(metrics_path: Path) -> Dict[str, Metrics]:
    """Load all version metrics from metrics.json."""
    if not metrics_path.exists():
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError:
            return {}
    return {v: Metrics(**d) for v, d in raw.items()}


def print_metrics_report(metrics: Metrics, version: str = "", scene: str = "") -> None:
    header = f"场景: {scene}  版本: {version}" if (scene or version) else "评估结果"
    print(f"\n{'=' * 55}")
    print(f" {header}")
    print(f"{'='*55}")
    print(f" 混淆矩阵:  TP={metrics.TP}  FP={metrics.FP}  FN={metrics.FN}  TN={metrics.TN}")
    print(f" 召回率 Recall    : {metrics.recall:.1f}%")      # primary
    print(f" 精确率 Precision : {metrics.precision:.1f}%")   # primary
    print(f" F1 Score         : {metrics.f1:.1f}%")
    print(f" 准确率 Accuracy  : {metrics.accuracy:.1f}%")    # secondary
    print(f" 样本总数: {metrics.total}  成功: {metrics.success}  失败: {metrics.failed}")
    print(f" Token 消耗: {metrics.tokens_used:,}")
    print(f"{'=' * 55}\n")


def best_version(all_versions: Dict[str, Metrics]) -> Tuple[str, Metrics]:
    """Return the version with the best recall + precision (primary), F1 (tiebreaker)."""
    return max(
        all_versions.items(),
        key=lambda kv: (kv[1].opt_score(), kv[1].f1, kv[1].recall, kv[1].precision),
    )


def print_versions_table(
    all_versions: Dict[str, Metrics],
    highlight: Optional[str] = None,
) -> None:
    """Print a comparison table of all prompt versions.

    Columns ordered by importance: Recall, Precision, F1 (primary), Accuracy (secondary).
    highlight: version name to mark with ★.
    """
    if not all_versions:
        return
    print(f"\n{'版本':<10} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Accuracy':>9} {'Tokens':>10}")
    print("-" * 61)
    for ver, m in sorted(all_versions.items()):
        marker = " ★" if ver == highlight else "  "
        print(
            f"{ver:<8}{marker} {m.recall:>7.1f}% {m.precision:>9.1f}%"
            f" {m.f1:>7.1f}% {m.accuracy:>8.1f}% {m.tokens_used:>10,}"
        )
    print()
