"""FP/FN failure analyzer. Categorizes errors and produces structured report."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pipeline.inference.dashscope_client import InferResult


@dataclass
class FailureCase:
    video_path: str
    label: str
    predicted: str
    reason: str
    error_type: str   # FP | FN


@dataclass
class FailureReport:
    fp_cases: List[FailureCase] = field(default_factory=list)   # 误报 (否→是)
    fn_cases: List[FailureCase] = field(default_factory=list)   # 漏报 (是→否)

    @property
    def total_failures(self) -> int:
        return len(self.fp_cases) + len(self.fn_cases)

    def summary(self) -> str:
        lines = [
            f"  失败分析: FP={len(self.fp_cases)}  FN={len(self.fn_cases)}",
        ]
        if self.fp_cases:
            lines.append("  误报案例 (FP - 实为否，预测为是):")
            for c in self.fp_cases[:3]:
                lines.append(f"    - {Path(c.video_path).parent.name}: {c.reason[:60]}")
        if self.fn_cases:
            lines.append("  漏报案例 (FN - 实为是，预测为否):")
            for c in self.fn_cases[:3]:
                lines.append(f"    - {Path(c.video_path).parent.name}: {c.reason[:60]}")
        return "\n".join(lines)


def analyze_failures(results: List[InferResult]) -> FailureReport:
    """Identify FP and FN cases from inference results."""
    report = FailureReport()
    for r in results:
        if r.status != "success":
            continue
        if r.label == "否" and r.result == "是":
            report.fp_cases.append(FailureCase(
                video_path=r.video_path,
                label=r.label,
                predicted=r.result,
                reason=r.reason,
                error_type="FP",
            ))
        elif r.label == "是" and r.result == "否":
            report.fn_cases.append(FailureCase(
                video_path=r.video_path,
                label=r.label,
                predicted=r.result,
                reason=r.reason,
                error_type="FN",
            ))
    return report
