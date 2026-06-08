"""Multi-scene parallel evaluation flow.

Runs multiple scenes concurrently (with Prefect) or sequentially (without).
Aggregates results into a cross-scene summary report.

Usage:
    python flows/multi_scene_flow.py --scenes blind_curve waitzone_left --sample 20
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Optional Prefect import ──────────────────────────────────
try:
    from prefect import flow, task
    from prefect.futures import PrefectFuture
    _PREFECT = True
except ImportError:
    _PREFECT = False

    def flow(fn=None, **kw):
        return fn if fn else (lambda f: f)

    def task(fn=None, **kw):
        return fn if fn else (lambda f: f)

from flows.scene_flow import run_scene_flow
from pipeline.evaluate.metrics import Metrics, print_versions_table


# ─────────────────────────────────────────────────────────────
# Multi-scene flow
# ─────────────────────────────────────────────────────────────

@flow(name="multi-scene-flow", log_prints=True)
def run_multi_scene_flow(
    scenes: List[str],
    sample_size: Optional[int] = None,
    iterate: bool = False,
    max_rounds: int = 3,
) -> Dict[str, Metrics]:
    """Run evaluation for multiple scenes.

    With Prefect installed, scenes are submitted concurrently using
    .submit() and results are gathered after all complete.

    Without Prefect, scenes are processed sequentially.

    Returns:
        Dict mapping scene name → final Metrics.
    """
    results: Dict[str, Metrics] = {}

    if _PREFECT:
        # Submit all scenes concurrently
        futures: Dict[str, PrefectFuture] = {}
        for scene in scenes:
            futures[scene] = run_scene_flow.submit(
                scene=scene,
                sample_size=sample_size,
                iterate=iterate,
                max_rounds=max_rounds,
            )
        for scene, fut in futures.items():
            try:
                results[scene] = fut.result()
            except Exception as e:
                print(f"  [ERROR] 场景 {scene} 失败: {e}")
    else:
        # Sequential fallback
        for scene in scenes:
            print(f"\n{'='*55}")
            print(f"  场景: {scene}")
            print(f"{'='*55}")
            try:
                m = run_scene_flow(
                    scene=scene,
                    sample_size=sample_size,
                    iterate=iterate,
                    max_rounds=max_rounds,
                )
                results[scene] = m
            except Exception as e:
                print(f"  [ERROR] 场景 {scene} 失败: {e}")

    _print_summary(results)
    return results


def _print_summary(results: Dict[str, Metrics]) -> None:
    if not results:
        return
    print(f"\n{'='*65}")
    print(f"  多场景汇总报告")
    print(f"{'='*65}")
    print(f"{'场景':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10} {'Tokens':>12}")
    print("-" * 65)
    for scene, m in sorted(results.items()):
        print(
            f"{scene:<22} {m.precision:>9.1f}% {m.recall:>7.1f}% "
            f"{m.f1:>7.1f}% {m.accuracy:>9.1f}% {m.tokens_used:>12,}"
        )
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", required=True, help="Scene names")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--iterate", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    run_multi_scene_flow(
        scenes=args.scenes,
        sample_size=args.sample,
        iterate=args.iterate,
        max_rounds=args.max_rounds,
    )
