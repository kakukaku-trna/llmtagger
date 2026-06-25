#!/usr/bin/env python3
"""Batch generate BEV videos for all UUID folders under input_dir.

Usage:
    # Process all UUID folders with default settings
    python batch_bev.py --input-dir /data/clips --output-dir /data/bev

    # Only Front120 camera, parallel 8 workers
    python batch_bev.py \
        --input-dir /data/clips \
        --output-dir /data/bev \
        --camera-name Front120 \
        --workers 8

    # Skip existing outputs (resume)
    python batch_bev.py \
        --input-dir /data/clips \
        --output-dir /data/bev \
        --skip-existing

Environment:
    export XUTILS_LIB_PATH=/path/to/xutils/build/Linux-x86_64/lib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple

sys.path.insert(0, str(Path(__file__).parent))

from generate_bev import (
    DEFAULT_CAMERA_NAME,
    DEFAULT_CALIB_CAMERA_NAME,
    DEFAULT_CODEC,
    DEFAULT_FPS,
    find_video_and_calib,
    generate_bev,
)
from bev_utils.front120_bev_projector import Front120BevConfig


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_WORKERS = 4


class TaskConfig(NamedTuple):
    """Parameters passed to each worker process."""

    uuid: str
    input_dir: Path
    output_dir: Path
    camera_name: str
    calib_camera_name: str
    fps: float
    codec: str
    crf: int
    preset: str
    bev_config: Front120BevConfig
    fallback_calib: Path | None
    skip_existing: bool


def _process_single(config: TaskConfig) -> dict:
    """Process one UUID. Returns a result dict for logging."""
    result = {
        "uuid": config.uuid,
        "status": "unknown",
        "error": None,
        "video_path": None,
        "calib_path": None,
        "output_path": None,
        "duration_sec": None,
    }

    start = datetime.now()
    try:
        video_path, calib_path = find_video_and_calib(
            config.uuid,
            config.input_dir,
            config.camera_name,
            config.calib_camera_name,
            config.fallback_calib,
        )
        result["video_path"] = str(video_path)
        result["calib_path"] = str(calib_path)

        output_path = (
            config.output_dir
            / config.uuid
            / f"{config.uuid}_{config.camera_name}_bev_1280x720.mp4"
        )
        result["output_path"] = str(output_path)

        # Resume: skip if output already exists and looks valid
        if config.skip_existing and output_path.exists():
            result["status"] = "skipped"
            return result

        generate_bev(
            video_path,
            calib_path,
            output_path,
            fps=config.fps,
            codec=config.codec,
            crf=config.crf,
            preset=config.preset,
            config=config.bev_config,
            show_progress=False,
        )
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["duration_sec"] = (datetime.now() - start).total_seconds()

    return result


def _discover_uuids(input_dir: Path, camera_name: str) -> List[str]:
    """Discover UUID folders that contain the target camera video."""
    uuids: list[str] = []
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        uuid = subdir.name
        # Quick check: does a matching video exist?
        video_candidates = list(subdir.glob(f"{uuid}_{camera_name}.mp4"))
        if video_candidates:
            uuids.append(uuid)

    return uuids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generate BEV videos for all UUID folders"
    )

    # Directories
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing UUID sub-folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root directory",
    )

    # Camera selection
    parser.add_argument(
        "--camera-name",
        type=str,
        default=DEFAULT_CAMERA_NAME,
        help=f"Camera name suffix in MP4 filename (default: {DEFAULT_CAMERA_NAME})",
    )
    parser.add_argument(
        "--calib-camera-name",
        type=str,
        default=DEFAULT_CALIB_CAMERA_NAME,
        help=(
            f"Calibration sub-directory / file base name"
            f" (default: {DEFAULT_CALIB_CAMERA_NAME})"
        ),
    )

    # Processing options
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel worker processes (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip UUIDs whose output already exists",
    )
    parser.add_argument(
        "--uuids",
        type=str,
        default=None,
        help="Comma-separated UUID list (instead of auto-discover)",
    )

    # Video parameters
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help="Output frame rate"
    )
    parser.add_argument(
        "--codec",
        type=str,
        default=DEFAULT_CODEC,
        help="FourCC codec string (deprecated)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 CRF quality, lower=better (default: 23)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="x264 encoding preset (default: medium)",
    )

    # BEV configuration
    parser.add_argument(
        "--bev-fx",
        type=float,
        default=300.0,
        help="BEV camera focal length X",
    )
    parser.add_argument(
        "--bev-fy",
        type=float,
        default=300.0,
        help="BEV camera focal length Y",
    )
    parser.add_argument(
        "--bev-pos-x",
        type=float,
        default=5.0,
        help="BEV camera X position (forward)",
    )
    parser.add_argument(
        "--bev-pos-y",
        type=float,
        default=0.0,
        help="BEV camera Y position (left)",
    )
    parser.add_argument(
        "--bev-pos-z",
        type=float,
        default=10.0,
        help="BEV camera Z position (up)",
    )
    parser.add_argument(
        "--bev-pitch-deg",
        type=float,
        default=15.0,
        help="BEV camera forward pitch from vertical-down (deg)",
    )

    # Fallback calibration
    parser.add_argument(
        "--fallback-calib",
        type=Path,
        default=None,
        help="Fallback calibration JSON when UUID-specific calib is missing",
    )

    args = parser.parse_args()

    # Build BEV config once
    bev_config = Front120BevConfig(
        bev_fx=args.bev_fx,
        bev_fy=args.bev_fy,
        bev_pos_x=args.bev_pos_x,
        bev_pos_y=args.bev_pos_y,
        bev_pos_z=args.bev_pos_z,
        bev_pitch_deg=args.bev_pitch_deg,
    )

    # Discover UUIDs
    if args.uuids:
        uuids = [u.strip() for u in args.uuids.split(",")]
    else:
        print(f"Scanning {args.input_dir} for {args.camera_name} videos...")
        uuids = _discover_uuids(args.input_dir, args.camera_name)

    print(f"Found {len(uuids)} UUID(s) to process")
    if not uuids:
        return

    # Prepare output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = [
        TaskConfig(
            uuid=u,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            camera_name=args.camera_name,
            calib_camera_name=args.calib_camera_name,
            fps=args.fps,
            codec=args.codec,
            crf=args.crf,
            preset=args.preset,
            bev_config=bev_config,
            fallback_calib=args.fallback_calib,
            skip_existing=args.skip_existing,
        )
        for u in uuids
    ]

    # Run parallel processing
    results: list[dict] = []
    success = skipped = failed = 0

    print(f"Starting with {args.workers} worker(s)...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(_process_single, t): t for t in tasks}
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)

            status = result["status"]
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

            print(
                f"[{result['uuid']}] {status.upper()} ({result['duration_sec']:.1f}s)"
            )
            if result["error"]:
                print(f"  Error: {result['error']}")

    # Summary
    total = len(tasks)
    print("\n" + "=" * 50)
    print(
        f"Batch complete: {success}/{total} success, {skipped} skipped, {failed} failed"
    )
    print("=" * 50)

    # Write JSON log
    log_path = args.output_dir / "batch_bev_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "camera_name": args.camera_name,
                "summary": {
                    "total": total,
                    "success": success,
                    "skipped": skipped,
                    "failed": failed,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Log written to: {log_path}")


if __name__ == "__main__":
    main()
