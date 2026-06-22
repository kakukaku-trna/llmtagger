#!/usr/bin/env python3
"""Generate BEV video from Front120 camera.

Usage:
    # Manual mode (all paths explicit)
    python generate_front120_bev.py \
        --video /path/to/Front120.mp4 \
        --calib /path/to/front_wide.json \
        --output /path/to/output.mp4

    # UUID auto-resolution mode
    python generate_front120_bev.py \
        --input-dir /data/clips \
        --uuid 7dc9e003-57c6-474e-9904-16ef29a4923c \
        --camera-name Front120

    # Set xutils C library path via environment variable
    export XUTILS_LIB_PATH=/path/to/xutils/build/Linux-x86_64/lib
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from bev_utils.front120_bev_projector import Front120BevProjector, Front120BevConfig


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_FPS = 15
DEFAULT_CODEC = "mp4v"
DEFAULT_CAMERA_NAME = "Front120"
DEFAULT_CALIB_CAMERA_NAME = "front_wide"


# ── Path Resolution ──────────────────────────────────────────────────────────


def find_video_and_calib(
    uuid: str,
    input_dir: Path,
    camera_name: str = DEFAULT_CAMERA_NAME,
    calib_camera_name: str = DEFAULT_CALIB_CAMERA_NAME,
    fallback_calib: Path | None = None,
) -> tuple[Path, Path]:
    """Find video and calibration for a UUID under input_dir.

    Searches:
        video: {input_dir}/{uuid}/{uuid}_{camera_name}.mp4
        calib: {input_dir}/{uuid}/online_calibration_main_s1/camera/{calib_camera_name}/{calib_camera_name}.json

    Args:
        uuid: Data UUID.
        input_dir: Root directory containing UUID sub-folders.
        camera_name: Camera name suffix in the MP4 filename.
        calib_camera_name: Calibration sub-directory / file base name.
        fallback_calib: If calibration not found for this UUID, use this path instead.

    Returns:
        (video_path, calib_path)

    Raises:
        FileNotFoundError: If video or calibration cannot be found.
    """
    video_dir = input_dir / uuid

    # Video
    video_candidates = list(video_dir.glob(f"{uuid}_{camera_name}.mp4"))
    if not video_candidates:
        raise FileNotFoundError(f"{camera_name} video not found in {video_dir}")
    video_path = video_candidates[0]

    # Calibration
    calib_path = (
        video_dir
        / "online_calibration_main_s1"
        / "camera"
        / calib_camera_name
        / f"{calib_camera_name}.json"
    )
    if not calib_path.exists():
        if fallback_calib and fallback_calib.exists():
            print(
                f"  Warning: calibration not found for {uuid},"
                f" using fallback: {fallback_calib}"
            )
            calib_path = fallback_calib
        else:
            raise FileNotFoundError(f"Calibration not found: {calib_path}")

    return video_path, calib_path


# ── BEV Generation ───────────────────────────────────────────────────────────


def generate_bev(
    video_path: Path,
    calib_path: Path,
    output_path: Path,
    fps: float = DEFAULT_FPS,
    codec: str = DEFAULT_CODEC,
    crf: int = 23,
    preset: str = "medium",
    config: Front120BevConfig | None = None,
    show_progress: bool = True,
) -> None:
    """Generate BEV video from Front120 source.

    Args:
        video_path: Path to Front120 MP4.
        calib_path: Path to front_wide.json calibration.
        output_path: Output MP4 path.
        fps: Output frame rate.
        codec: FourCC codec string (deprecated, kept for API compat).
        crf: H.264 CRF quality (lower=better, 23 default).
        preset: x264 encoding preset (default "medium").
        config: Optional BEV configuration overrides.
        show_progress: Print progress.
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize projector
    projector = Front120BevProjector(config)
    projector.init(calib_path, video_w, video_h)

    bev_w, bev_h = projector.get_bev_size()
    rx, ry = projector.get_remap_tables()

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start ffmpeg pipe for H.264 encoding
    ffmpeg_cmd = [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{bev_w}x{bev_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
        "-y",
    ]
    ffmpeg = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if show_progress:
        print(f"Processing {total_frames} frames: {video_path} -> {output_path}")
        print(f"  Video: {video_w}x{video_h}, BEV: {bev_w}x{bev_h}")
        print(f"  Encoding: libx264 crf={crf} preset={preset}")

    try:
        # Process frames
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Remap to BEV
            bev_frame = cv2.remap(
                frame,
                rx,
                ry,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            ffmpeg.stdin.write(bev_frame.tobytes())

            if show_progress and (frame_idx + 1) % 30 == 0:
                pct = (frame_idx + 1) / total_frames * 100
                print(
                    f"  {frame_idx + 1}/{total_frames} frames ({pct:.1f}%)",
                    flush=True,
                )
    finally:
        cap.release()
        ffmpeg.stdin.close()
        retcode = ffmpeg.wait()
        if retcode != 0:
            stderr = ffmpeg.stderr.read().decode()[-500:] if ffmpeg.stderr else ""
            raise RuntimeError(f"ffmpeg failed (exit {retcode}): {stderr}")

    if show_progress:
        print(f"  Done: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BEV video from Front120 camera"
    )

    # Path group
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(os.environ.get("BEV_INPUT_DIR", ".")),
        help="Root directory containing UUID sub-folders (default: $BEV_INPUT_DIR or .)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("BEV_OUTPUT_DIR", ".")),
        help="Output root directory (default: $BEV_OUTPUT_DIR or .)",
    )
    parser.add_argument("--uuid", type=str, help="UUID for auto path resolution")
    parser.add_argument("--video", type=Path, help="Path to Front120 MP4 (manual mode)")
    parser.add_argument(
        "--calib", type=Path, help="Path to front_wide.json (manual mode)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path (manual mode, or override auto mode)",
    )

    # Camera name group
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

    # Resolve paths
    if args.uuid:
        video_path, calib_path = find_video_and_calib(
            args.uuid,
            args.input_dir,
            args.camera_name,
            args.calib_camera_name,
            args.fallback_calib,
        )
        if args.output:
            output_path = args.output
        else:
            output_path = (
                args.output_dir
                / args.uuid
                / f"{args.uuid}_{args.camera_name}_bev_1280x720.mp4"
            )
    else:
        if not args.video or not args.calib:
            parser.error("--uuid or both --video and --calib required")
        video_path = args.video
        calib_path = args.calib
        output_path = args.output or Path("front120_bev_output.mp4")

    # Build config
    config = Front120BevConfig(
        bev_fx=args.bev_fx,
        bev_fy=args.bev_fy,
        bev_pos_x=args.bev_pos_x,
        bev_pos_y=args.bev_pos_y,
        bev_pos_z=args.bev_pos_z,
        bev_pitch_deg=args.bev_pitch_deg,
    )

    generate_bev(
        video_path,
        calib_path,
        output_path,
        fps=args.fps,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        config=config,
    )


if __name__ == "__main__":
    main()
