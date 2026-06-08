"""Video modality: extract key frames and base64-encode video files.

Extracted frames can be passed individually as images (for models that
do not support direct video input) or the raw video can be base64-encoded
for models that accept video_url (DashScope / vLLM).
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import List, Optional


# ─────────────────────────────────────────────────────────────
# Full-video encode (used by DashScope client)
# ─────────────────────────────────────────────────────────────

def encode_video(path: str) -> str:
    """Base64-encode a video file (no compression)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────

def extract_frames(
    path: str,
    fps: float = 1.0,
    max_frames: int = 16,
    max_side: int = 720,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Extract frames from a video using ffmpeg.

    Returns a list of absolute paths to the extracted JPEG frames.
    Frames are written to output_dir (or a sibling directory of the video).
    """
    src = Path(path)
    out_dir = Path(output_dir) if output_dir else src.parent / f".frames_{src.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale frames
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()

    vf = f"fps={fps},scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-frames:v", str(max_frames),
        "-q:v", "3",
        str(out_dir / "frame_%04d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:400]}")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [str(f) for f in frames]


def frames_to_b64(frame_paths: List[str]) -> List[str]:
    """Base64-encode a list of frame image files."""
    from pipeline.data.modality.image import encode_image
    return [encode_image(p) for p in frame_paths]


def compress_video(
    src: str,
    dst: str,
    width: int = 480,
    height: int = 270,
    fps: int = 1,
    crf: int = 30,
    max_seconds: int = 15,
) -> str:
    """Re-encode video to a smaller resolution for API upload.

    Returns the destination path.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-t", str(max_seconds),
        "-vf", f"fps={fps},scale={width}:{height}",
        "-c:v", "libx264", "-crf", str(crf),
        "-an",
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg compress failed: {result.stderr.decode()[:400]}")
    return dst
