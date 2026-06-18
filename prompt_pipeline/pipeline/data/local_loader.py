"""Local file data loader. Discovers videos or frame sets from positive/negative directories."""

from __future__ import annotations

import glob
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pipeline.config import DataConfig


@dataclass
class Sample:
    video_path: str  # primary media path (mp4 or representative frame path)
    label: str  # "是" for positive, "否" for negative
    uuid: str  # derived from directory name or filename


def _discover_media(directory: str, camera_suffix: str | None = None) -> List[str]:
    """Recursively discover all media under directory.

    Args:
        directory: Root directory to scan.
        camera_suffix: If set, only include files/directories whose stem (for
            MP4) or name (for frame dirs) ends with this suffix.

    Returns:
        - For frame sets (frame_*.jpg): the parent directory path
          (DashScope client auto-detects frame_*.jpg inside)
        - For MP4: the .mp4 file path directly
    Frame mode takes priority over MP4 in the same directory.
    """
    root = Path(directory)
    media: List[str] = []
    seen: set = set()

    def _matches(name: str) -> bool:
        """Check if name ends with camera_suffix (or always True if not set)."""
        return camera_suffix is None or name.endswith(camera_suffix)

    # 1. Frame mode first: find directories containing frame_*.jpg
    for jpg in sorted(root.rglob("frame_*.jpg")):
        dir_ = jpg.parent
        if dir_ not in seen and _matches(dir_.name):
            media.append(str(dir_))
            seen.add(dir_)

    # 2. MP4 mode: find .mp4 files whose parent dir is not already in frame mode
    for mp4 in sorted(glob.glob(str(root / "**" / "*.mp4"), recursive=True)):
        parent = Path(mp4).parent
        if parent not in seen and _matches(Path(mp4).stem):
            media.append(mp4)
            # Note: we do NOT add parent to seen here; multiple .mp4 files
            # inside the same parent directory are all valid samples.

    return media


def find_mp4s(directory: str, camera_suffix: str | None = None) -> List[str]:
    """Recursively find all .mp4 files under directory.

    Args:
        directory: Root directory to scan.
        camera_suffix: If set, only include files whose stem ends with this suffix.
    """
    all_mp4s = sorted(glob.glob(str(Path(directory) / "**" / "*.mp4"), recursive=True))
    if not camera_suffix:
        return all_mp4s
    return [p for p in all_mp4s if Path(p).stem.endswith(camera_suffix)]


def load_samples(
    config: DataConfig,
    sample_size: int | None = None,
    seed: int = 42,
    camera_suffix: str | None = None,
) -> List[Sample]:
    """Load all video samples from positive and negative directories.

    Args:
        config: DataConfig with positive_dir and negative_dir.
        sample_size: If set, randomly sample this many total samples (balanced).
        seed: Random seed for reproducibility.
        camera_suffix: If set, only include files whose stem ends with this suffix.
    """
    positives = _load_dir(config.positive_dir, "是", camera_suffix=camera_suffix)
    negatives = _load_dir(config.negative_dir, "否", camera_suffix=camera_suffix)
    all_samples = positives + negatives

    if sample_size and sample_size < len(all_samples):
        rng = random.Random(seed)
        # balanced sampling
        n_pos = min(sample_size // 2, len(positives))
        n_neg = min(sample_size - n_pos, len(negatives))
        sampled = rng.sample(positives, n_pos) + rng.sample(negatives, n_neg)
        rng.shuffle(sampled)
        return sampled

    return all_samples


def _load_dir(
    directory: str, label: str, camera_suffix: str | None = None
) -> List[Sample]:
    if not directory or not Path(directory).exists():
        return []
    items = _discover_media(directory, camera_suffix=camera_suffix)
    return [
        Sample(
            video_path=p,
            label=label,
            uuid=_extract_uuid(p),
        )
        for p in items
    ]


def _extract_uuid(video_path: str) -> str:
    """Extract UUID from path: use file stem (for mp4) or directory name (for frame dirs)."""
    p = Path(video_path)
    if p.is_dir():
        return p.name
    return p.stem
