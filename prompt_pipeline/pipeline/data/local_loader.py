"""Local file data loader. Discovers videos or frame sets from positive/negative directories."""
from __future__ import annotations

import glob
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pipeline.config import DataConfig


@dataclass
class Sample:
    video_path: str     # primary media path (mp4 or representative frame path)
    label: str          # "是" for positive, "否" for negative
    uuid: str           # derived from directory name or filename


def _discover_media(directory: str) -> List[str]:
    """Recursively discover all media under directory.

    Returns:
        - For frame sets (frame_*.jpg): the parent directory path
          (DashScope client auto-detects frame_*.jpg inside)
        - For MP4: the .mp4 file path directly
    Frame mode takes priority over MP4 in the same directory.
    """
    root = Path(directory)
    media: List[str] = []
    seen: set = set()

    # 1. Frame mode first: find directories containing frame_*.jpg
    for jpg in sorted(root.rglob("frame_*.jpg")):
        dir_ = jpg.parent
        if dir_ not in seen:
            media.append(str(dir_))
            seen.add(dir_)

    # 2. MP4 mode: find .mp4 files whose parent dir is not already in frame mode
    for mp4 in sorted(glob.glob(str(root / "**" / "*.mp4"), recursive=True)):
        parent = Path(mp4).parent
        if parent not in seen:
            media.append(mp4)
            seen.add(parent)

    return media


def find_mp4s(directory: str) -> List[str]:
    """Recursively find all .mp4 files under directory."""
    return sorted(glob.glob(str(Path(directory) / "**" / "*.mp4"), recursive=True))


def load_samples(
    config: DataConfig,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[Sample]:
    """Load all video samples from positive and negative directories.

    Args:
        config: DataConfig with positive_dir and negative_dir.
        sample_size: If set, randomly sample this many total samples (balanced).
        seed: Random seed for reproducibility.
    """
    positives = _load_dir(config.positive_dir, "是")
    negatives = _load_dir(config.negative_dir, "否")
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


def _load_dir(directory: str, label: str) -> List[Sample]:
    if not directory or not Path(directory).exists():
        return []
    items = _discover_media(directory)
    return [
        Sample(
            video_path=p,
            label=label,
            uuid=_extract_uuid(p),
        )
        for p in items
    ]


def _extract_uuid(video_path: str) -> str:
    """Extract UUID from path: use parent directory name or directory itself."""
    p = Path(video_path)
    if p.is_dir():
        return p.name
    return p.parent.name
