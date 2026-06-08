"""Local file data loader. Discovers videos from positive/negative directories."""
from __future__ import annotations

import glob
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pipeline.config import DataConfig


@dataclass
class Sample:
    video_path: str
    label: str      # "是" for positive, "否" for negative
    uuid: str       # derived from directory name or filename


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
    return [
        Sample(
            video_path=p,
            label=label,
            uuid=_extract_uuid(p),
        )
        for p in find_mp4s(directory)
    ]


def _extract_uuid(video_path: str) -> str:
    """Extract UUID from path: use parent directory name."""
    return Path(video_path).parent.name
