"""Stream loader: pull samples from a remote dataset API without full local download.

Compared to local_loader (which scans local directories), stream_loader
queries a dataset service by scene UUID lists and fetches video/image URLs
on demand. This avoids downloading the full dataset before inference.

Interface is identical to local_loader so BatchRunner is transparent to
the data source.
"""
from __future__ import annotations

import random
import uuid as uuid_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.config import DataConfig
from pipeline.data.local_loader import Sample

# Default UUID list directory (relative to this file's package root)
_UUID_DIR = Path(__file__).parent.parent.parent / "data_uuid"


@dataclass
class StreamConfig:
    """Remote dataset API configuration."""
    api_base: str = "http://dataset-api.internal/v1"
    api_key: str = ""
    # Local cache directory for downloaded files
    cache_dir: str = "/tmp/llmtagger_stream_cache"
    # Max concurrent downloads
    download_workers: int = 4
    # Timeout for each download request (seconds)
    download_timeout: int = 30


@dataclass
class RemoteSample:
    """A sample fetched from the remote dataset API."""
    uuid: str
    label: str
    scene: str
    modalities: Dict[str, str] = field(default_factory=dict)
    # e.g. {"video": "https://...", "bev": "https://...", "pointcloud": "https://..."}
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_sample(self, local_path: str) -> Sample:
        """Convert to the pipeline-internal Sample type after local caching."""
        return Sample(video_path=local_path, label=self.label, uuid=self.uuid)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def load_uuid_list(scene: str, label: str, uuid_dir: Optional[Path] = None) -> List[str]:
    """Read UUID list from data_uuid/{scene}_{true|false}.txt."""
    base = uuid_dir or _UUID_DIR
    suffix = "true" if label == "是" else "false"
    path = base / f"{scene}_{suffix}.txt"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def load_samples_from_uuids(
    scene: str,
    stream_cfg: StreamConfig,
    sample_size: Optional[int] = None,
    seed: int = 42,
    uuid_dir: Optional[Path] = None,
) -> List[Sample]:
    """Fetch samples from the remote API by UUID.

    Falls back to local cache if already downloaded.
    Returns the same List[Sample] interface as local_loader.
    """
    pos_uuids = load_uuid_list(scene, "是", uuid_dir)
    neg_uuids = load_uuid_list(scene, "否", uuid_dir)

    candidates: List[RemoteSample] = []
    for u in pos_uuids:
        candidates.append(RemoteSample(uuid=u, label="是", scene=scene))
    for u in neg_uuids:
        candidates.append(RemoteSample(uuid=u, label="否", scene=scene))

    if sample_size and sample_size < len(candidates):
        rng = random.Random(seed)
        candidates = rng.sample(candidates, sample_size)

    return _download_batch(candidates, stream_cfg)


def load_samples(
    config: DataConfig,
    stream_cfg: StreamConfig,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[Sample]:
    """Unified entry point for stream loading. Parallel to local_loader.load_samples."""
    scene = getattr(config, "scene", "unknown")
    return load_samples_from_uuids(scene, stream_cfg, sample_size, seed)


# ─────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────

def _download_batch(
    samples: List[RemoteSample],
    cfg: StreamConfig,
) -> List[Sample]:
    """Download (or hit cache for) each sample. Returns local Sample list."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: List[Sample] = []

    with ThreadPoolExecutor(max_workers=cfg.download_workers) as pool:
        futures = {pool.submit(_fetch_one, s, cfg): s for s in samples}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                results.append(result)

    return results


def _fetch_one(remote: RemoteSample, cfg: StreamConfig) -> Optional[Sample]:
    """Download a single sample to the local cache directory.

    This is a stub that generates a deterministic local path.
    Replace _api_get_video_url() with your actual API call.
    """
    import os
    cache = Path(cfg.cache_dir) / remote.scene / remote.uuid
    cache.mkdir(parents=True, exist_ok=True)
    local_path = str(cache / "video.mp4")

    if not os.path.exists(local_path):
        video_url = _api_get_video_url(remote.uuid, remote.scene, cfg)
        if video_url is None:
            return None
        _http_download(video_url, local_path, cfg)

    return remote.to_sample(local_path)


def _api_get_video_url(uuid: str, scene: str, cfg: StreamConfig) -> Optional[str]:
    """Query the dataset API for a video URL.

    Stub implementation — replace with your actual API call.
    Example real call:
        resp = requests.get(
            f"{cfg.api_base}/samples/{uuid}/video",
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=10,
        )
        return resp.json()["url"]
    """
    # TODO: replace with actual API
    return None


def _http_download(url: str, dest: str, cfg: StreamConfig) -> None:
    """Download a file from a URL to a local destination."""
    import urllib.request
    urllib.request.urlretrieve(url, dest)
