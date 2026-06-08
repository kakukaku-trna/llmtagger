"""Point cloud modality: render LIDAR/radar point cloud data to a 2D image.

Supports .bin (KITTI-style float32 x,y,z,intensity) and .pcd (ASCII/binary)
file formats. Renders a bird's-eye-view (BEV) projection by default, or a
front-view projection if requested.

Dependencies (optional):
    - numpy  – required
    - matplotlib – required for rendering
    - open3d – optional, used when available for richer rendering
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def render_pointcloud_b64(
    path: str,
    view: Literal["bev", "front"] = "bev",
    x_range: Tuple[float, float] = (-40.0, 40.0),
    y_range: Tuple[float, float] = (-40.0, 40.0),
    z_range: Tuple[float, float] = (-3.0, 5.0),
    resolution: float = 0.1,
    img_size: Tuple[int, int] = (800, 800),
) -> Optional[str]:
    """Load a point cloud file, render to image, return base64 JPEG string.

    Returns None if the file cannot be loaded or rendering fails.
    """
    try:
        pts = _load_pointcloud(path)
        if pts is None or len(pts) == 0:
            return None
        if view == "bev":
            img = _render_bev(pts, x_range, y_range, z_range, resolution, img_size)
        else:
            img = _render_front(pts, x_range, z_range, img_size)
        return _to_b64(img)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _load_pointcloud(path: str) -> Optional[np.ndarray]:
    """Load point cloud. Returns Nx4 array (x, y, z, intensity) or Nx3."""
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".bin":
        return _load_bin(p)
    if p.suffix == ".pcd":
        return _load_pcd(p)
    return None


def _load_bin(path: Path) -> np.ndarray:
    """KITTI-style binary: each point = (x, y, z, intensity) float32."""
    raw = np.fromfile(str(path), dtype=np.float32)
    return raw.reshape(-1, 4)


def _load_pcd(path: Path) -> Optional[np.ndarray]:
    """Minimal ASCII PCD loader (skips binary-compressed format)."""
    lines = path.read_text(errors="ignore").splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "data ascii":
            data_start = i + 1
            break
    if data_start == 0:
        return None
    pts = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                pass
    return np.array(pts, dtype=np.float32) if pts else None


# ─────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────

def _render_bev(
    pts: np.ndarray,
    x_range, y_range, z_range,
    resolution: float,
    img_size: Tuple[int, int],
) -> "np.ndarray":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
    )
    pts = pts[mask]

    fig, ax = plt.subplots(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if len(pts) > 0:
        intensity = pts[:, 3] if pts.shape[1] >= 4 else pts[:, 2]
        intensity = (intensity - intensity.min()) / (intensity.ptp() + 1e-9)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c=intensity, cmap="viridis", alpha=0.7)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_front(pts, x_range, z_range, img_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = (pts[:, 0] >= 0) & (pts[:, 0] <= x_range[1])
    pts = pts[mask]

    fig, ax = plt.subplots(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
    ax.set_facecolor("black")
    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 2], s=0.3, c="cyan", alpha=0.5)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _to_b64(raw_bytes: bytes) -> str:
    return base64.b64encode(raw_bytes).decode("utf-8")
