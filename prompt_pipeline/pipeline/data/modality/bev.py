"""BEV (Bird's Eye View) image modality.

Autonomous driving datasets often include BEV images that show a top-down
fused view of the scene (camera + lidar projection). This module loads and
encodes those images for inclusion in multimodal prompts.

Expected file layout (matches the da_mining dataset structure):
    {uuid}_FW_{timestamp}-v1/1.tsr.sensor.default.FW.jpg   ← front-wide camera
    {uuid}_FW_{timestamp}-v1/bev.jpg                        ← BEV if present
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional


def load_bev_b64(
    sample_dir: str,
    max_side: int = 800,
) -> Optional[str]:
    """Load the BEV image from a sample directory and return base64 string.

    Returns None if no BEV image is found in the directory.
    """
    bev_path = _find_bev(Path(sample_dir))
    if bev_path is None:
        return None
    return _encode(bev_path, max_side)


def load_all_views(
    sample_dir: str,
    max_side: int = 800,
) -> List[str]:
    """Load all camera-view images from a sample directory.

    Returns list of base64-encoded images (front, rear, BEV, etc.).
    """
    views = []
    d = Path(sample_dir)
    for img_path in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
        views.append(_encode(img_path, max_side))
    return views


def _find_bev(directory: Path) -> Optional[Path]:
    for name in ["bev.jpg", "bev.png", "BEV.jpg", "BEV.png"]:
        p = directory / name
        if p.exists():
            return p
    return None


def _encode(path: Path, max_side: int) -> str:
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
