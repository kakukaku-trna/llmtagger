"""Image modality: load, resize, and base64-encode image files."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Tuple


def load_image_b64(
    path: str,
    max_side: int = 1024,
    fmt: str = "JPEG",
) -> str:
    """Load an image file, resize to max_side on the longest edge, return base64 string."""
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("RGB")
        img = _resize(img, max_side)
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        # Pillow not installed — raw encode
        return encode_image(path)


def encode_image(path: str) -> str:
    """Raw base64 encode without resizing (fast path)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_url(b64: str, mime: str = "image/jpeg") -> str:
    """Build a data URI suitable for the DashScope multimodal API."""
    return f"data:{mime};base64,{b64}"


def _resize(img, max_side: int):
    """Resize image so its longest side is at most max_side pixels."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)))
