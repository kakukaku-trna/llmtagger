"""Topic data modality: extract non-visual sensor/metadata from dataset topics.

In autonomous driving datasets, a "topic" represents a named data channel
(e.g., /ego_speed, /traffic_light_state, /localization). This module reads
topic metadata from JSON/YAML sidecar files and formats them as text context
that can be appended to the LLM prompt.

Expected sidecar file layout:
    {sample_dir}/topic.json      ← preferred
    {sample_dir}/meta.json       ← fallback
    {sample_dir}/meta.yaml
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def load_topic_meta(sample_dir: str) -> Dict[str, Any]:
    """Load topic metadata from a sample directory.

    Returns a flat dict: {topic_name: value, ...}. Empty dict if no
    sidecar file is found.
    """
    for name in ["topic.json", "meta.json"]:
        p = Path(sample_dir) / name
        if p.exists():
            return _read_json(p)
    for name in ["meta.yaml", "topic.yaml"]:
        p = Path(sample_dir) / name
        if p.exists():
            return _read_yaml(p)
    return {}


def topic_context_text(
    meta: Dict[str, Any],
    include: Optional[List[str]] = None,
) -> str:
    """Format topic metadata as a human-readable text block for LLM context.

    Args:
        meta: dict from load_topic_meta()
        include: optional whitelist of topic names to include

    Returns a markdown-formatted string, or empty string if meta is empty.
    """
    if not meta:
        return ""
    lines = ["**传感器/元数据上下文:**"]
    for k, v in meta.items():
        if include and k not in include:
            continue
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def parse_ego_motion(meta: Dict[str, Any]) -> Dict[str, float]:
    """Extract ego vehicle motion signals (speed, yaw rate, etc.) from topic meta."""
    keys = {
        "ego_speed": ["ego_speed", "speed", "velocity", "vehicle_speed"],
        "yaw_rate":  ["yaw_rate", "angular_velocity_z", "gyro_z"],
        "acceleration": ["acceleration", "accel", "longitudinal_accel"],
    }
    result = {}
    for canonical, aliases in keys.items():
        for alias in aliases:
            if alias in meta:
                try:
                    result[canonical] = float(meta[alias])
                except (TypeError, ValueError):
                    pass
                break
    return result


def parse_traffic_state(meta: Dict[str, Any]) -> Dict[str, str]:
    """Extract traffic-related state signals from topic meta."""
    keys = {
        "traffic_light": ["traffic_light_state", "tl_state", "signal_state"],
        "turn_signal":   ["turn_signal", "blinker", "indicator"],
        "gear":          ["gear", "gear_state"],
    }
    result = {}
    for canonical, aliases in keys.items():
        for alias in aliases:
            if alias in meta:
                result[canonical] = str(meta[alias])
                break
    return result


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return {}
