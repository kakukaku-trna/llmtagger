"""
Download videos via DMS API (bypassing ADW storage_state=7 restriction).

DMS (Data Management Service) maintains its own copy of video data,
which is accessible even when ADW storage_state is 7 (deleted).

Usage:
    python download_via_dms.py \
        --uuid-file data_uuid/failed_uuids.txt \
        --output-dir /tmp/dms_downloads \
        --camera Front30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ── configuration ─────────────────────────────────────────────────────────────

DEFAULT_CAMERA = "Front30"
ENCODE_CRF = "23"
ENCODE_SCALE = "1280:720"

# DMS API configuration
DMS_API_HOST = "https://da-ms.nioint.com"
DMS_VIDEO_API = f"{DMS_API_HOST}/api/v1/videos"

# Camera name mapping (DMS uses different names)
CAMERA_MAPPING = {
    "Front30": "FW",  # Front Wide
    "Front120": "F120",  # Front 120
    "Front": "FW",
    "Rear": "R",
    "SideView_FL": "SL",
    "SideView_FR": "SR",
    "SideView_RL": "RL",
    "SideView_RR": "RR",
}


def get_dms_video_url(
    uuid: str, camera: str, timeout: int = 30
) -> Tuple[Optional[str], str]:
    """Get video download URL from DMS API.

    Returns:
        (url, error_message)
    """
    # Map camera name to DMS sensor code
    sensor = CAMERA_MAPPING.get(camera, camera)

    # Try different API endpoints
    endpoints = [
        f"{DMS_API_HOST}/api/v1/clips/{uuid}/videos",
        f"{DMS_API_HOST}/api/clip/{uuid}/videos",
        f"{DMS_API_HOST}/api/v1/clip/{uuid}",
        f"{DMS_API_HOST}/api/videos/{uuid}",
    ]

    for url in endpoints:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # Parse response to find video URL for specific camera
                video_url = _extract_video_url(data, sensor)
                if video_url:
                    return video_url, ""
        except Exception as e:
            continue

    return None, f"Could not find video URL for {camera} (sensor: {sensor})"


def _extract_video_url(data: dict, sensor: str) -> Optional[str]:
    """Extract video URL from DMS API response."""
    # Try different response formats
    if isinstance(data, dict):
        # Format 1: {"videos": [{"sensor": "FW", "url": "..."}]}
        videos = data.get("videos", data.get("data", []))
        if isinstance(videos, list):
            for video in videos:
                if isinstance(video, dict):
                    video_sensor = video.get("sensor", video.get("camera", ""))
                    if video_sensor.upper() == sensor.upper():
                        return video.get("url", video.get("download_url", ""))

        # Format 2: {"FW": {"url": "..."}}
        if sensor in data:
            sensor_data = data[sensor]
            if isinstance(sensor_data, dict):
                return sensor_data.get("url", "")

    return None


def download_file(url: str, dest: Path, timeout: int = 300) -> Tuple[bool, str]:
    """Download file from URL. Returns (success, error_message)."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def convert_to_mp4(src: Path, dest: Path) -> Tuple[bool, str]:
    """Scale to 720p H264 MP4. Returns (success, error_message)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        str(src),
        "-vf",
        f"scale={ENCODE_SCALE}",
        "-vcodec",
        "libx264",
        "-crf",
        ENCODE_CRF,
        "-preset",
        "veryfast",
        "-an",
        str(dest),
        "-y",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 100_000:
        dest.unlink(missing_ok=True)
        stderr = result.stderr.decode()[-500:] if result.stderr else "(no stderr)"
        return False, stderr
    return True, ""


def process_uuid(uuid: str, output_dir: Path, camera: str) -> Tuple[bool, str]:
    """Download one UUID via DMS. Returns (success, error_message)."""
    mp4_dest = output_dir / uuid / f"{uuid}_{camera}.mp4"

    # Skip if already exists
    if mp4_dest.exists() and mp4_dest.stat().st_size > 100_000:
        print(f"[SKIP] {uuid[:12]}  already exists")
        return True, ""

    h265_tmp = output_dir / uuid / f"{uuid}_{camera}.h265"

    # Get DMS video URL
    print(f"  Getting DMS URL for {uuid[:12]}...")
    url, error = get_dms_video_url(uuid, camera)
    if not url:
        print(f"[FAIL] {uuid[:12]}  {error}")
        return False, error

    # Download video
    print(f"  Downloading {uuid[:12]}...")
    ok, error = download_file(url, h265_tmp)
    if not ok:
        print(f"[FAIL] {uuid[:12]}  download failed: {error}")
        return False, error

    # Convert to MP4
    print(f"  Converting {uuid[:12]}...")
    ok, error = convert_to_mp4(h265_tmp, mp4_dest)
    if not ok:
        print(f"[FAIL] {uuid[:12]}  conversion failed: {error}")
        return False, error

    # Cleanup
    h265_tmp.unlink(missing_ok=True)
    print(f"[ OK ] {uuid[:12]}  → {mp4_dest.name}")
    return True, ""


def read_uuids(path: Path) -> List[str]:
    uuids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                uuids.append(line)
    return uuids


def main():
    parser = argparse.ArgumentParser(
        description="Download videos via DMS API (bypass ADW storage_state=7)",
    )
    parser.add_argument("--uuid-file", required=True, help="Path to UUID list file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--camera", default=DEFAULT_CAMERA, help=f"Camera (default: {DEFAULT_CAMERA})"
    )
    parser.add_argument(
        "--workers", type=int, default=3, help="Parallel workers (default: 3)"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Only process first N UUIDs"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uuids = read_uuids(Path(args.uuid_file))
    if args.sample:
        uuids = uuids[: args.sample]

    total = len(uuids)
    print(f"Downloading {total} videos via DMS → {output_dir}")
    print(f"  Camera: {args.camera}")
    print(f"  Workers: {args.workers}\n")

    success, fail = 0, 0
    failed: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_uuid, uuid, output_dir, args.camera): uuid
            for uuid in uuids
        }
        for i, future in enumerate(as_completed(futures), 1):
            uuid = futures[future]
            ok, error = future.result()
            if ok:
                success += 1
            else:
                fail += 1
                failed.append((uuid, error))
            if i % 5 == 0 or i == total:
                print(f"  --- {i}/{total}: {success} ok, {fail} failed ---")

    print(f"\nDone: {success} success, {fail} failed")

    if failed:
        fail_log = output_dir / "failed_uuids.txt"
        with open(fail_log, "w") as f:
            for uuid, error in failed:
                f.write(f"{uuid}\t{error}\n")
        print(f"Failed UUIDs saved to: {fail_log}")


if __name__ == "__main__":
    main()
