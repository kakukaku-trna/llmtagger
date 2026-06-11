"""
Download videos from NIO ADW (Autonomous Data Warehouse) platform.

Uses adwsdk to locate Front-camera H265 files by UUID, obtains presigned
download URLs, downloads the raw bitstream, and re-encodes to 720p H264 MP4
(scale=1280:720, preset veryfast) so that file sizes are small enough for the
DashScope VL API (~1-2 MB vs 31 MB for the raw 4K H265).

Output layout (compatible with pipeline/data/local_loader.py):
    {output_dir}/{uuid}/{uuid}_{camera}.mp4

Prerequisites:
  - adwsdk installed for the Python being used (see .claude/CLAUDE.md)
  - PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  (set automatically below)
  - ADW_USER / ADW_PROD_PASS environment variables (or pass via CLI)

Usage:
    python scripts/download_adw_videos.py \\
        --uuid-file /path/to/uuids.txt \\
        --output-dir /path/to/output \\
        --workers 6

    # Test with 5 videos first:
    python scripts/download_adw_videos.py \\
        --uuid-file data_uuid/bus_lane_true.txt \\
        --output-dir /home/huajiang.sun/model_muse/bus_lane_pos \\
        --sample 5

    # Different camera:
    python scripts/download_adw_videos.py \\
        --uuid-file data_uuid/bus_lane_true.txt \\
        --output-dir /home/huajiang.sun/model_muse/bus_lane_pos \\
        --camera Front120
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# ── configuration ─────────────────────────────────────────────────────────────
# Park_Front/Park_Rear/Park_Right/Park_Left/Front120/Front30/Rear/SideView_FR/SideView_FL/SideView_RR/SideView_RL
DEFAULT_CAMERA = "Front120"
ENCODE_CRF = "38"
ENCODE_SCALE = "1280:720"

DLB_NAMESPACES = [
    "production/collection",
    "production/alps/collection",
    "production/nt3/collection",
    "eu/production/collection",
]

# ── adwsdk setup ──────────────────────────────────────────────────────────────

# Must be set before importing adwsdk (protobuf pure-python mode)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from adwsdk.adw_client import AdwClient
    from adwsdk import cmm
except ImportError as e:
    print(f"ERROR: adwsdk not importable: {e}")
    print("See .claude/CLAUDE.md §adwsdk setup for installation instructions.")
    sys.exit(1)


# ── shared ADW clients ────────────────────────────────────────────────────────

_adw_prod: Optional["AdwClient"] = None
_adw_stg: Optional["AdwClient"] = None


def get_client(env: str = "prod", user: str = "", prod_pass: str = "", stg_pass: str = "") -> "AdwClient":
    global _adw_prod, _adw_stg
    if env == "prod":
        if _adw_prod is None:
            _adw_prod = AdwClient(env="prod", adw_user=user, adw_pass=prod_pass)
        return _adw_prod
    else:
        if _adw_stg is None:
            _adw_stg = AdwClient(env="stg", adw_user=user, adw_pass=stg_pass)
        return _adw_stg


# ── helpers ───────────────────────────────────────────────────────────────────

def find_camera_rowkey(adw: "AdwClient", uuid: str, camera: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (rowkey, table_name) for the camera H265/H264 file of this UUID."""
    for namespace in DLB_NAMESPACES:
        table = f"{namespace}/datafiles"
        try:
            files = adw.scan(cmm.Scan(
                table_name=table,
                where=f"uuid='{uuid}'",
                limit=200,
            ))
        except Exception:
            continue

        for f in files:
            fname = f.meta.get("file_name", "")
            if re.search(rf"_{camera}_.*\.(h265|h264)$", fname):
                return f.rowkey, table

    return None, None


def get_presigned_url(adw: "AdwClient", rowkey: str, table: str) -> Optional[str]:
    result = adw.get_download_url(cmm.GetUrl(table_name=table, rowkey=rowkey))
    return result.url if result.success else None


def download_file(url: str, dest: Path) -> bool:
    import requests
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  download error: {e}")
        return False


def convert_to_mp4(src: Path, dest: Path) -> bool:
    """Scale to 720p H264 MP4 (~7s per 4K H265 file with veryfast preset).

    Output is ~1-2 MB, well within DashScope API's base64 string size limit.
    vcodec copy was tried first but produces 31 MB files that exceed the limit.
    h264_nvenc was tried but the A100 here is a compute-only card (no NVENC).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(src),
        "-vf", f"scale={ENCODE_SCALE}",
        "-vcodec", "libx264", "-crf", ENCODE_CRF, "-preset", "veryfast",
        "-an", str(dest), "-y",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 100_000:
        dest.unlink(missing_ok=True)
        print(f"  ffmpeg error: {result.stderr.decode()[-300:]}")
        return False
    return True


# ── per-UUID pipeline ─────────────────────────────────────────────────────────

def process_uuid(uuid: str, output_dir: Path, camera: str,
                 adw_user: str, adw_prod_pass: str, adw_stg_pass: str) -> bool:
    """Download + re-encode one UUID. Returns True on success."""
    mp4_dest = output_dir / uuid / f"{uuid}_{camera}.mp4"

    # Skip if a valid (non-partial) MP4 already exists
    if mp4_dest.exists() and mp4_dest.stat().st_size > 100_000:
        print(f"[SKIP] {uuid[:12]}  already exists")
        return True
    mp4_dest.unlink(missing_ok=True)  # remove any partial/corrupt file

    h265_tmp = output_dir / uuid / f"{uuid}_{camera}.h265"
    h265_tmp.parent.mkdir(parents=True, exist_ok=True)

    # Skip re-download if H265 already present and non-empty
    if not h265_tmp.exists() or h265_tmp.stat().st_size < 1_000_000:
        adw = get_client("prod", adw_user, adw_prod_pass, adw_stg_pass)

        rowkey, table = find_camera_rowkey(adw, uuid, camera)
        if not rowkey:
            adw_stg = get_client("stg", adw_user, adw_prod_pass, adw_stg_pass)
            rowkey, table = find_camera_rowkey(adw_stg, uuid, camera)
            if not rowkey:
                print(f"[FAIL] {uuid[:12]}  {camera} file not found in any namespace")
                return False
            adw = adw_stg

        url = get_presigned_url(adw, rowkey, table)  # type: ignore[arg-type]
        if not url:
            print(f"[FAIL] {uuid[:12]}  could not get download URL")
            return False

        if not download_file(url, h265_tmp):
            print(f"[FAIL] {uuid[:12]}  download failed")
            return False

    # Re-encode H265 → 720p H264 MP4
    if not convert_to_mp4(h265_tmp, mp4_dest):
        print(f"[FAIL] {uuid[:12]}  ffmpeg conversion failed")
        h265_tmp.unlink(missing_ok=True)
        return False

    h265_tmp.unlink(missing_ok=True)
    print(f"[ OK ] {uuid[:12]}  → {mp4_dest.name}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

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
        description="Download NIO ADW videos by UUID list → 720p H264 MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--uuid-file", required=True,
        help="Path to a text file with one UUID per line (blank lines / # comments ignored)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Root directory for downloaded videos: {output_dir}/{uuid}/{uuid}_{camera}.mp4",
    )
    parser.add_argument(
        "--camera", default=DEFAULT_CAMERA,
        help=f"Camera stream to download (default: {DEFAULT_CAMERA})",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Parallel download/encode workers (default: 6)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only process the first N UUIDs (for testing)",
    )
    parser.add_argument(
        "--adw-user", default=os.environ.get("ADW_USER", "dayun.shen"),
        help="ADW username (default: $ADW_USER or 'dayun.shen')",
    )
    parser.add_argument(
        "--adw-prod-pass", default=os.environ.get("ADW_PROD_PASS", ""),
        help="ADW production password (default: $ADW_PROD_PASS)",
    )
    parser.add_argument(
        "--adw-stg-pass", default=os.environ.get("ADW_STG_PASS", ""),
        help="ADW staging password (default: $ADW_STG_PASS)",
    )
    args = parser.parse_args()

    if not args.adw_prod_pass:
        parser.error("ADW_PROD_PASS not set. Export it or pass --adw-prod-pass.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uuids = read_uuids(Path(args.uuid_file))
    if args.sample:
        uuids = uuids[: args.sample]

    total = len(uuids)
    print(f"Downloading {total} videos → {output_dir}")
    print(f"  UUID file : {args.uuid_file}")
    print(f"  Camera    : {args.camera}")
    print(f"  Workers   : {args.workers}")
    print(f"  Encoding  : scale={ENCODE_SCALE}, libx264 crf={ENCODE_CRF} preset=veryfast\n")

    success, fail = 0, 0
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_uuid, uuid, output_dir, args.camera,
                args.adw_user, args.adw_prod_pass, args.adw_stg_pass,
            ): uuid
            for uuid in uuids
        }
        for i, future in enumerate(as_completed(futures), 1):
            uuid = futures[future]
            ok = future.result()
            if ok:
                success += 1
            else:
                fail += 1
                failed.append(uuid)
            if i % 10 == 0 or i == total:
                print(f"  --- {i}/{total}: {success} ok, {fail} failed ---")

    print(f"\nDone: {success} success, {fail} failed")
    if failed:
        fail_log = output_dir / "failed_uuids.txt"
        fail_log.write_text("\n".join(failed))
        print(f"Failed UUIDs saved to: {fail_log}")


if __name__ == "__main__":
    main()
