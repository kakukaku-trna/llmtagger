#!/usr/bin/env python3
"""
多摄像头视频拼接脚本

从 ADW 下载多个摄像头的原始 H265 视频，拼接成单个 MP4 视频。
布局: Front120 居中放大，SideView_FL/FR 两侧，Rear 下方居中 — 映射物理摄像头位置。

         +-----------+
         | Front120  |
+--------+   960x540  +--------+
|SideView|           |SideView|
|  _FL   |           |  _FR   |
|480x270 |           |480x270 |
+--------+-----------+--------+
         |   Rear    |
         |  960x270  |
         +-----------+
         Canvas: 1920x810

用法:
    ADW_USER=xxx ADW_PROD_PASS=xxx ADW_STG_PASS=xxx \\
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \\
    python3 scripts/download_and_stitch_video.py \\
        --uuid-file uuids.txt \\
        --output-dir /path/to/output \\
        --cameras Front120,SideView_FL,SideView_FR,Rear \\
        --workers 4
"""
import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import importlib.util
spec = importlib.util.spec_from_file_location("dl", PROJECT_ROOT / "scripts" / "download_adw_videos.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

# ── 布局配置 ──────────────────────────────────────────────────────────────────
# Front120 居中放大 (960x540), 侧视摄像头在两侧 (480x270), 后视在下方居中 (960x270)
# 侧视摄像头垂直居中在 Front120 的高度内 (y=135, 即 (540-270)/2)
# Canvas 总尺寸: 1920x810

CAM_LAYOUT = {
    "Front120":    {"w": 960, "h": 540, "x": 480,  "y": 0,   "fontsize": 28},
    "SideView_FL": {"w": 480, "h": 270, "x": 0,    "y": 135, "fontsize": 18},
    "SideView_FR": {"w": 480, "h": 270, "x": 1440, "y": 135, "fontsize": 18},
    "Rear":        {"w": 960, "h": 270, "x": 480,  "y": 540, "fontsize": 22},
}

CANVAS_W = 1920
CANVAS_H = 810
CLIP_DURATION = 10
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ── 下载 + remux ─────────────────────────────────────────────────────────────

def download_and_remux(uuid, camera, tmp_dir, adw_user, adw_prod_pass, adw_stg_pass):
    """下载单个摄像头 H265 并 remux 为 MP4。返回 MP4 路径或 None。"""
    h265_path = tmp_dir / f"{uuid}_{camera}.h265"

    if not h265_path.exists() or h265_path.stat().st_size < 1_000_000:
        adw = dl.get_client("prod", adw_user, adw_prod_pass, adw_stg_pass)
        rowkey, table = dl.find_camera_rowkey(adw, uuid, camera)
        if not rowkey:
            adw_stg = dl.get_client("stg", adw_user, adw_prod_pass, adw_stg_pass)
            rowkey, table = dl.find_camera_rowkey(adw_stg, uuid, camera)
            if not rowkey:
                return None
            adw = adw_stg

        url = dl.get_presigned_url(adw, rowkey, table)
        if not url:
            return None

        if not dl.download_file(url, h265_path):
            return None

    mp4_path = tmp_dir / f"{uuid}_{camera}.mp4"
    remuxed = dl.remux_to_mp4(h265_path)
    if remuxed is None:
        h265_path.unlink(missing_ok=True)
        return None

    shutil.move(str(remuxed), str(mp4_path))
    h265_path.unlink(missing_ok=True)
    return mp4_path


# ── ffmpeg 拼接 ──────────────────────────────────────────────────────────────

def build_ffmpeg_command(cam_mp4s, cameras, output_path):
    """构建 ffmpeg 命令: xstack 拼接多摄像头视频，Front120 居中放大。"""

    first_cam = next((c for c in cameras if c in cam_mp4s), None)
    if first_cam is None:
        return None

    total_duration = dl.get_duration(cam_mp4s[first_cam])
    seek = max(0, (total_duration - CLIP_DURATION) / 2)

    inputs = []
    filter_parts = []
    xstack_inputs = []
    layout_parts = []

    for i, cam in enumerate(cameras):
        layout = CAM_LAYOUT.get(cam)
        if layout is None:
            continue

        w, h = layout["w"], layout["h"]
        x, y = layout["x"], layout["y"]
        fs = layout["fontsize"]

        if cam in cam_mp4s:
            inputs.extend(["-ss", f"{seek:.3f}", "-i", str(cam_mp4s[cam]),
                           "-t", str(CLIP_DURATION + 2)])
        else:
            inputs.extend(["-f", "lavfi", "-i",
                           f"color=black:s={w}x{h}:d={CLIP_DURATION + 2}:r=30"])

        filter_parts.append(
            f"[{i}:v]scale={w}:{h},"
            f"drawtext=text='{cam}':fontsize={fs}:fontcolor=yellow:"
            f"x=10:y=10:box=1:boxcolor=black@0.5:fontfile='{FONT_PATH}',"
            f"format=yuv420p[v{i}]"
        )
        xstack_inputs.append(f"[v{i}]")
        layout_parts.append(f"{x}_{y}")

    xstack_filter = (
        f"{''.join(xstack_inputs)}"
        f"xstack=inputs={len(xstack_inputs)}:layout={'|'.join(layout_parts)}:fill=black,"
        f"format=yuv420p[out]"
    )

    filter_complex = ";".join(filter_parts + [xstack_filter])

    cmd = [
        "ffmpeg",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-t", str(CLIP_DURATION),
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
        "-y",
    ]

    return cmd


# ── 单个 UUID 处理 ───────────────────────────────────────────────────────────

def process_one(uuid, output_dir, cameras, adw_user, adw_prod_pass, adw_stg_pass, tmp_base):
    """处理单个 UUID: 下载所有摄像头, 拼接成视频。"""
    try:
        tmp_dir = tmp_base / uuid
        tmp_dir.mkdir(parents=True, exist_ok=True)

        cam_mp4s = {}
        for cam in cameras:
            mp4_path = download_and_remux(uuid, cam, tmp_dir,
                                          adw_user, adw_prod_pass, adw_stg_pass)
            if mp4_path:
                cam_mp4s[cam] = mp4_path

        if not cam_mp4s:
            return uuid, False, "no cameras downloaded"

        output_path = output_dir / uuid / f"{uuid}_multicam.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_ffmpeg_command(cam_mp4s, cameras, output_path)
        if cmd is None:
            return uuid, False, "failed to build ffmpeg command"

        result = subprocess.run(cmd, capture_output=True, timeout=300)

        if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size < 50_000:
            stderr = result.stderr.decode()[-500:]
            return uuid, False, f"ffmpeg: {stderr}"

        meta = {
            "uuid": uuid,
            "cameras": list(cam_mp4s.keys()),
            "layout": "Front120 center 960x540, SideView_FL/FR sides 480x270, Rear bottom 960x270",
            "canvas": f"{CANVAS_W}x{CANVAS_H}",
            "duration": CLIP_DURATION,
        }
        (output_path.parent / "_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        shutil.rmtree(tmp_dir, ignore_errors=True)

        size_mb = output_path.stat().st_size / 1024 / 1024
        return uuid, True, f"{len(cam_mp4s)} cams, {size_mb:.1f}MB"

    except Exception as e:
        return uuid, False, str(e)


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="多摄像头视频拼接 (Front120 居中)")
    parser.add_argument("--uuid-file", required=True, help="UUID 列表文件")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--cameras", default="Front120,SideView_FL,SideView_FR,Rear",
                        help="摄像头列表（逗号分隔）")
    parser.add_argument("--workers", type=int, default=4, help="并发数")
    parser.add_argument("--sample", type=int, default=None, help="只处理前 N 条")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",")]
    output_dir = Path(args.output_dir)
    tmp_base = output_dir / "_tmp_download"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.uuid_file) as f:
        uuids = [line.strip() for line in f if line.strip()]

    if args.sample:
        uuids = uuids[:args.sample]

    adw_user = os.environ.get("ADW_USER", "")
    adw_prod_pass = os.environ.get("ADW_PROD_PASS", "")
    adw_stg_pass = os.environ.get("ADW_STG_PASS", "")

    if not adw_prod_pass:
        print("ERROR: ADW_PROD_PASS 未设置")
        sys.exit(1)

    print(f"Downloading {len(uuids)} clips x {len(cameras)} cameras -> {output_dir}")
    print(f"  Cameras: {cameras}")
    print(f"  Layout:  Front120 center (960x540), sides (480x270), Rear bottom (960x270)")
    print(f"  Canvas:  {CANVAS_W}x{CANVAS_H}, duration: {CLIP_DURATION}s")
    print()

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, uuid, output_dir, cameras,
                        adw_user, adw_prod_pass, adw_stg_pass, tmp_base): uuid
            for uuid in uuids
        }
        for i, fut in enumerate(as_completed(futures)):
            uuid, success, msg = fut.result()
            status = "[ OK ]" if success else "[FAIL]"
            print(f"  {status} {uuid[:12]}  {msg}")
            if success:
                ok += 1
            else:
                fail += 1
            if (i + 1) % 10 == 0:
                print(f"  --- {i+1}/{len(uuids)}: {ok} ok, {fail} failed ---")

    shutil.rmtree(tmp_base, ignore_errors=True)
    print(f"\nDone: {ok} success, {fail} failed")


if __name__ == "__main__":
    main()
