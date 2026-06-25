#!/usr/bin/env python3
"""
多摄像头下载 + 拼接脚本

从 ADW 下载多个摄像头的帧，拼接成 2×2 网格图像，供 pipeline 打标使用。

用法:
    ADW_USER=xxx ADW_PROD_PASS=xxx ADW_STG_PASS=xxx \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    python3 scripts/download_and_stitch_multicam.py \
        --uuid-file uuids.txt \
        --output-dir /path/to/output \
        --cameras Front120,SideView_FL,SideView_FR,Rear \
        --frame-count 5 \
        --workers 4
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import importlib.util
spec = importlib.util.spec_from_file_location("dl", PROJECT_ROOT / "scripts" / "download_adw_videos.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def download_camera_frames(uuid, output_base, camera, frame_count, adw_user, adw_prod_pass, adw_stg_pass):
    """下载单个 UUID 单个摄像头的帧，返回帧目录路径"""
    cam_base = output_base / f"_tmp_{camera}"
    frame_dir = cam_base / uuid
    if list(frame_dir.glob("frame_*.jpg")):
        return frame_dir
    dl.process_uuid(
        uuid, cam_base, camera,
        adw_user, adw_prod_pass, adw_stg_pass,
        frame_mode=True, frame_count=frame_count,
    )
    return frame_dir


def stitch_frames(cam_dirs, cameras, output_dir, frame_count, target_w=640, target_h=360):
    """将多个摄像头的帧拼接成 2×2 网格，保存到 output_dir/frame_XX.jpg"""
    output_dir.mkdir(parents=True, exist_ok=True)

    font = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    label_h = 20
    cell_w = target_w
    cell_h = target_h + label_h
    canvas_w = cell_w * 2
    canvas_h = cell_h * 2

    for i in range(1, frame_count + 1):
        fname = f"frame_{i:02d}.jpg"
        canvas = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for idx, cam in enumerate(cameras):
            col = idx % 2
            row = idx // 2
            x_offset = col * cell_w
            y_offset = row * cell_h

            cam_dir = cam_dirs.get(cam)
            frame_path = cam_dir / fname if cam_dir else None

            if frame_path and frame_path.exists():
                img = Image.open(frame_path)
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                canvas.paste(img, (x_offset, y_offset + label_h))

            draw.text((x_offset + 5, y_offset + 2), cam, fill=(255, 255, 0), font=font)

        out_path = output_dir / fname
        canvas.save(out_path, "JPEG", quality=85)

    meta = {
        "cameras": cameras,
        "layout": "2x2 grid",
        "frame_count": frame_count,
        "cell_size": f"{target_w}x{target_h}",
    }
    (output_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def process_one(uuid, output_dir, cameras, frame_count, adw_user, adw_prod_pass, adw_stg_pass, tmp_base):
    """处理单个 UUID：下载多摄 + 拼接"""
    try:
        cam_dirs = {}
        for cam in cameras:
            d = download_camera_frames(uuid, tmp_base, cam, frame_count,
                                       adw_user, adw_prod_pass, adw_stg_pass)
            if d and list(d.glob("frame_*.jpg")):
                cam_dirs[cam] = d

        if not cam_dirs:
            return uuid, False, "no frames downloaded"

        stitch_dir = output_dir / uuid
        stitch_frames(cam_dirs, cameras, stitch_dir, frame_count)

        for cam in cameras:
            if cam in cam_dirs:
                import shutil
                shutil.rmtree(cam_dirs[cam].parent, ignore_errors=True)

        return uuid, True, f"{len(cam_dirs)} cameras stitched"
    except Exception as e:
        return uuid, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="多摄像头下载+拼接")
    parser.add_argument("--uuid-file", required=True, help="UUID 列表文件")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--cameras", default="Front120,SideView_FL,SideView_FR,Rear",
                        help="摄像头列表（逗号分隔）")
    parser.add_argument("--frame-count", type=int, default=5, help="每个 clip 提取帧数")
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

    print(f"Downloading {len(uuids)} clips × {len(cameras)} cameras → {output_dir}")
    print(f"  Cameras: {cameras}")
    print(f"  Frames per clip: {args.frame_count}")
    print()

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, uuid, output_dir, cameras, args.frame_count,
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

    import shutil
    shutil.rmtree(tmp_base, ignore_errors=True)

    print(f"\nDone: {ok} success, {fail} failed")


if __name__ == "__main__":
    main()
