"""
从 NIO ADW（自动驾驶数据仓库）平台下载视频并提取帧。

三种模式：
  1. 帧模式（默认）：从原始 H265 视频提取 5 张高分辨率 JPEG 帧。
     VL 模型可获得更清晰的图像，同时 token 消耗更低。
  2. MP4 模式（--mp4）：重编码为 720p H264 MP4，用于视频推理。
  3. 转换模式（--from-mp4）：从已下载的 MP4 文件中提取帧。
  4. 重提取模式（--re-extract）：从已下载的 H265/H264 文件重新提取帧（不重新下载）。

输出结构（帧模式）：
    {output_dir}/{uuid}/frame_01.jpg, frame_02.jpg, ..., frame_05.jpg

输出结构（MP4 模式）：
    {output_dir}/{uuid}/{uuid}_{camera}.mp4

前置条件：
  - adwsdk 已安装（详见 .claude/CLAUDE.md）
  - PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python（下方自动设置）
  - ADW_USER / ADW_PROD_PASS 环境变量（或通过命令行传入）

用法：
    # 默认：每个视频提取 5 张高分辨率帧
    python scripts/download_adw_videos.py \\
        --uuid-file /path/to/uuids.txt \\
        --output-dir /path/to/output \\
        --workers 6

    # 从已有 MP4 文件提取帧（无需重新下载）
    python scripts/download_adw_videos.py \\
        --output-dir /home/huajiang.sun/data/晴天 \\
        --from-mp4 --frame-count 5

    # 从已有 H265 文件重新提取帧（无需重新下载）
    python scripts/download_adw_videos.py \\
        --output-dir /home/huajiang.sun/data/多云 \\
        --re-extract --frame-count 5

    # 旧 MP4 模式（720p 视频）
    python scripts/download_adw_videos.py \\
        --uuid-file data_uuid/bus_lane_true.txt \\
        --output-dir /home/huajiang.sun/model_muse/bus_lane_pos \\
        --mp4

    # 先测试 5 个视频：
    python scripts/download_adw_videos.py \\
        --uuid-file data_uuid/bus_lane_true.txt \\
        --output-dir /tmp/test_output \\
        --sample 5
"""

from __future__ import annotations

import argparse
import json
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
ENCODE_CRF = "23"
ENCODE_SCALE = "1280:720"

DLB_NAMESPACES = [
    "production/collection",
    "production/alps/collection",
    "production/nt3/collection",
    "eu/production/collection",
]

# ── adwsdk 初始化 ────────────────────────────────────────────────────────────

# 必须在导入 adwsdk 前设置（protobuf 纯 Python 模式）
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from adwsdk.adw_client import AdwClient
    from adwsdk import cmm
except ImportError as e:
    print(f"ERROR: adwsdk not importable: {e}")
    print("See .claude/CLAUDE.md §adwsdk setup for installation instructions.")
    sys.exit(1)


# ── 共享 ADW 客户端 ──────────────────────────────────────────────────────────

_adw_prod: Optional["AdwClient"] = None
_adw_stg: Optional["AdwClient"] = None


def get_client(
    env: str = "prod", user: str = "", prod_pass: str = "", stg_pass: str = ""
) -> "AdwClient":
    global _adw_prod, _adw_stg
    if env == "prod":
        if _adw_prod is None:
            _adw_prod = AdwClient(env="prod", adw_user=user, adw_pass=prod_pass)
        return _adw_prod
    else:
        if _adw_stg is None:
            _adw_stg = AdwClient(env="stg", adw_user=user, adw_pass=stg_pass)
        return _adw_stg


# ── 工具函数 ─────────────────────────────────────────────────────────────────


def find_camera_rowkey(
    adw: "AdwClient", uuid: str, camera: str
) -> Tuple[Optional[str], Optional[str]]:
    """查找该 UUID 对应相机的 H265/H264 文件，返回 (rowkey, table_name)。"""
    for namespace in DLB_NAMESPACES:
        table = f"{namespace}/datafiles"
        try:
            files = adw.scan(
                cmm.Scan(
                    table_name=table,
                    where=f"uuid='{uuid}'",
                    limit=200,
                )
            )
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
    """缩放为 720p H264 MP4（4K H265 文件约 7 秒/个，veryfast 预设）。

    输出约 1-2 MB，在 DashScope API 的 base64 字符串大小限制内。
    曾尝试 vcodec copy 但输出 31 MB 超出限制。
    曾尝试 h264_nvenc 但当前 A100 是纯计算卡（无 NVENC）。
    """
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
        print(f"  ffmpeg error: {result.stderr.decode()[-300:]}")
        return False
    return True


# ── 快速帧提取（remux + 关键帧定位）────────────────────────────────────────


def remux_to_mp4(src: Path) -> Optional[Path]:
    """将原始 H265/H264 封装为 MP4 容器（stream copy，不重编码）。

    返回临时 MP4 路径，失败返回 None。
    """
    tmp = src.with_name(src.stem + "._remux_tmp.mp4")
    cmd = ["ffmpeg", "-i", str(src), "-c", "copy", "-an", str(tmp), "-y"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            tmp.unlink(missing_ok=True)
            return None
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return None
    if tmp.exists() and tmp.stat().st_size > 1_000:
        return tmp
    tmp.unlink(missing_ok=True)
    return None


def get_duration(src: Path) -> float:
    """通过 ffprobe 获取视频时长（秒），失败时回退到 10.0 秒。"""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(src),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        val = float(probe.stdout.strip().split("\n")[0])
        return val if val > 0 else 10.0
    except (ValueError, IndexError):
        return 10.0


def extract_frames_fast(
    src: Path, dest_dir: Path, count: int = 5, *, need_remux: bool = True
) -> bool:
    """使用快速关键帧定位提取 `count` 张均匀分布的 JPEG 帧。

    对于原始 H265/H264（need_remux=True）：先封装为 MP4，再定位提取。
    对于 MP4 文件（need_remux=False）：直接定位提取。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    seek_source: Path
    remux_tmp: Optional[Path] = None

    if need_remux:
        remux_tmp = remux_to_mp4(src)
        if remux_tmp is None:
            return False
        seek_source = remux_tmp
    else:
        seek_source = src

    try:
        duration = get_duration(seek_source)
        margin = duration * 0.1
        usable = duration - 2 * margin
        if usable <= 0:
            margin = 0.0
            usable = duration

        timestamps = [margin + usable * (i + 0.5) / count for i in range(count)]

        extracted: List[Path] = []
        for i, ts in enumerate(timestamps):
            out_jpg = dest_dir / f"frame_{i + 1:02d}.jpg"
            cmd = [
                "ffmpeg",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(seek_source),
                "-vframes",
                "1",
                "-q:v",
                "2",
                "-an",
                str(out_jpg),
                "-y",
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                continue
            if out_jpg.exists() and out_jpg.stat().st_size > 1_000:
                extracted.append(out_jpg)
            else:
                out_jpg.unlink(missing_ok=True)

        return len(extracted) > 0

    finally:
        if remux_tmp is not None:
            remux_tmp.unlink(missing_ok=True)


def extract_frames(src: Path, dest_dir: Path, count: int = 5) -> bool:
    """提取帧：优先使用快速路径，失败时回退到慢速 select 滤镜路径。"""
    is_mp4 = src.suffix.lower() == ".mp4"
    if extract_frames_fast(src, dest_dir, count, need_remux=not is_mp4):
        return True
    print(f"  fast extraction failed for {src.name}, falling back to slow path")
    return extract_jpeg_frames(src, dest_dir, count)


def extract_jpeg_frames(src: Path, dest_dir: Path, count: int = 3) -> bool:
    """从 H265/H264 提取 `count` 张均匀分布的高分辨率 JPEG 帧（慢速回退方案）。

    保留原始视频文件。使用 `-q:v 2` 输出高质量 JPEG。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 通过 ffprobe 获取总帧数
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    try:
        total_frames = int(probe.stdout.strip().split("\n")[0] or 0)
    except (ValueError, IndexError):
        total_frames = 0

    if total_frames < count:
        # 视频太短：提取所有帧
        step = 1
        count = min(count, max(1, total_frames))
        indices = list(range(count))
    else:
        # 均匀分布，跳过首尾几帧
        step = total_frames // (count + 1)
        indices = [step * (i + 1) for i in range(count)]

    extracted: List[Path] = []
    for i, idx in enumerate(indices):
        out_jpg = dest_dir / f"frame_{i + 1:02d}.jpg"
        cmd = [
            "ffmpeg",
            "-i",
            str(src),
            "-vf",
            f"select='eq(n\\,{idx})',format=yuv420p",
            "-vframes",
            "1",
            "-q:v",
            "2",
            "-an",
            str(out_jpg),
            "-y",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if out_jpg.exists() and out_jpg.stat().st_size > 1_000:
            extracted.append(out_jpg)
        else:
            out_jpg.unlink(missing_ok=True)
            # 回退：尝试视频中点位置
            if i == 0:
                cmd = [
                    "ffmpeg",
                    "-ss",
                    "00:00:01.500",
                    "-i",
                    str(src),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    "-an",
                    str(out_jpg),
                    "-y",
                ]
                subprocess.run(cmd, capture_output=True)
                if out_jpg.exists() and out_jpg.stat().st_size > 1_000:
                    extracted.append(out_jpg)

    if not extracted:
        print(f"  ffmpeg error: no JPEG frames extracted")
        return False

    print(f"  extracted {len(extracted)} frames ({total_frames} total)")
    return True


def convert_existing_mp4s(
    output_dir: Path, frame_count: int = 5, workers: int = 6, keep_mp4: bool = False
) -> Tuple[int, int]:
    """从已下载的 MP4 文件中提取高分辨率帧。

    扫描 output_dir 下的 {uuid}/{uuid}_*.mp4，在 MP4 旁边提取帧，
    可选是否保留原 MP4 文件。

    返回 (成功数, 失败数)。
    """
    mp4_files = sorted(output_dir.rglob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files found under {output_dir}")
        return 0, 0

    print(f"Found {len(mp4_files)} MP4 files to convert to frames")

    success, fail = 0, 0
    for i, mp4_path in enumerate(mp4_files, 1):
        uuid_dir = mp4_path.parent
        uuid = uuid_dir.name

        # 如果帧已存在则跳过
        existing_frames = list(uuid_dir.glob("frame_*.jpg"))
        if len(existing_frames) >= frame_count:
            print(f"[SKIP] {uuid[:12]}  {len(existing_frames)} frames already exist")
            success += 1
            continue

        if not extract_frames(mp4_path, uuid_dir, count=frame_count):
            print(f"[FAIL] {uuid[:12]}  frame extraction from MP4 failed")
            fail += 1
            continue

        # 写入 _meta.json
        meta = uuid_dir / "_meta.json"
        meta.write_text(
            json.dumps(
                {
                    "uuid": uuid,
                    "camera": mp4_path.stem.rsplit("_", 1)[-1]
                    if "_" in mp4_path.stem
                    else "unknown",
                    "frames": [f"frame_{j + 1:02d}.jpg" for j in range(frame_count)],
                    "source": "mp4_conversion",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if not keep_mp4:
            mp4_path.unlink(missing_ok=True)

        actual = len(list(uuid_dir.glob("frame_*.jpg")))
        print(f"[ OK ] {uuid[:12]}  → {actual} frames from MP4")
        success += 1

        if i % 10 == 0 or i == len(mp4_files):
            print(f"  --- {i}/{len(mp4_files)}: {success} ok, {fail} failed ---")

    return success, fail


# ── 单个 UUID 处理流程 ──────────────────────────────────────────────────────


def process_uuid(
    uuid: str,
    output_dir: Path,
    camera: str,
    adw_user: str,
    adw_prod_pass: str,
    adw_stg_pass: str,
    frame_mode: bool = False,
    frame_count: int = 3,
) -> bool:
    """下载并处理单个 UUID。成功返回 True。"""
    if frame_mode:
        frame_dir = output_dir / uuid
        existing_frames = list(frame_dir.glob("frame_*.jpg"))
        if len(existing_frames) >= frame_count:
            print(f"[SKIP] {uuid[:12]}  {len(existing_frames)} frames already exist")
            return True
    else:
        mp4_dest = output_dir / uuid / f"{uuid}_{camera}.mp4"
        if mp4_dest.exists() and mp4_dest.stat().st_size > 100_000:
            print(f"[SKIP] {uuid[:12]}  already exists")
            return True
        mp4_dest.unlink(missing_ok=True)

    h265_tmp = output_dir / uuid / f"{uuid}_{camera}.h265"
    h265_tmp.parent.mkdir(parents=True, exist_ok=True)

    # 如果 H265 已存在且非空则跳过重新下载
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

    if frame_mode:
        # 提取高分辨率 JPEG 帧
        if not extract_frames(h265_tmp, frame_dir, count=frame_count):
            print(f"[FAIL] {uuid[:12]}  frame extraction failed")
            # 保留 h265 以便重试
            return False
        # 写入元数据文件
        meta = frame_dir / "_meta.json"
        meta.write_text(
            json.dumps(
                {
                    "uuid": uuid,
                    "camera": camera,
                    "frames": [f"frame_{i + 1:02d}.jpg" for i in range(frame_count)],
                    "total_frames_in_video": frame_count,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"[ OK ] {uuid[:12]}  → {frame_dir.name} ({len(list(frame_dir.glob('frame_*.jpg')))} frames)"
        )
        return True

    # 重编码 H265 → 720p H264 MP4
    if not convert_to_mp4(h265_tmp, mp4_dest):
        print(f"[FAIL] {uuid[:12]}  ffmpeg conversion failed")
        h265_tmp.unlink(missing_ok=True)
        return False

    h265_tmp.unlink(missing_ok=True)
    print(f"[ OK ] {uuid[:12]}  → {mp4_dest.name}")
    return True


# ── 入口 ─────────────────────────────────────────────────────────────────────


def re_extract_from_raw(output_dir: Path, frame_count: int = 5) -> Tuple[int, int]:
    """从已下载的 H265/H264 文件重新提取帧。

    扫描 output_dir 下的 {uuid}/*.h265 或 *.h264，清除旧帧后重新提取。
    如果 _meta.json 存在则更新。

    返回 (成功数, 失败数)。
    """
    raw_files = sorted(
        list(output_dir.rglob("*.h265")) + list(output_dir.rglob("*.h264"))
    )
    if not raw_files:
        print(f"No H265/H264 files found under {output_dir}")
        return 0, 0

    print(f"Found {len(raw_files)} raw video files to re-extract frames from")

    success, fail = 0, 0
    for i, raw_path in enumerate(raw_files, 1):
        uuid_dir = raw_path.parent
        uuid = uuid_dir.name

        existing = list(uuid_dir.glob("frame_*.jpg"))
        if len(existing) >= frame_count:
            print(f"[SKIP] {uuid[:12]}  {len(existing)} frames already exist")
            success += 1
            continue

        for old in uuid_dir.glob("frame_*.jpg"):
            old.unlink()

        ok = extract_frames_fast(raw_path, uuid_dir, count=frame_count, need_remux=True)
        actual = len(list(uuid_dir.glob("frame_*.jpg")))

        if ok:
            meta = uuid_dir / "_meta.json"
            if meta.exists():
                data = json.loads(meta.read_text(encoding="utf-8"))
                data["frames"] = [f"frame_{j + 1:02d}.jpg" for j in range(actual)]
                data["total_frames_in_video"] = actual
                data["extraction_method"] = "fast_seek"
                meta.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        status = " OK " if ok else "FAIL"
        print(f"[{status}] {uuid[:12]}  → {actual} frames")
        if ok:
            success += 1
        else:
            fail += 1

        if i % 10 == 0 or i == len(raw_files):
            print(f"  --- {i}/{len(raw_files)}: {success} ok, {fail} failed ---")

    return success, fail


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
        description="从 NIO ADW 下载视频并提取帧 / 重编码为 720p MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--uuid-file",
        required=False,
        default=None,
        help="UUID 列表文件路径，每行一个 UUID（空行和 # 注释会被忽略）",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出根目录：{output_dir}/{uuid}/frame_*.jpg 或 {uuid}_{camera}.mp4",
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"相机流名称（默认：{DEFAULT_CAMERA}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="并行下载/编码线程数（默认：6）",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="仅处理前 N 个 UUID（用于测试）",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="重编码为 720p H264 MP4 而非提取帧（旧默认模式）",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=5,
        help="每个视频提取的帧数（默认：5）",
    )
    parser.add_argument(
        "--adw-user",
        default=os.environ.get("ADW_USER", "dayun.shen"),
        help="ADW 用户名（默认：$ADW_USER 或 'dayun.shen'）",
    )
    parser.add_argument(
        "--adw-prod-pass",
        default=os.environ.get("ADW_PROD_PASS", ""),
        help="ADW 生产环境密码（默认：$ADW_PROD_PASS）",
    )
    parser.add_argument(
        "--adw-stg-pass",
        default=os.environ.get("ADW_STG_PASS", ""),
        help="ADW 预发布环境密码（默认：$ADW_STG_PASS）",
    )
    parser.add_argument(
        "--from-mp4",
        action="store_true",
        help="从 --output-dir 中已有的 MP4 文件提取帧（不下载）",
    )
    parser.add_argument(
        "--keep-mp4",
        action="store_true",
        help="提取帧后保留原 MP4 文件（配合 --from-mp4 使用）",
    )
    parser.add_argument(
        "--re-extract",
        action="store_true",
        help="从 --output-dir 中已有的 H265/H264 文件重新提取帧（不下载）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --from-mp4：从已有 MP4 文件提取帧，无需下载
    if args.from_mp4:
        success, fail = convert_existing_mp4s(
            output_dir,
            frame_count=args.frame_count,
            workers=args.workers,
            keep_mp4=args.keep_mp4,
        )
        print(f"\nDone: {success} success, {fail} failed")
        return

    # --re-extract：从已有 H265/H264 文件重新提取帧，无需下载
    if args.re_extract:
        success, fail = re_extract_from_raw(
            output_dir,
            frame_count=args.frame_count,
        )
        print(f"\nDone: {success} success, {fail} failed")
        return

    if not args.uuid_file:
        parser.error("不使用 --from-mp4 或 --re-extract 时必须指定 --uuid-file。")

    if not args.adw_prod_pass:
        parser.error(
            "ADW_PROD_PASS 未设置。请通过环境变量导出或 --adw-prod-pass 传入。"
        )

    uuids = read_uuids(Path(args.uuid_file))
    if args.sample:
        uuids = uuids[: args.sample]

    total = len(uuids)
    frame_mode = not args.mp4
    mode_str = "extract frames" if frame_mode else "MP4 encode"
    print(f"Downloading {total} videos → {output_dir}")
    print(f"  UUID file : {args.uuid_file}")
    print(f"  Camera    : {args.camera}")
    print(f"  Workers   : {args.workers}")
    if frame_mode:
        print(f"  Mode      : {mode_str}, {args.frame_count} frames per clip")
    else:
        print(
            f"  Encoding  : scale={ENCODE_SCALE}, libx264 crf={ENCODE_CRF} preset=veryfast"
        )
    print()

    success, fail = 0, 0
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_uuid,
                uuid,
                output_dir,
                args.camera,
                args.adw_user,
                args.adw_prod_pass,
                args.adw_stg_pass,
                frame_mode,
                args.frame_count,
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
