#!/usr/bin/env python3
"""Update eval_benchmark.csv and sync benchmark videos.

Changes applied:
1. Merge "阴天" into "晴天".
2. Add the "路面积水" column if missing.
3. Append new clips for 沙尘 / 非机动车道 / 路面积水.
4. Convert or copy the selected source clips into MP4 under eval_videos.
"""

from __future__ import annotations

import csv
import datetime as dt
import shutil
import subprocess
from pathlib import Path

CSV_PATH = Path("/home/huajiang.sun/data/eval_benchmark.csv")
EVAL_VIDEOS_DIR = Path("/home/huajiang.sun/data/eval_videos")
TXT_DIR = Path("/home/huajiang.sun/data/txt")
DATA_DIR = Path("/home/huajiang.sun/data")

NEW_COLUMN = "路面积水"
MP4_SCALE = "1280:720"
MP4_CRF = "38"

CANDIDATES = [
    {
        "name": "沙尘",
        "txt": TXT_DIR / "沙尘-150clip.txt",
        "source_dir": DATA_DIR / "沙尘",
        "target_label": ("天气", "沙尘"),
        "sample_size": 25,
        "selection_txt": TXT_DIR / "沙尘-eval25.txt",
    },
    {
        "name": "非机动车道",
        "txt": TXT_DIR / "da_mining_bicycle_lane_20260616_subclip_eval_uuid.txt",
        "source_dir": DATA_DIR / "da_mining_bicycle_lane_20260616_subclip",
        "target_label": ("非机动车道", "是"),
        "sample_size": 25,
        "selection_txt": TXT_DIR / "非机动车道-eval25.txt",
    },
    {
        "name": "路面积水",
        "txt": TXT_DIR / "da_mining_RoadWater_20260615_subclip_eval_uuid.txt",
        "source_dir": DATA_DIR / "da_mining_RoadWater_20260615_subclip",
        "target_label": ("路面积水", "是"),
        "sample_size": 21,
        "selection_txt": TXT_DIR / "路面积水-eval21.txt",
    },
]


def load_rows() -> tuple[list[str], list[dict[str, str]]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if not headers:
        raise RuntimeError(f"Empty CSV: {CSV_PATH}")
    return headers, rows


def ensure_headers(headers: list[str]) -> list[str]:
    if NEW_COLUMN not in headers:
        idx = headers.index("非机动车道") + 1
        headers = headers[:idx] + [NEW_COLUMN] + headers[idx:]
    return headers


def merge_weather(rows: list[dict[str, str]]) -> int:
    merged = 0
    for row in rows:
        if row.get("天气") == "阴天":
            row["天气"] = "晴天"
            merged += 1
    return merged


def normalize_rows(headers: list[str], rows: list[dict[str, str]]) -> None:
    for row in rows:
        for header in headers:
            if header == "clip_id":
                continue
            if row.get(header, "") == "":
                row[header] = "null"


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_source_video(source_dir: Path, clip_id: str) -> Path | None:
    clip_dir = source_dir / clip_id
    if not clip_dir.is_dir():
        return None
    candidates: list[Path] = []
    for pattern in ("*.h265", "*.h264", "*.mp4", "*.mkv"):
        candidates.extend(sorted(clip_dir.glob(pattern)))
    return candidates[0] if candidates else None


def select_ids(
    rows: list[dict[str, str]],
    source_dir: Path,
    candidate_ids: list[str],
    sample_size: int,
) -> list[str]:
    used = {row["clip_id"] for row in rows}
    selected: list[str] = []
    for clip_id in candidate_ids:
        if clip_id in used:
            continue
        if find_source_video(source_dir, clip_id) is None:
            continue
        selected.append(clip_id)
        used.add(clip_id)
        if len(selected) >= sample_size:
            break
    return selected


def append_rows(
    headers: list[str],
    rows: list[dict[str, str]],
    selected_map: dict[str, list[str]],
) -> int:
    config_by_name = {cfg["name"]: cfg for cfg in CANDIDATES}
    added = 0
    for name, clip_ids in selected_map.items():
        label_col, label_val = config_by_name[name]["target_label"]
        for clip_id in clip_ids:
            row = {header: "null" for header in headers}
            row["clip_id"] = clip_id
            row[label_col] = label_val
            rows.append(row)
            added += 1
    return added


def backup_csv() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CSV_PATH.with_name(f"{CSV_PATH.name}.bak_{stamp}")
    backup.write_text(CSV_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def write_csv(headers: list[str], rows: list[dict[str, str]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_selection_txt(path: Path, clip_ids: list[str]) -> Path:
    path.write_text("\n".join(clip_ids) + "\n", encoding="utf-8")
    return path


def convert_to_mp4(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", str(src),
        "-vf", f"scale={MP4_SCALE}",
        "-vcodec", "libx264", "-crf", MP4_CRF, "-preset", "veryfast",
        "-an", str(dest), "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size < 100_000:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {src}: {result.stderr[-300:]}")


def sync_eval_videos(selected_map: dict[str, list[str]]) -> int:
    config_by_name = {cfg["name"]: cfg for cfg in CANDIDATES}
    synced = 0
    for name, clip_ids in selected_map.items():
        source_dir = Path(config_by_name[name]["source_dir"])
        for clip_id in clip_ids:
            src = find_source_video(source_dir, clip_id)
            if src is None:
                raise RuntimeError(f"Missing source video for {clip_id} in {source_dir}")
            camera = src.stem.replace(f"{clip_id}_", "", 1)
            dest = EVAL_VIDEOS_DIR / clip_id / f"{clip_id}_{camera}.mp4"
            if dest.exists() and dest.stat().st_size > 100_000:
                continue
            if src.suffix.lower() == ".mp4":
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            else:
                convert_to_mp4(src, dest)
            synced += 1
    return synced


def main() -> None:
    headers, rows = load_rows()
    headers = ensure_headers(headers)
    normalize_rows(headers, rows)
    merged = merge_weather(rows)

    selected_map: dict[str, list[str]] = {}
    for cfg in CANDIDATES:
        selected = select_ids(
            rows=rows,
            source_dir=Path(cfg["source_dir"]),
            candidate_ids=read_ids(Path(cfg["txt"])),
            sample_size=int(cfg["sample_size"]),
        )
        if len(selected) < int(cfg["sample_size"]):
            raise RuntimeError(
                f"{cfg['name']} only selected {len(selected)} clips, expected {cfg['sample_size']}"
            )
        selected_map[str(cfg["name"])] = selected

    added = append_rows(headers, rows, selected_map)
    backup = backup_csv()
    write_csv(headers, rows)

    selection_files = {
        str(cfg["name"]): write_selection_txt(Path(cfg["selection_txt"]), selected_map[str(cfg["name"])])
        for cfg in CANDIDATES
    }
    synced = sync_eval_videos(selected_map)

    print(f"Backup CSV     : {backup}")
    print(f"Merged weather : {merged}")
    print(f"Added rows     : {added}")
    print(f"Synced MP4     : {synced}")
    print(f"New row count  : {len(rows)}")
    for cfg in CANDIDATES:
        name = str(cfg["name"])
        print(f"{name:8} : {len(selected_map[name])} clips, txt={selection_files[name]}")


if __name__ == "__main__":
    main()
