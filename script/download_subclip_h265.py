"""
下载 SUBCLIP 类型数据集的 H264/H265 视频文件

【背景】
dataset_management_sdk 支持两种数据类型：
  - FRAME：每条数据对应一帧图像，文件路径存储在 public_files 字段，可直接通过 niofs 下载。
  - SUBCLIP：每条数据对应一段原始录像片段，public_files 为 None，文件不通过数据集管理系统存储，
    而是存放在 ADW（Autonomous Driving Warehouse，自动驾驶数据仓库）中。

【为什么 downloads_dataset.py 对 SUBCLIP 数据集无效】
downloads_dataset.py 依赖 public_files 字段获取文件路径，SUBCLIP 数据该字段始终为 None，
因此脚本找不到任何可下载的文件（0 downloads）。

【本脚本的实现方案】
1. 通过 dataset_management_sdk 获取数据集的所有 clip_id（SUBCLIP 数据唯一有效的字段）。
2. 对每个 clip_id，使用 da_data_sdk.ClipDownloader 向 ADW 发起下载请求：
   - ADW 存储了原始车辆录像，按 clip_id 索引，支持按传感器（SENSORS）筛选。
   - SENSORS=["FW"] 对应前向广角相机（/camera/front/main），下载后文件名包含 Front120。
   - ClipDownloader 需要 ADW_USER / ADW_PASS（生产环境）及 ADW_USER_STG / ADW_PASS_STG
     （预发布环境）四个环境变量，缺少任意一个会抛出 RuntimeError。
3. 下载到临时目录后，将 *.h264 / *.h265 文件移动到最终目录 data/{DATASET_NAME}/{clip_id}/，
   并清理临时目录。
4. 跳过已下载的 clip_id，支持断点续传。

【注意事项】
- 部分 clip 永久失败：clip 已删除、无访问权限（EU 数据）、旧 ES8 车型只有 .mkv 格式等，
  这些属于数据集本身的问题，无法修复。
- 避免并发运行同一脚本，否则临时目录可能产生冲突。
"""

import os
import shutil
from pathlib import Path

os.environ['ADW_USER'] = 'dayun.shen'
os.environ['ADW_PASS'] = 'ORSGVU6EF9'
os.environ['ADW_USER_STG'] = 'dayun.shen'
os.environ['ADW_PASS_STG'] = 'G4IDTL2YJW'

from dataset_management_sdk.v2.alpha.ds_static_env import set_access_token, set_environment
from dataset_management_sdk.v2.alpha.ds_pool import DatasetPool
from da_data_sdk.clip import ClipDownloader

ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6Imp3dCJ9.eyJleHAiOjU0MDgwNjMwOTEsImlhdCI6MTc3OTI2MzA5MSwiZGF0YSI6eyJ1c2VybmFtZSI6InR1b3l1LnlhbmciLCJhd3NfYWNjZXNzX2tleV9pZCI6IjllMWVkNzIzOTQ1Mzc2YjAiLCJhd3Nfc2VjcmV0X2FjY2Vzc19rZXkiOiJhYWVmZmU0ODA4MGU0YWM0YmJmMWVjOWFhYmM3ZGU5YyJ9fQ.QlofshD7XIkenzzXcHTdlRzVcu4EV2HeVMIc3PT68v4"

# 要下载的相机（FW = 前向广角 Front120）
SENSORS = ["FW"]

# 数据集名称
DATASET_NAME = "da_mining_two_way_single_lane_20260610_subclip"

# 输出目录
DATA_ROOT = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_ROOT / DATASET_NAME


def download_clip_h265(clip_id: str, output_dir: Path) -> bool:
    """下载单条clip的h265/h264文件，保存到output_dir/{clip_id}/"""
    clip_output = output_dir / clip_id

    # 检查是否已下载
    existing = list(clip_output.glob("*.h264")) + list(clip_output.glob("*.h265"))
    if existing:
        print(f"  [跳过] {clip_id}: 已有 {len(existing)} 个文件")
        return True

    import tempfile
    tmp_dir = output_dir / f"_tmp_{clip_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloader = ClipDownloader(clip_id)
        downloader.add_sensors(SENSORS)
        downloader.download(str(tmp_dir))

        # 找所有h264/h265文件，移动到最终目录
        h_files = list(tmp_dir.rglob("*.h264")) + list(tmp_dir.rglob("*.h265"))
        if not h_files:
            print(f"  [警告] {clip_id}: 未找到h264/h265文件")
            return False

        clip_output.mkdir(parents=True, exist_ok=True)
        for f in h_files:
            dest = clip_output / f.name
            shutil.move(str(f), str(dest))
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  [成功] {clip_id}: {f.name} ({size_mb:.1f} MB)")

        return True

    except Exception as e:
        print(f"  [失败] {clip_id}: {e}")
        return False
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


def main():
    set_access_token(ACCESS_TOKEN)
    set_environment("prod")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"下载相机: {SENSORS}")

    # 获取数据集所有clip_id
    pool = DatasetPool("1691331358571270144")
    dataset = pool.fetch_dataset(DATASET_NAME, "latest")

    data_list = []
    offset = 0
    limit = 100
    while True:
        batch = dataset.fetch_data_list(offset=offset, limit=limit)
        if not batch:
            break
        data_list.extend(batch)
        offset += len(batch)
        print(f"  已获取: {len(data_list)} 条")
        if len(batch) < limit:
            break

    print(f"\n共 {len(data_list)} 条数据，开始下载...\n")

    success = 0
    failed = 0
    skipped = 0

    for i, item in enumerate(data_list):
        clip_id = item.clip_id
        print(f"[{i+1}/{len(data_list)}] {clip_id}")

        existing = list((OUTPUT_DIR / clip_id).glob("*.h264")) + \
                   list((OUTPUT_DIR / clip_id).glob("*.h265"))
        if existing:
            skipped += 1
            print(f"  [跳过] 已存在 {len(existing)} 个文件")
            continue

        ok = download_clip_h265(clip_id, OUTPUT_DIR)
        if ok:
            success += 1
        else:
            failed += 1

    print("\n" + "="*60)
    print("下载统计")
    print("="*60)
    print(f"  总条数: {len(data_list)}")
    print(f"  成功:   {success}")
    print(f"  跳过:   {skipped}")
    print(f"  失败:   {failed}")
    print(f"  输出:   {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
