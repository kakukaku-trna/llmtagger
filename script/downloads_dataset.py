import os
from pathlib import Path

from dataset_management_sdk.v2.alpha.ds_static_env import set_access_token
from dataset_management_sdk.v2.alpha.ds_static_env import set_environment
from dataset_management_sdk.v2.alpha.ds_static_env import get_niofs_util

from dataset_management_sdk.v2.alpha.ds_data import DatasetData
from dataset_management_sdk.v2.alpha.ds_set import DatasetSet
from dataset_management_sdk.v2.alpha.ds_file import DatasetFile
from dataset_management_sdk.v2.alpha.ds_pool import DatasetPool


def download_ceph_file(remote_path, local_path):
    """使用niofs下载ceph文件到本地"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        # 使用SDK的niofs工具下载
        niofs_util = get_niofs_util()
        bucket, key = niofs_util.split(remote_path)
        niofs_util.download(bucket, key, local_path)
        print(f"  下载成功: {os.path.basename(local_path)}")
        return True
    except Exception as e:
        print(f"  下载失败: {remote_path}")
        print(f"  错误: {e}")
        return False


def should_download_image(file: DatasetFile) -> bool:
    """判断是否是图片或视频文件"""
    raw_key = file.raw_key() or ""

    download_extensions = [
        '.jpg', '.jpeg', '.png', '.bmp', '.gif',   # 图片
        '.mp4', '.avi', '.mov', '.mkv', '.ts',      # 视频
    ]

    raw_key_lower = raw_key.lower()
    for ext in download_extensions:
        if raw_key_lower.endswith(ext):
            return True

    return False


def scan_existing_data(base_dir):
    """
    扫描已有数据目录，返回 {data_id: set(已存在的文件名)} 的映射
    
    Args:
        base_dir: 原数据目录路径
        
    Returns:
        dict: {data_id: set(filenames)}
    """
    existing_data = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return existing_data
    
    for item in base_path.iterdir():
        if item.is_dir():
            data_id = item.name
            files = {f.name for f in item.iterdir() if f.is_file()}
            existing_data[data_id] = files
    
    return existing_data


# 设置访问token（用于权限验证）
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6Imp3dCJ9.eyJleHAiOjU0MDgwNjMwOTEsImlhdCI6MTc3OTI2MzA5MSwiZGF0YSI6eyJ1c2VybmFtZSI6InR1b3l1LnlhbmciLCJhd3NfYWNjZXNzX2tleV9pZCI6IjllMWVkNzIzOTQ1Mzc2YjAiLCJhd3Nfc2VjcmV0X2FjY2Vzc19rZXkiOiJhYWVmZmU0ODA4MGU0YWM0YmJmMWVjOWFhYmM3ZGU5YyJ9fQ.QlofshD7XIkenzzXcHTdlRzVcu4EV2HeVMIc3PT68v4"


def download_dataset_images(
    dataset_nickname="da_mining_etc_20260520_frame",
    env="prod",
    base_dir=None,
    supplement_dir=None,
    max_data_count=None,
):
    """
    增量下载数据集图片
    
    - 新 data_id -> 下载到 supplement_dir
    - 已有 data_id + 缺失文件 -> 在原目录补全缺失文件
    - 已有 data_id + 文件完整 -> 跳过
    
    Args:
        dataset_nickname: 数据集名称
        env: 环境 (prod 或 stg)
        base_dir: 原数据目录，默认使用 dataset_nickname
        supplement_dir: 补充数据存放目录，默认使用 dataset_nickname + "_supplement"
        max_data_count: 最多处理多少条数据 (None表示全部)
    """
    
    # 如果未指定目录，使用默认值
    if base_dir is None:
        base_dir = dataset_nickname
    if supplement_dir is None:
        supplement_dir = f"{dataset_nickname}_supplement"
    
    # 扫描已有数据
    print(f"扫描已有数据: {base_dir}")
    existing_data = scan_existing_data(base_dir)
    print(f"  发现 {len(existing_data)} 个已有 data_id")
    
    # 设置token和环境
    set_access_token(ACCESS_TOKEN)
    set_environment(env)
    
    # 获取pool_id - 数据集完整名称通常是 name-version 格式
    # 如果没有版本号，默认使用 "latest"
    if "-" in dataset_nickname:
        name, version = dataset_nickname.rsplit("-", 1)
    else:
        name = dataset_nickname
        version = "latest"
    
    # TODO: 需要获取pool_id
    # 这里需要先实现获取pool_id的逻辑
    # pool_id = query_pool_id(dataset_nickname, env)
    # 暂时硬编码一个示例pool_id，实际使用时需要替换
    pool_id = "your_pool_id_here"
    
    print(f"Pool ID: {pool_id}")
    print(f"数据集名称: {name}, 版本: {version}")
    
    # 创建数据池对象
    data_pool = DatasetPool(pool_id)
    
    # 获取数据集
    dataset = data_pool.fetch_dataset(name, version)
    if dataset is None:
        print(f"未找到数据集: {name} (版本: {version})")
        return
    
    print(f"数据集昵称: {dataset.dataset_nickname}")
    print(f"数据集UUID: {dataset.dataset_uuid}")
    
    # 获取数据列表 - 直接获取所有数据
    # 由于fetch_data_count有问题，我们直接分页获取所有数据
    data_list = []
    offset = 0
    limit = 100  # 每批获取100条
    
    while True:
        batch = dataset.fetch_data_list(offset=offset, limit=limit)
        if not batch:
            break
        data_list.extend(batch)
        offset += len(batch)
        print(f"  已获取: {len(data_list)} 条数据")
        
        # 如果获取的数量小于limit，说明已经获取完毕
        if len(batch) < limit:
            break
    
    print(f"数据总条数: {len(data_list)}")
    
    # 限制处理数量
    if max_data_count is not None:
        data_list = data_list[:max_data_count]
    
    # 创建补充目录
    os.makedirs(supplement_dir, exist_ok=True)
    
    # 统计变量
    total_skipped = 0          # 完整跳过的 data_id 数量
    total_new_data_ids = 0     # 全新下载的 data_id 数量
    total_supplement_files = 0  # 补充目录中下载的文件数
    total_patched_files = 0    # 原目录补全的文件数
    total_patch_data_ids = 0   # 需要补全的 data_id 数量
    
    # 遍历每条数据
    _debug_printed = False  # 只打印第一条的结构
    for data_item in data_list:
        data_id = data_item.data_id

        # 调试：打印第一条数据的原始结构
        if not _debug_printed:
            print(f"\n[DEBUG] 第一条数据结构:")
            print(f"  data_id: {data_id}")
            print(f"  data_type: {getattr(data_item, 'data_type', 'N/A')}")
            print(f"  public_files: {getattr(data_item, 'public_files', 'N/A')}")
            print(f"  private_files: {getattr(data_item, 'private_files', 'N/A')}")
            print(f"  public_metas: {getattr(data_item, 'public_metas', 'N/A')}")
            print(f"  所有属性: {[k for k in vars(data_item).keys()]}")
            _debug_printed = True

        # 获取数据详情和文件列表
        # 从data_item中获取public_files
        file_list = []
        if hasattr(data_item, 'public_files') and data_item.public_files:
            for file_info in data_item.public_files:
                ds_file = DatasetFile(
                    raw_key=file_info.get("key", ""),
                    path=file_info.get("val", ""),
                )
                file_list.append(ds_file)
        
        # 筛选出图片文件
        image_files = [f for f in file_list if should_download_image(f)]
        
        if not image_files:
            continue
        
        # 判断是全新数据还是已有数据
        if data_id not in existing_data:
            # ===== 全新 data_id -> 下载到补充目录 =====
            target_dir = os.path.join(supplement_dir, data_id)
            os.makedirs(target_dir, exist_ok=True)
            
            data_image_count = 0
            for file in image_files:
                ceph_file = file.raw_key()
                if not ceph_file:
                    continue
                
                file_name = os.path.basename(ceph_file)
                local_file = os.path.join(target_dir, file_name)
                
                remote_path = file._path
                if remote_path:
                    if download_ceph_file(remote_path, local_file):
                        data_image_count += 1
                        total_supplement_files += 1
            
            if data_image_count > 0:
                total_new_data_ids += 1
                print(f"  [新数据] {data_id}: 下载 {data_image_count} 张图片到补充目录")
        else:
            # ===== 已有 data_id -> 检查缺失文件 =====
            existing_files = existing_data[data_id]
            missing_files = []
            
            for file in image_files:
                ceph_file = file.raw_key()
                if not ceph_file:
                    continue
                
                file_name = os.path.basename(ceph_file)
                if file_name not in existing_files:
                    missing_files.append(file)
            
            if not missing_files:
                # 文件完整，跳过
                total_skipped += 1
                continue
            
            # 有缺失文件，在原目录补全
            target_dir = os.path.join(base_dir, data_id)
            os.makedirs(target_dir, exist_ok=True)
            
            patch_count = 0
            for file in missing_files:
                ceph_file = file.raw_key()
                file_name = os.path.basename(ceph_file)
                local_file = os.path.join(target_dir, file_name)
                
                remote_path = file._path
                if remote_path:
                    if download_ceph_file(remote_path, local_file):
                        patch_count += 1
                        total_patched_files += 1
            
            if patch_count > 0:
                total_patch_data_ids += 1
                print(f"  [补全] {data_id}: 补全 {patch_count} 张缺失图片")
    
    # ===== 输出统计报告 =====
    print("\n" + "="*70)
    print("下载统计报告")
    print("="*70)
    print(f"  数据总条数: {len(data_list)}")
    print(f"  已有 data_id 数量: {len(existing_data)}")
    print(f"  新 data_id 数量: {total_new_data_ids} (存放于 {supplement_dir})")
    print(f"  新下载文件数: {total_supplement_files}")
    print(f"  补全 data_id 数量: {total_patch_data_ids}")
    print(f"  补全文件数: {total_patched_files}")
    print(f"  完整跳过: {total_skipped}")
    print(f"  原数据目录: {os.path.abspath(base_dir)}")
    print(f"  补充目录: {os.path.abspath(supplement_dir)}")
    print("="*70)


if __name__ == '__main__':
    print("=============================== 增量下载数据集图片 =====================")
    
    # 配置参数
    ENV = "prod"  # 环境: prod 或 stg
    MAX_DATA_COUNT = None  # 最多处理多少条数据，None表示全部
    
    # 要下载的数据集列表
    # 注释掉旧数据集，只下载新数据集
    DATASET_LIST = [
        # "da_mining_etc_20260520_frame",
        # "da_mining_etc_only_20260520_frame",
        # "da_mining_car_etc_20260520_frame",
        # "da_mining_truck_etc_20260520_frame",
        # "da_mining_etc_manual_20260520_frame",
        # "da_mining_manual_20260520_frame",
        # "da_mining_self_service_etc_20260520_frame",
        # "da_mining_lane_closed_20260521_frame",
        # "da_mining_car_truck_etc_train_20260526",
        # "da_mining_free_train_20260526",
        "da_mining_two_way_single_lane_20260610_subclip",   # 双向单车道数据集
    ]

    # 数据存放根目录
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    # 循环处理每个数据集
    for dataset_name in DATASET_LIST:
        print("\n" + "="*70)
        print(f"开始处理数据集: {dataset_name}")
        print("="*70)

        try:
            download_dataset_images(
                dataset_nickname=dataset_name,
                env=ENV,
                base_dir=os.path.join(DATA_ROOT, dataset_name),
                supplement_dir=os.path.join(DATA_ROOT, f"{dataset_name}_supplement"),
                max_data_count=MAX_DATA_COUNT,
            )
        except Exception as e:
            print(f"\n处理数据集 {dataset_name} 时出错: {e}")
            print("继续处理下一个数据集...")
        
        print("\n" + "="*70)
        print(f"数据集 {dataset_name} 处理完成")
        print("="*70)
    
    print("\n\n所有数据集处理完成！")
