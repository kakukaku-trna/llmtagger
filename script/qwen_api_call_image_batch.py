#!/usr/bin/env python3
"""
Qwen-VL 批量 API 调用脚本（统一版）
支持：并行处理、断点续传、速率限制、智能重试、结果扁平化

用法示例：
  # 基础模式（< 500 张，无断点续传）
  python qwen_api_call_image_batch.py --prompt p.md --input_dir imgs

  # 生产模式（> 500 张，启用所有功能）
  python qwen_api_call_image_batch.py --prompt p.md --input_dir imgs --enable-progress --qps 10 --workers 4

  # 强制重新开始
  python qwen_api_call_image_batch.py --prompt p.md --input_dir imgs --enable-progress --no-resume
"""

import os
import sys
import base64
import glob
import json
import re
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import dashscope
from dashscope import MultiModalConversation


class ThreadSafeLogger:
    """线程安全的日志记录器"""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, message):
        with self.lock:
            self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log_file.flush()

    def close(self):
        with self.lock:
            self.log_file.close()


class ProgressTracker:
    """追踪处理进度，支持断点续传（可选）"""
    def __init__(self, progress_file=None):
        self.progress_file = progress_file
        self.processed = set()
        if progress_file:
            self.load()

    def load(self):
        """从文件加载已处理的文件列表"""
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    self.processed = set(json.load(f))
            except Exception as e:
                print(f"警告: 无法加载进度文件: {e}")

    def save(self):
        """保存进度"""
        if self.progress_file:
            with open(self.progress_file, 'w') as f:
                json.dump(list(self.processed), f)

    def mark_done(self, image_path):
        """标记文件已处理"""
        self.processed.add(image_path)
        self.save()

    def is_done(self, image_path):
        """检查文件是否已处理"""
        return image_path in self.processed

    def get_pending(self, all_files):
        """获取待处理的文件"""
        return [f for f in all_files if not self.is_done(f)]


class RateLimiter:
    """简单的速率限制器（可选）"""
    def __init__(self, qps=None):
        """初始化 QPS (queries per second) 限制"""
        self.qps = qps
        if qps:
            self.min_interval = 1.0 / qps
            self.last_time = time.time()
            self.lock = threading.Lock()
        else:
            self.min_interval = None

    def acquire(self):
        """获取一个请求配额"""
        if not self.qps:
            return
        with self.lock:
            elapsed = time.time() - self.last_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_time = time.time()


def extract_json_from_text(text):
    """从文本中提取 JSON 对象"""
    text = text.strip()

    # 移除 markdown 代码块标记
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


def image_to_base64(image_path):
    """将图片文件转为 base64 字符串"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def call_qwen_vl_with_image(image_path, prompt, max_retries=3, initial_delay=1):
    """调用 Qwen-VL API，支持指数退避重试"""
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{image_to_base64(image_path)}"},
                {"text": prompt},
            ],
        }
    ]

    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = MultiModalConversation.call(
                model="qwen3.6-plus",
                messages=messages,
                temperature=0.000,
                max_tokens=2048,
                thinking_budget=4096,
                enable_thinking=True,
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise Exception(f"API 错误: {response.code} - {response.message}")

        except Exception as e:
            if attempt < max_retries:
                print(f"  第 {attempt} 次调用失败，{delay} 秒后重试: {e}", flush=True)
                time.sleep(delay)
                delay *= 2  # 指数退避
            else:
                raise Exception(f"API 调用在 {max_retries} 次重试后仍然失败: {e}")


def process_single_image(image_path, data_id, text_prompt, stats, stats_lock, rate_limiter=None):
    """处理单张图片的函数"""
    time_start = datetime.now()
    result = {
        "data_id": data_id,
        "image_path": image_path,
        "result": None,
        "text": None,
        "mask": None,
        "line": None,
        "error": None,
    }

    # 速率限制
    if rate_limiter:
        rate_limiter.acquire()

    try:
        print(f"[{data_id}] 处理中...", flush=True)
        response = call_qwen_vl_with_image(image_path, text_prompt, max_retries=3)
        parsed_json = extract_json_from_text(response)

        if parsed_json:
            result["result"] = parsed_json.get("result")
            result["text"] = parsed_json.get("text")
            result["mask"] = parsed_json.get("mask")
            result["line"] = parsed_json.get("line")
            print(f"[{data_id}] ✓ 成功", flush=True)
            with stats_lock:
                stats["success"] += 1
        else:
            error_msg = "无法解析 JSON"
            print(f"[{data_id}] ✗ {error_msg}", flush=True)
            result["error"] = error_msg
            with stats_lock:
                stats["failed"] += 1

    except Exception as e:
        error_msg = str(e)
        print(f"[{data_id}] ✗ 错误: {error_msg}", flush=True)
        result["error"] = error_msg
        with stats_lock:
            stats["failed"] += 1

    elapsed = (datetime.now() - time_start).total_seconds()
    result["elapsed_seconds"] = elapsed

    return result


def read_prompt_from_file(file_path):
    """读取 prompt 文件内容"""
    if not os.path.exists(file_path):
        print(f"错误: prompt 文件不存在: {file_path}")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen-VL 批量 API 调用脚本（统一版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基础模式（小规模，无进度追踪）
  python qwen_api_call_image_batch.py --prompt prompt.md --input_dir ./images

  # 生产模式（大规模，启用进度追踪）
  python qwen_api_call_image_batch.py --prompt prompt.md --input_dir ./images \\
    --enable-progress --workers 4 --qps 10

  # 强制重新开始
  python qwen_api_call_image_batch.py --prompt prompt.md --input_dir ./images \\
    --enable-progress --no-resume
        """
    )

    # 必需参数
    parser.add_argument("--prompt", type=str, required=True, help="prompt 文件路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录路径")

    # 可选参数：性能
    parser.add_argument("--workers", type=int, default=4, help="并行处理线程数（默认: 4）")
    parser.add_argument("--qps", type=int, default=None, help="API 速率限制（queries/second）")

    # 可选参数：输出
    parser.add_argument("--log", type=str, default="result.log", help="日志文件路径")
    parser.add_argument("--result_json", type=str, default="result.json", help="结果 JSON 文件路径")

    # 可选参数：进度追踪
    parser.add_argument(
        "--enable-progress",
        action="store_true",
        help="启用断点续传功能（用于大规模处理）"
    )
    parser.add_argument(
        "--progress_file",
        type=str,
        default=".progress.json",
        help="进度文件路径（仅在 --enable-progress 时使用，默认: .progress.json）"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="跳过断点续传，从头开始处理"
    )

    args = parser.parse_args()

    # 初始化日志
    logger = ThreadSafeLogger(args.log)
    sys.stdout = logger
    sys.stderr = logger

    # 配置 API Key
    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-3d1aade4fc12419ea0101c5128466658")

    # 读取 prompt
    text_prompt = read_prompt_from_file(args.prompt)

    # 获取输入文件（支持递归搜索）
    jpg_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    jpg_files = sorted(jpg_files)

    if not jpg_files:
        print(f"错误: 在 {args.input_dir} 中没有找到 jpg 文件")
        logger.close()
        sys.exit(1)

    # 初始化进度追踪（如果启用）
    progress_file = args.progress_file if args.enable_progress else None
    progress_tracker = ProgressTracker(progress_file)

    # 获取待处理文件
    if args.enable_progress and args.no_resume:
        progress_tracker.processed.clear()
        pending_files = jpg_files
        print(f"选项: 跳过断点续传，从头开始处理")
    elif args.enable_progress:
        pending_files = progress_tracker.get_pending(jpg_files)
        if len(pending_files) < len(jpg_files):
            print(f"继续之前的处理: 已完成 {len(jpg_files) - len(pending_files)}/{len(jpg_files)}")
    else:
        pending_files = jpg_files

    if not pending_files:
        print(f"所有 {len(jpg_files)} 个文件已处理完毕！")
        logger.close()
        sys.exit(0)

    print(f"找到 {len(pending_files)} 个待处理文件（总计 {len(jpg_files)}）")
    print(f"使用 {args.workers} 个线程", end="")
    if args.qps:
        print(f"，速率限制 {args.qps} QPS")
    else:
        print()
    if args.enable_progress:
        print(f"进度文件: {progress_file}")
    print("=" * 60)

    # 全局统计锁
    stats_lock = threading.Lock()
    stats = {"total": len(jpg_files), "success": 0, "failed": 0}
    results_list = []

    # 速率限制（如果启用）
    rate_limiter = RateLimiter(qps=args.qps) if args.qps else None
    time_start_total = datetime.now()

    # 并行处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for image_path in pending_files:
            data_id = os.path.splitext(os.path.basename(image_path))[0]
            future = executor.submit(
                process_single_image,
                image_path,
                data_id,
                text_prompt,
                stats,
                stats_lock,
                rate_limiter,
            )
            futures[future] = (data_id, image_path)

        # 收集结果
        for future in as_completed(futures):
            try:
                result = future.result()
                results_list.append(result)
                data_id, image_path = futures[future]
                if args.enable_progress:
                    progress_tracker.mark_done(image_path)
            except Exception as e:
                data_id, image_path = futures[future]
                print(f"[{data_id}] 任务异常: {e}")
                with stats_lock:
                    stats["failed"] += 1

    time_end_total = datetime.now()
    total_elapsed = (time_end_total - time_start_total).total_seconds()

    # 计算耗时统计
    elapsed_times = [r["elapsed_seconds"] for r in results_list if r["error"] is None]

    # 输出统计
    print("=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"总任务数: {stats['total']}")
    print(f"本次处理: {len(pending_files)}")
    print(f"处理成功: {stats['success']} 个")
    print(f"处理失败: {stats['failed']} 个")
    if stats['total'] > 0:
        print(f"总成功率: {stats['success'] / stats['total'] * 100:.1f}%")
    print(f"本次耗时: {total_elapsed:.2f} 秒")
    if total_elapsed > 0:
        print(f"吞吐量: {len(pending_files) / total_elapsed:.2f} 张/秒")

    if elapsed_times:
        avg_seconds = sum(elapsed_times) / len(elapsed_times)
        min_seconds = min(elapsed_times)
        max_seconds = max(elapsed_times)
        print("-" * 60)
        print("耗时分析")
        print("-" * 60)
        print(f"平均耗时 (单个): {avg_seconds:.2f} 秒")
        print(f"最短耗时 (单个): {min_seconds:.2f} 秒")
        print(f"最长耗时 (单个): {max_seconds:.2f} 秒")

    print("=" * 60)

    # 处理结果：提取 JSON 并扁平化
    processed_results = []
    for item in results_list:
        if item["error"] is not None:
            continue

        data_id = item["data_id"]
        image_path = item["image_path"]
        result_field = item.get("result")

        if result_field:
            processed_item = {
                "data_id": data_id,
                "image_path": image_path,
                "result": item.get("result"),
                "text": item.get("text"),
                "mask": item.get("mask"),
                "line": item.get("line"),
            }
            processed_results.append(processed_item)

    # 保存处理后的结果
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {args.result_json}")
    print(f"成功提取: {len(processed_results)} 条结果")

    # 关闭日志
    logger.close()
