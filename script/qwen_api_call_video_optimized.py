"""
优化的Qwen视频分析脚本 - 使用进程池实现真正的并行处理
"""
import os
import sys
import base64
import glob
import argparse
import json
import time
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Tuple
from multiprocessing import Manager

# 设置环境变量以避免多进程警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def encode_video(video_path: str) -> str:
    """将视频文件转为 base64 字符串"""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def parse_qwen_response(response) -> Tuple[str, str]:
    """
    从 Qwen API 的各种返回格式中提取 JSON 并解析
    """
    if isinstance(response, list):
        if len(response) > 0 and isinstance(response[0], dict) and 'text' in response[0]:
            text = response[0]['text']
        else:
            text = str(response[0]) if len(response) > 0 else str(response)
    elif isinstance(response, dict):
        text = response.get('text', str(response))
    else:
        text = str(response)

    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("未在响应中找到 JSON 对象")

    json_str = match.group(0)
    data = json.loads(json_str)

    result = data.get("result", "")
    result_mapping = {
        "yes": "是",
        "Yes": "是",
        "YES": "是",
        "no": "否",
        "No": "否",
        "NO": "否",
    }
    if result in result_mapping:
        result = result_mapping[result]

    return result, data.get("reason", "")


def process_single_video(
    video_file: str,
    text_prompt: str,
    api_key: str,
    base_url: str,
    max_retries: int = 3,
) -> Dict:
    """
    处理单个视频（独立进程中使用）
    """
    from openai import OpenAI
    
    time1 = datetime.now()
    result_data = {
        "video_file": video_file,
        "status": "failed",
        "result": "",
        "reason": "",
        "elapsed_seconds": 0.0,
        "error": "",
    }
    
    try:
        # 编码视频
        encode_start = time.time()
        base64_file = encode_video(video_file)
        encode_time = time.time() - encode_start
        
        # 创建独立客户端（每个进程一个）
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        # API 调用，带重试
        api_start = time.time()
        response_content = None
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model="qwen-latest-series-invite-beta-v37",
                    temperature=0.6,
                    top_p=0.95,
                    extra_body={"enable_thinking": True,
                                "thinking_budget": 4096,
                                 "top_k": 20,
                                "repetition_penalty": 1.0,
                                "presence_penalty": 0.0},
                    # extra_body={"enable_thinking": False},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {"url": f"data:video/mp4;base64,{base64_file}", "fps": 4},
                                },
                                {"type": "text", "text": text_prompt},
                            ],
                        }
                    ],
                )
                response_content = completion.choices[0].message.content
                break
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    raise
        
        api_time = time.time() - api_start
        
        if response_content is None:
            raise Exception(f"API 调用失败，已重试 {max_retries} 次")
        
        # 解析结果
        try:
            result, reason = parse_qwen_response(response_content)
            result_data["result"] = result
            result_data["reason"] = reason
            result_data["status"] = "success"
            result_data["encode_time"] = encode_time
            result_data["api_time"] = api_time
        except (json.JSONDecodeError, ValueError) as e:
            result_data["status"] = "parse_error"
            result_data["error"] = f"无法解析 API 返回的 JSON: {e}"
            result_data["raw_response"] = response_content
            
    except Exception as e:
        result_data["error"] = f"{last_error if 'last_error' in dir() else str(e)}"
    
    time2 = datetime.now()
    elapsed = (time2 - time1).total_seconds()
    result_data["elapsed_seconds"] = elapsed
    
    return result_data


def read_prompt_from_file(file_path: str) -> str:
    """读取 prompt 文件内容"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"prompt 文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="使用 Qwen API 并发分析视频（进程池优化版）")
    parser.add_argument("--prompt", type=str, required=True, help="prompt 文件路径（md 文件）")
    parser.add_argument("--log", type=str, default="result.log", help="日志文件路径")
    parser.add_argument("--input_dir", type=str, required=True, help="输入视频文件夹路径")
    parser.add_argument("--workers", type=int, default=5, help="并发进程数（建议 3-8）")
    parser.add_argument("--max_retries", type=int, default=3, help="API 调用失败重试次数")
    parser.add_argument("--evaluate", action="store_true", help="启用评测模式")
    parser.add_argument("--ground_truth", type=str, choices=["是", "否"], help="真值标签")
    args = parser.parse_args()
    
    # 读取 prompt
    text_prompt = read_prompt_from_file(args.prompt)
    
    # API 配置
    api_key = "sk-4cc9e93494ca4b8e88f2f7b071e57db3"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 获取所有视频文件
    video_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    
    if not video_files:
        print(f"警告: 在 {args.input_dir} 目录下未找到 .mp4 文件")
        return
    
    print(f"共找到 {len(video_files)} 个视频文件")
    print(f"并发进程数: {args.workers}")
    if args.evaluate and args.ground_truth:
        print(f"评测模式: 真值={args.ground_truth}")
    print(f"开始处理...")
    print("=" * 60)
    
    # 统计
    stats = {
        "total": len(video_files),
        "success": 0,
        "yes": 0,
        "no": 0,
        "parse_error": 0,
        "failed": 0,
    }
    
    eval_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    all_results = []
    total_encode_time = 0.0
    total_api_time = 0.0
    
    start_time = time.time()
    
    # 使用进程池实现真正的并行
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        future_to_video = {
            executor.submit(
                process_single_video, 
                vf, 
                text_prompt, 
                api_key, 
                base_url, 
                args.max_retries
            ): vf 
            for vf in video_files
        }
        
        # 收集结果
        for future in as_completed(future_to_video):
            result_data = future.result()
            all_results.append(result_data)
            
            video_file = result_data["video_file"]
            status = result_data["status"]
            result = result_data["result"]
            elapsed = result_data["elapsed_seconds"]
            
            print(f"{'*' * 30} {os.path.basename(video_file)}")
            print(f"总耗时: {elapsed:.2f} 秒")
            
            if "encode_time" in result_data:
                print(f"编码耗时: {result_data['encode_time']:.2f} 秒")
                total_encode_time += result_data['encode_time']
            if "api_time" in result_data:
                print(f"API耗时: {result_data['api_time']:.2f} 秒")
                total_api_time += result_data['api_time']
            
            if status == "success":
                print(f"result: {result}")
                reason = result_data["reason"]
                print(f"reason: {reason[:200]}..." if len(reason) > 200 else f"reason: {reason}")
                stats["success"] += 1
                if result == "是":
                    stats["yes"] += 1
                elif result == "否":
                    stats["no"] += 1
                
                if args.evaluate and args.ground_truth:
                    gt = args.ground_truth
                    pred = result
                    if gt == "是" and pred == "是":
                        eval_stats["TP"] += 1
                    elif gt == "否" and pred == "是":
                        eval_stats["FP"] += 1
                    elif gt == "否" and pred == "否":
                        eval_stats["TN"] += 1
                    elif gt == "是" and pred == "否":
                        eval_stats["FN"] += 1
            elif status == "parse_error":
                print(f"解析错误: {result_data['error']}")
                stats["parse_error"] += 1
            else:
                print(f"处理失败: {result_data['error']}")
                stats["failed"] += 1
            
            print()
    
    total_wall_time = time.time() - start_time
    
    # 输出统计
    print("=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总视频数: {stats['total']}")
    print(f"处理成功: {stats['success']} 个")
    print(f'  结果 为 "是": {stats["yes"]} 个')
    print(f'  结果 为 "否": {stats["no"]} 个')
    print(f"解析失败: {stats['parse_error']} 个")
    print(f"处理失败: {stats['failed']} 个")
    
    if args.evaluate and args.ground_truth:
        print()
        print("=" * 60)
        print("评测指标")
        print("=" * 60)
        print(f"TP: {eval_stats['TP']}, FP: {eval_stats['FP']}")
        print(f"TN: {eval_stats['TN']}, FN: {eval_stats['FN']}")
        
        if args.ground_truth == "是":
            precision_denom = eval_stats["TP"] + eval_stats["FP"]
            recall_denom = eval_stats["TP"] + eval_stats["FN"]
            precision = eval_stats["TP"] / precision_denom * 100 if precision_denom > 0 else 0
            recall = eval_stats["TP"] / recall_denom * 100 if recall_denom > 0 else 0
        else:
            precision_denom = eval_stats["TN"] + eval_stats["FN"]
            recall_denom = eval_stats["TN"] + eval_stats["FP"]
            precision = eval_stats["TN"] / precision_denom * 100 if precision_denom > 0 else 0
            recall = eval_stats["TN"] / recall_denom * 100 if recall_denom > 0 else 0
        
        print(f"\nPrecision: {precision:.1f}%")
        print(f"Recall: {recall:.1f}%")
    
    # 耗时分析
    print()
    print("=" * 60)
    print("耗时分析")
    print("=" * 60)
    print(f"总 wall-clock 时间: {total_wall_time:.2f} 秒")
    print(f"总编码时间: {total_encode_time:.2f} 秒")
    print(f"总 API 时间: {total_api_time:.2f} 秒")
    print(f"理论串行时间: {total_encode_time + total_api_time:.2f} 秒")
    print(f"实际并发加速比: {(total_encode_time + total_api_time) / total_wall_time:.2f}x")
    print("=" * 60)
    
    # 保存结果到日志
    if args.log:
        with open(args.log, 'w', encoding='utf-8') as f:
            f.write(f"处理时间: {datetime.now()}\n")
            f.write(f"总视频数: {stats['total']}\n")
            f.write(f"并发进程数: {args.workers}\n")
            f.write(f"总 wall-clock 时间: {total_wall_time:.2f} 秒\n")
            f.write(f"实际并发加速比: {(total_encode_time + total_api_time) / total_wall_time:.2f}x\n")
            f.write("\n")
            for r in all_results:
                f.write(f"{'*' * 30} {r['video_file']}\n")
                f.write(f"耗时: {r.get('elapsed_seconds', 0):.2f} 秒\n")
                if 'encode_time' in r:
                    f.write(f"编码耗时: {r['encode_time']:.2f} 秒\n")
                if 'api_time' in r:
                    f.write(f"API耗时: {r['api_time']:.2f} 秒\n")
                f.write(f"result: {r.get('result', 'N/A')}\n")
                if 'reason' in r:
                    f.write(f"reason: {r['reason']}\n")
                if 'error' in r and r['error']:
                    f.write(f"error: {r['error']}\n")
                if 'raw_response' in r:
                    f.write(f"raw_response: {r['raw_response'][:500]}...\n")
                f.write("\n")


if __name__ == "__main__":
    main()
