#!/usr/bin/env python3
"""
自动迭代评测脚本
基于评测结果自动迭代改进prompt，直到Precision和Recall都达到80%以上
最多迭代5轮
"""

import os
import sys
import re
import json
import glob
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

# 配置
MAX_ITERATIONS = 5
TARGET_PRECISION = 80.0
TARGET_RECALL = 80.0
WORKERS = 5
BASE_DIR = "/home/nio/EOL_cpp/data_label_prompt"
MODEL_SCRIPT = f"{BASE_DIR}/model_muse/qwen_api_call_video.py"
TRUE_DIR = f"{BASE_DIR}/left_video_true"
FALSE_DIR = f"{BASE_DIR}/left_video_false"
PROMPT_BASE = f"{BASE_DIR}/waitzone_left"
EVAL_LOG_DIR = f"{BASE_DIR}/video_label_log"

def run_evaluation(prompt_file: str, input_dir: str, ground_truth: str, log_file: str) -> Tuple[bool, Dict]:
    """
    运行单次评测
    
    Returns:
        (success, eval_stats)
    """
    cmd = [
        "python3", MODEL_SCRIPT,
        "--input_dir", input_dir,
        "--prompt", prompt_file,
        "--log", log_file,
        "--workers", str(WORKERS),
        "--evaluate",
        "--ground_truth", ground_truth,
    ]
    
    print(f"  运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1小时超时
        )
        
        # 解析输出中的评测指标
        output = result.stdout + result.stderr
        
        # 提取 Precision 和 Recall
        precision_match = re.search(r'Precision:\s*([\d.]+)%', output)
        recall_match = re.search(r'Recall:\s*([\d.]+)%', output)
        
        # 提取 TP, FP, TN, FN
        tp_match = re.search(r'TP \(真正例\):\s*(\d+)', output)
        fp_match = re.search(r'FP \(假正例\):\s*(\d+)', output)
        tn_match = re.search(r'TN \(真负例\):\s*(\d+)', output)
        fn_match = re.search(r'FN \(假负例\):\s*(\d+)', output)
        
        eval_stats = {
            "precision": float(precision_match.group(1)) if precision_match else 0.0,
            "recall": float(recall_match.group(1)) if recall_match else 0.0,
            "TP": int(tp_match.group(1)) if tp_match else 0,
            "FP": int(fp_match.group(1)) if fp_match else 0,
            "TN": int(tn_match.group(1)) if tn_match else 0,
            "FN": int(fn_match.group(1)) if fn_match else 0,
            "raw_output": output,
        }
        
        return True, eval_stats
        
    except subprocess.TimeoutExpired:
        print(f"  错误: 评测超时")
        return False, {}
    except Exception as e:
        print(f"  错误: 运行评测失败: {e}")
        return False, {}


def analyze_failures(eval_stats: Dict, ground_truth: str) -> List[str]:
    """
    分析失败模式，返回改进建议
    
    Args:
        eval_stats: 评测统计
        ground_truth: "是" 或 "否"
    
    Returns:
        改进建议列表
    """
    suggestions = []
    
    if ground_truth == "是":
        # 正样本评测
        # FN: 真值=是, 预测=否 (漏检)
        fn = eval_stats.get("FN", 0)
        if fn > 0:
            suggestions.append(f"正样本中有 {fn} 个漏检（FN），需要增强对待转区特征的描述，特别是夜间/模糊场景")
        
        # FP: 真值=否, 预测=是 (误检)
        # 在正样本评测中，FP代表其他情况
    else:
        # 负样本评测
        # FP: 真值=否, 预测=是 (误检)
        fp = eval_stats.get("FP", 0)
        if fp > 0:
            suggestions.append(f"负样本中有 {fp} 个误检（FP），需要加强对导流线和待转区区分度的描述")
        
        # FN: 真值=是, 预测=否 (漏检)
        # 在负样本评测中，FN代表其他情况
    
    return suggestions


def generate_improved_prompt(current_prompt_file: str, iteration: int, 
                            true_eval: Dict, false_eval: Dict) -> str:
    """
    基于评测结果生成改进的prompt
    
    Args:
        current_prompt_file: 当前prompt文件路径
        iteration: 当前迭代次数
        true_eval: 正样本评测结果
        false_eval: 负样本评测结果
    
    Returns:
        新prompt文件路径
    """
    # 读取当前prompt
    with open(current_prompt_file, 'r', encoding='utf-8') as f:
        current_prompt = f.read()
    
    # 分析失败模式
    true_suggestions = analyze_failures(true_eval, "是")
    false_suggestions = analyze_failures(false_eval, "否")
    
    # 生成改进提示
    improvements = []
    
    # 基于Precision和Recall分析
    true_precision = true_eval.get("precision", 0)
    true_recall = true_eval.get("recall", 0)
    false_precision = false_eval.get("precision", 0)
    false_recall = false_eval.get("recall", 0)
    
    if true_recall < TARGET_RECALL:
        improvements.append(
            f"正样本Recall仅{true_recall:.1f}%，需要增强对待转区关键特征的识别能力，"
            f"特别是在复杂场景（夜间、雨天、视角不佳）下的识别"
        )
    
    if false_precision < TARGET_PRECISION:
        improvements.append(
            f"负样本Precision仅{false_precision:.1f}%，需要加强对导流线和待转区的区分能力，"
            f"明确导流线贯穿路口、无停止线截断的特征"
        )
    
    # 合并所有建议
    all_suggestions = true_suggestions + false_suggestions + improvements
    
    # 生成新prompt（在当前prompt基础上增加改进说明）
    new_prompt = current_prompt + "\n\n"
    new_prompt += f"# 第{iteration}轮迭代改进（基于评测结果）\n\n"
    new_prompt += "基于上一轮评测结果，以下场景需要特别注意：\n\n"
    
    for i, suggestion in enumerate(all_suggestions, 1):
        new_prompt += f"{i}. {suggestion}\n"
    
    new_prompt += "\n"
    new_prompt += "重要提醒：\n"
    new_prompt += "- 在不确定的情况下，优先考虑停止线截断特征作为判断依据\n"
    new_prompt += "- 夜间场景需要结合多帧信息和车辆行为辅助判断\n"
    new_prompt += "- 导流线一定不会有停止线截断，这是与待转区的核心区别\n"
    
    # 保存新prompt
    new_prompt_file = f"{PROMPT_BASE}_v{6+iteration}.md"
    with open(new_prompt_file, 'w', encoding='utf-8') as f:
        f.write(new_prompt)
    
    return new_prompt_file


def main():
    """主流程"""
    print("=" * 80)
    print("自动迭代评测系统")
    print("=" * 80)
    print(f"目标: Precision >= {TARGET_PRECISION}%, Recall >= {TARGET_RECALL}%")
    print(f"最大迭代次数: {MAX_ITERATIONS}")
    print()
    
    current_prompt_file = f"{PROMPT_BASE}_v6.md"
    iteration = 0
    
    # 记录所有迭代结果
    all_results = []
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*80}")
        print(f"第 {iteration}/{MAX_ITERATIONS} 轮迭代")
        print(f"使用 prompt: {os.path.basename(current_prompt_file)}")
        print(f"{'='*80}")
        
        # 1. 评测正样本 (left_video_true, 真值=是)
        print("\n【1/2】评测正样本 (left_video_true, 真值=是)...")
        true_log = f"{EVAL_LOG_DIR}/eval_v{6+iteration-1}_true_{timestamp}.log"
        success, true_eval = run_evaluation(
            current_prompt_file, TRUE_DIR, "是", true_log
        )
        
        if not success:
            print("正样本评测失败，跳过本轮")
            continue
        
        print(f"  Precision: {true_eval['precision']:.1f}%")
        print(f"  Recall: {true_eval['recall']:.1f}%")
        
        # 2. 评测负样本 (left_video_false, 真值=否)
        print("\n【2/2】评测负样本 (left_video_false, 真值=否)...")
        false_log = f"{EVAL_LOG_DIR}/eval_v{6+iteration-1}_false_{timestamp}.log"
        success, false_eval = run_evaluation(
            current_prompt_file, FALSE_DIR, "否", false_log
        )
        
        if not success:
            print("负样本评测失败，跳过本轮")
            continue
        
        print(f"  Precision: {false_eval['precision']:.1f}%")
        print(f"  Recall: {false_eval['recall']:.1f}%")
        
        # 记录结果
        result_record = {
            "iteration": iteration,
            "prompt_version": f"v{6+iteration-1}",
            "prompt_file": current_prompt_file,
            "true_eval": true_eval,
            "false_eval": false_eval,
            "timestamp": timestamp,
        }
        all_results.append(result_record)
        
        # 检查是否达到目标
        true_precision = true_eval["precision"]
        true_recall = true_eval["recall"]
        false_precision = false_eval["precision"]
        false_recall = false_eval["recall"]
        
        # 综合Precision和Recall（取平均值）
        avg_precision = (true_precision + false_precision) / 2
        avg_recall = (true_recall + false_recall) / 2
        
        print(f"\n{'='*80}")
        print(f"第 {iteration} 轮结果汇总")
        print(f"{'='*80}")
        print(f"正样本 - Precision: {true_precision:.1f}%, Recall: {true_recall:.1f}%")
        print(f"负样本 - Precision: {false_precision:.1f}%, Recall: {false_recall:.1f}%")
        print(f"综合 - Precision: {avg_precision:.1f}%, Recall: {avg_recall:.1f}%")
        
        if avg_precision >= TARGET_PRECISION and avg_recall >= TARGET_RECALL:
            print(f"\n🎉 达到目标！Precision ({avg_precision:.1f}%) >= {TARGET_PRECISION}%")
            print(f"🎉 Recall ({avg_recall:.1f}%) >= {TARGET_RECALL}%")
            print(f"🎉 总共迭代 {iteration} 轮")
            break
        
        if iteration >= MAX_ITERATIONS:
            print(f"\n⚠️  达到最大迭代次数 {MAX_ITERATIONS}，停止迭代")
            break
        
        # 生成改进的prompt
        print(f"\n生成第 {iteration+1} 轮改进prompt...")
        new_prompt_file = generate_improved_prompt(
            current_prompt_file, iteration, true_eval, false_eval
        )
        print(f"新prompt已保存: {new_prompt_file}")
        
        current_prompt_file = new_prompt_file
    
    # 保存所有迭代结果
    results_file = f"{BASE_DIR}/iteration_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("迭代完成")
    print(f"{'='*80}")
    print(f"结果已保存: {results_file}")
    print("\n所有迭代记录:")
    for result in all_results:
        print(f"  轮次 {result['iteration']}: {result['prompt_version']} - "
              f"Precision: {(result['true_eval']['precision'] + result['false_eval']['precision'])/2:.1f}%, "
              f"Recall: {(result['true_eval']['recall'] + result['false_eval']['recall'])/2:.1f}%")


if __name__ == "__main__":
    main()
