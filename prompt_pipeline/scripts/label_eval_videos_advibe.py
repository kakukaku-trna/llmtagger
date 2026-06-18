#!/usr/bin/env python3
"""使用 Advibe BaseChat 后台任务批量标注 eval_videos 视频。

当前实际限制：
1. 当前安装的 advibe-sdk 版本是 0.0.7，其中 VideoInput 只接受 http/https URL。
   文档里提到的“本地视频上传”能力，在这个版本的 SDK 里并没有实际封装出来，
   所以不能直接把本地 mp4 路径提交给模型。
2. 因此脚本需要外部可访问的视频 URL，可以通过以下三种方式提供：
   - 批量目录模式：传入 --video-root 和 --video-url-prefix
   - 单条直链模式：传入 --video-url，走 BaseChat
   - 单条 Adviz 模式：传入 --adviz-url，走 AdvizChat
3. /share-global/huajiang.sun/eval_videos 这份目录可能仍在同步中。如果共享目录还没
   完整同步，或者扫描很慢，优先使用 /data-algorithm/huajiang.sun/eval_videos 作为输入。

所以这个脚本按“两阶段、可断点续跑”来写：
- 第一阶段：提交后台任务
- 第二阶段：调用 GetTask 轮询结果，并把解析后的 JSON 展平到 CSV

Kimi 使用建议：
- 单条或批量 HTTP(S) 视频 URL：优先走 BaseChat
- Adviz URL：仍然走 AdvizChat，这条链路和 BaseChat 不是同一个服务接口

当前默认配置意图：
- user_name: huajiang.sun
- model_name: Kimi
- prompt_file: prompts/eval_benchmark/v9_qwen35_full_20260616.md
- video_root: /share-global/huajiang.sun/eval_videos

示例：
  export ADVIBE_USER_NAME="huajiang.sun"
  export ADVIBE_API_KEY="..."
  python3 scripts/label_eval_videos_advibe.py     --video-root /share-global/huajiang.sun/eval_videos     --prompt-file prompts/eval_benchmark/v9_qwen35_full_20260616.md     --video-url-prefix https://your-host/share-global/huajiang.sun/eval_videos/     --model-name Kimi

  python3 scripts/label_eval_videos_advibe.py     --video-url https://www.w3school.com.cn/i/movie.mp4     --clip-id smoke_movie     --prompt-file prompts/eval_benchmark/v9_qwen35_full_20260616.md     --model-name Kimi
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from advibe_sdk import AdvizChat, BaseChat, GetTask, VideoInput

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ROOT = Path('/share-global/huajiang.sun/eval_videos')
DEFAULT_PROMPT_FILE = ROOT / 'prompts/eval_benchmark/v9_qwen35_full_20260616.md'
DEFAULT_OUTPUT_DIR = ROOT / 'prompts/eval_benchmark'
DEFAULT_QUERY = '请严格按照系统提示词完成标注，并且只输出一个 JSON 对象，不要输出解释、markdown 或额外文本。'
PREFERRED_LABEL_ORDER = [
    '天气', '时段', '公交车道', '双向单车道', '前方左转待转区',
    '逆光', '炫光', '强反射', '光影', '道路积雪',
]
FINAL_STATUSES = {'success', 'failed'}


@dataclass
class VideoRecord:
    clip_id: str
    video_path: str
    video_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-label eval videos with Advibe background tasks.')
    parser.add_argument('--video-root', type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument('--prompt-file', type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--video-url-prefix', type=str, default=None,
                        help='Externally reachable HTTP(S) prefix that mirrors --video-root.')
    parser.add_argument('--video-url', type=str, default=None,
                        help='Single direct HTTP(S) video URL test mode. Uses BaseChat instead of scanning --video-root.')
    parser.add_argument('--adviz-url', type=str, default=None,
                        help='Single Adviz URL test mode. This uses AdvizChat and is separate from BaseChat video URL mode.')
    parser.add_argument('--layout-path', type=str, default=None,
                        help='Layout path required by AdvizChat. If omitted in adviz mode, the script falls back to layoutId from the URL for testing.')
    parser.add_argument('--clip-id', type=str, default=None,
                        help='Optional clip id / extra_info override for single URL / single Adviz test mode.')
    parser.add_argument('--model-name', type=str, default='Kimi')
    parser.add_argument('--user-name', type=str, default=os.getenv('ADVIBE_USER_NAME', ''))
    parser.add_argument('--api-key', type=str, default=os.getenv('ADVIBE_API_KEY', ''))
    parser.add_argument('--query', type=str, default=DEFAULT_QUERY)
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--submit-workers', type=int, default=4)
    parser.add_argument('--poll-interval', type=float, default=5.0)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--submit-only', action='store_true')
    parser.add_argument('--poll-only', action='store_true')
    parser.add_argument('--enable-thinking', action='store_true')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    return parser.parse_args()


def sanitize_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_')


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def read_jsonl_map(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return data
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get(key_field)
            if key:
                data[key] = obj
    return data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def infer_layout_from_adviz_url(adviz_url: str) -> str | None:
    parsed = urlparse(adviz_url)
    query = parse_qs(parsed.query)
    values = query.get('layoutId') or query.get('layout') or []
    return values[0] if values else None


def build_single_video_record(args: argparse.Namespace) -> VideoRecord:
    path_name = urlparse(args.video_url).path.rsplit('/', 1)[-1]
    clip_id = args.clip_id or sanitize_name(path_name) or f'video_{int(time.time())}'
    return VideoRecord(clip_id=clip_id, video_path='', video_url=args.video_url)


def submit_adviz_one(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    layout = args.layout_path or infer_layout_from_adviz_url(args.adviz_url)
    if not layout:
        raise ValueError('adviz mode requires --layout-path, or the URL must contain layoutId/layout.')
    clip_id = args.clip_id or infer_layout_from_adviz_url(args.adviz_url) or f'adviz_{int(time.time())}'
    client = AdvizChat(
        layout_path=layout,
        user_name=args.user_name,
        api_key=args.api_key,
        prompt=prompt,
        model_name=args.model_name,
        fps=args.fps,
        model_extra_params={'enable_thinking': True} if args.enable_thinking else None,
    )
    task = client.chat_by_url_background(
        query=args.query,
        adviz_url=args.adviz_url,
        extra_info=clip_id,
        prompt=prompt,
    )
    return {
        'clip_id': clip_id,
        'video_path': '',
        'video_url': args.adviz_url,
        'task_id': task.task_id,
        'submit_success': task.success,
        'submit_status': task.status,
        'submit_error': task.error_message,
        'submitted_at': int(time.time()),
        'source_type': 'adviz_url',
        'layout_path': layout,
    }


def build_video_url(video_root: Path, video_path: Path, prefix: str) -> str:
    rel = video_path.relative_to(video_root).as_posix().split('/')
    quoted = '/'.join(quote(part) for part in rel)
    return prefix.rstrip('/') + '/' + quoted


def discover_videos(video_root: Path, prefix: str) -> list[VideoRecord]:
    if not video_root.is_dir():
        raise FileNotFoundError(f'video_root not found: {video_root}')
    records: list[VideoRecord] = []
    for clip_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        mp4s = sorted(p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() == '.mp4')
        if not mp4s:
            continue
        video_path = mp4s[0]
        records.append(
            VideoRecord(
                clip_id=clip_dir.name,
                video_path=str(video_path),
                video_url=build_video_url(video_root, video_path, prefix),
            )
        )
    return records


def submit_one(record: VideoRecord, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    chat = BaseChat(
        user_name=args.user_name,
        api_key=args.api_key,
        prompt=prompt,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        model_extra_params={'enable_thinking': True} if args.enable_thinking else None,
    )
    video = VideoInput(url=record.video_url, fps=args.fps)
    task = chat.chat_background(query=args.query, video=video, extra_info=record.clip_id)
    row = asdict(record)
    row.update({
        'task_id': task.task_id,
        'submit_success': task.success,
        'submit_status': task.status,
        'submit_error': task.error_message,
        'submitted_at': int(time.time()),
        'source_type': 'video_url',
    })
    return row


def poll_task(task_client: GetTask, task_row: dict[str, Any]) -> dict[str, Any]:
    task_id = task_row['task_id']
    result = task_client.get_task(task_id)
    parsed = extract_json_from_text(result.content or '') if result.content else None
    row = dict(task_row)
    row.update({
        'status': result.status,
        'success': result.success,
        'error_message': result.error_message,
        'content': result.content,
        'reasoning_content': result.reasoning_content,
        'usage': result.usage,
        'extra_info': result.extra_info,
        'parsed': parsed,
        'updated_at': int(time.time()),
    })
    return row


def write_predictions_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    extra_keys: set[str] = set()
    for row in results.values():
        parsed = row.get('parsed') or {}
        if isinstance(parsed, dict):
            extra_keys.update(parsed.keys())
    ordered_labels = [k for k in PREFERRED_LABEL_ORDER if k in extra_keys]
    other_keys = sorted(k for k in extra_keys if k not in ordered_labels)
    fieldnames = [
        'clip_id', 'video_path', 'video_url', 'task_id', 'status',
        *ordered_labels, *other_keys,
        'raw_content', 'reasoning_content', 'error_message', 'usage_json',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for clip_id in sorted(results):
            row = results[clip_id]
            parsed = row.get('parsed') or {}
            out = {
                'clip_id': clip_id,
                'video_path': row.get('video_path', ''),
                'video_url': row.get('video_url', ''),
                'task_id': row.get('task_id', ''),
                'status': row.get('status', ''),
                'raw_content': row.get('content', ''),
                'reasoning_content': row.get('reasoning_content', ''),
                'error_message': row.get('error_message', ''),
                'usage_json': json.dumps(row.get('usage', {}), ensure_ascii=False),
            }
            if isinstance(parsed, dict):
                for key in ordered_labels + other_keys:
                    out[key] = parsed.get(key, '')
            writer.writerow(out)


def write_summary(path: Path, tasks: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]) -> None:
    task_submit_counter = Counter((row.get('submit_status') or 'unknown') for row in tasks.values())
    result_status_counter = Counter((row.get('status') or 'unknown') for row in results.values())
    summary = {
        'task_count': len(tasks),
        'result_count': len(results),
        'submit_status': dict(task_submit_counter),
        'result_status': dict(result_status_counter),
        'updated_at': int(time.time()),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def resolve_run_source_name(args: argparse.Namespace) -> str:
    if args.video_url:
        return build_single_video_record(args).clip_id
    if args.adviz_url:
        return args.clip_id or infer_layout_from_adviz_url(args.adviz_url) or 'single_adviz_url'
    return args.video_root.name


def main() -> int:
    args = parse_args()
    if not args.user_name:
        raise SystemExit('Missing user name. Pass --user-name or set ADVIBE_USER_NAME.')
    if not args.api_key:
        raise SystemExit('Missing api key. Pass --api-key or set ADVIBE_API_KEY.')
    if args.video_url and args.adviz_url:
        raise SystemExit('--video-url and --adviz-url cannot be used together.')
    if args.video_url and args.video_url_prefix:
        raise SystemExit('--video-url and --video-url-prefix cannot be used together.')
    if not args.video_url and not args.adviz_url and not args.video_url_prefix:
        raise SystemExit(
            'BaseChat video mode requires --video-url-prefix that maps the local video root to an externally reachable URL. '
            'If you want to test a single direct video link, pass --video-url. '
            'If you want to test a single Adviz link, pass --adviz-url.'
        )
    if args.submit_only and args.poll_only:
        raise SystemExit('--submit-only and --poll-only cannot be used together.')

    prompt = args.prompt_file.read_text(encoding='utf-8')
    run_name = sanitize_name(f"{args.model_name}_{args.prompt_file.stem}_{resolve_run_source_name(args)}")
    tasks_path = args.output_dir / f'tasks_{run_name}.jsonl'
    results_path = args.output_dir / f'results_{run_name}.jsonl'
    predictions_path = args.output_dir / f'predictions_{run_name}.csv'
    summary_path = args.output_dir / f'summary_{run_name}.json'

    tasks = {} if args.overwrite else read_jsonl_map(tasks_path, 'clip_id')
    results = {} if args.overwrite else read_jsonl_map(results_path, 'clip_id')

    if args.video_url:
        videos = [build_single_video_record(args)]
        print('single_mode=video_url')
        print(f'video_url={args.video_url}')
    elif args.adviz_url:
        videos: list[VideoRecord] = []
        print('single_mode=adviz_url')
        print(f'adviz_url={args.adviz_url}')
    else:
        videos = discover_videos(args.video_root, args.video_url_prefix)
        if args.sample > 0:
            videos = videos[:args.sample]
        print(f'discovered_videos={len(videos)} from {args.video_root}')

    if not args.poll_only:
        if args.adviz_url:
            adviz_clip_id = args.clip_id or infer_layout_from_adviz_url(args.adviz_url) or 'adviz_single'
            need_submit = args.overwrite or adviz_clip_id not in tasks or not tasks[adviz_clip_id].get('task_id')
            print(f'to_submit={1 if need_submit else 0}')
            if need_submit:
                try:
                    task_row = submit_adviz_one(prompt, args)
                except Exception as exc:
                    task_row = {
                        'clip_id': adviz_clip_id,
                        'video_path': '',
                        'video_url': args.adviz_url,
                        'task_id': None,
                        'submit_success': False,
                        'submit_status': 'failed',
                        'submit_error': str(exc),
                        'submitted_at': int(time.time()),
                        'source_type': 'adviz_url',
                        'layout_path': args.layout_path or infer_layout_from_adviz_url(args.adviz_url),
                    }
                tasks[task_row['clip_id']] = task_row
                print(f"submit {task_row['clip_id']}: task_id={task_row.get('task_id')} status={task_row.get('submit_status')} success={task_row.get('submit_success')}")
                write_jsonl(tasks_path, [tasks[k] for k in sorted(tasks)])
                write_summary(summary_path, tasks, results)
        else:
            to_submit = [
                record for record in videos
                if args.overwrite or record.clip_id not in tasks or not tasks[record.clip_id].get('task_id')
            ]
            print(f'to_submit={len(to_submit)}')
            if to_submit:
                with ThreadPoolExecutor(max_workers=max(1, args.submit_workers)) as pool:
                    futures = {pool.submit(submit_one, record, prompt, args): record for record in to_submit}
                    for future in as_completed(futures):
                        record = futures[future]
                        try:
                            task_row = future.result()
                        except Exception as exc:
                            task_row = asdict(record)
                            task_row.update({
                                'task_id': None,
                                'submit_success': False,
                                'submit_status': 'failed',
                                'submit_error': str(exc),
                                'submitted_at': int(time.time()),
                                'source_type': 'video_url',
                            })
                        tasks[record.clip_id] = task_row
                        print(f"submit {record.clip_id}: task_id={task_row.get('task_id')} status={task_row.get('submit_status')} success={task_row.get('submit_success')}")
                        write_jsonl(tasks_path, [tasks[k] for k in sorted(tasks)])
                        write_summary(summary_path, tasks, results)

    if args.submit_only:
        print(f'submit phase done: {tasks_path}')
        return 0

    task_client = GetTask(user_name=args.user_name, api_key=args.api_key)
    pending = {
        clip_id: row for clip_id, row in tasks.items()
        if row.get('task_id') and (clip_id not in results or results[clip_id].get('status') not in FINAL_STATUSES)
    }
    print(f'pending_tasks={len(pending)}')

    while pending:
        progressed = 0
        for clip_id in sorted(list(pending)):
            row = pending[clip_id]
            polled = poll_task(task_client, row)
            results[clip_id] = polled
            status = polled.get('status') or 'unknown'
            print(f'poll {clip_id}: status={status} success={polled.get("success")}')
            if status in FINAL_STATUSES:
                pending.pop(clip_id, None)
                progressed += 1
            write_jsonl(results_path, [results[k] for k in sorted(results)])
            write_predictions_csv(predictions_path, results)
            write_summary(summary_path, tasks, results)
        if pending:
            print(f'pending_remaining={len(pending)} sleeping={args.poll_interval}s progressed={progressed}')
            time.sleep(args.poll_interval)

    print(f'done results={results_path}')
    print(f'done predictions={predictions_path}')
    print(f'done summary={summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
