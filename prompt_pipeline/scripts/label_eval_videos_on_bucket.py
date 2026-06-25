#!/usr/bin/env python3
"""使用 BaseChat 后台任务批量标注 share-global 上的 eval_videos 视频。

通过 NioFS 内网预签名 URL（internet=False）访问 share-global COS 桶，
绕过公有云 OSS 桶的 VPC 外部访问限制。advibe 后台服务在内网，
可直接访问 cos.huailai.cos.nads3-hl.nioint.com 内网 URL。

两阶段、可断点续跑：
  第一阶段：提交后台任务（submit）
  第二阶段：轮询结果并写 CSV（poll）

默认配置：
  bucket:       share-global（内网 COS，eval_videos 文件在此）
  bucket_prefix: huajiang.sun/eval_videos/
  niofs_ak/sk:  huajiang.sun 的 AK/SK
  model_name:   Kimi
  prompt_file:  prompts/eval_benchmark/v10.md

示例：
  export ADVIBE_USER_NAME="huajiang.sun"
  export ADVIBE_API_KEY="..."

  # 全量提交 + 轮询
  python3 scripts/label_eval_videos_on_bucket.py

  # 只提交，不轮询
  python3 scripts/label_eval_videos_on_bucket.py --submit-only

  # 只轮询（已提交过的任务）
  python3 scripts/label_eval_videos_on_bucket.py --poll-only

  # 采样 10 条测试
  python3 scripts/label_eval_videos_on_bucket.py --sample 10
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
from pathlib import Path
from typing import Any

import niofs
from advibe_sdk import BaseChat, BucketVideo, GetTask, OnBucketChat, VideoInput, logger

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = ROOT / 'prompts/eval_benchmark/v10.md'
DEFAULT_OUTPUT_DIR = ROOT / 'prompts/eval_benchmark'
DEFAULT_QUERY = '请严格按照系统提示词完成标注，并且只输出一个 JSON 对象，不要输出解释、markdown 或额外文本。'

NIOFS_AK = '9632f7701fa64437'
NIOFS_SK = 'e324a4fc9b334967ab89320be53a58a4'
DEFAULT_BUCKET = 'share-global'          # 内网 COS 桶，advibe 后台可访问
DEFAULT_BUCKET_PREFIX = 'huajiang.sun/eval_videos/'

PREFERRED_LABEL_ORDER = [
    '天气', '时段', '公交车道', '非机动车道', '路面积水',
    '双向单车道', '前方左转待转区',
    '逆光', '炫光', '强反射', '光影', '道路积雪',
]
FINAL_STATUSES = {'success', 'failed'}


class IntranetOnBucketChat(OnBucketChat):
    """share-global 内网 COS 桶专用 OnBucketChat。

    原生 OnBucketChat 在 __init__ 中会拒绝 region 为 hefei/huailai 的桶
    (非公有云)。share-global 虽然 region='huailai'，但实际上 advibe 后台
    在公司内网，可访问 cos.nads3-hl.nioint.com 内网地址。

    用法：和 OnBucketChat 完全一致，只需替换类即可。
    """

    def __init__(
        self,
        bucket: str,
        niofs_ak: str,
        niofs_sk: str,
        user_name: str | None = None,
        api_key: str | None = None,
        prompt: str | None = '',
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        model_extra_params: dict | None = None,
        fps: int | None = None,
    ) -> None:
        # 跳过父类会抛异常的 region 检查，直接走底层逻辑
        from advibe_sdk.config import AdvibeConfig
        from advibe_sdk.base_chat import BaseChat
        self.config = AdvibeConfig()

        self.user_name = user_name if user_name else self.config.user_name
        if not self.user_name:
            raise ValueError('user_name not set')

        self.api_key = api_key if api_key else self.config.api_key
        if not self.api_key:
            raise ValueError('api_key not set')

        self.bucket = bucket
        self.prompt = prompt
        self.model_name = model_name if model_name else self.config.model_name
        self.fps = fps

        import niofs as _niofs
        self.niofs_client = _niofs.Client(niofs_ak, niofs_sk)
        self.base_chat = BaseChat(
            user_name=self.user_name,
            api_key=self.api_key,
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            model_extra_params=model_extra_params,
        )

    def _generate_presigned_url(self, key: str, expiration: int = None) -> str:
        """强制使用内网预签名 URL（internet=False）。"""
        if expiration is None:
            expiration = self.config.presign_url_timeout
        logger.info('Generating intranet presigned URL for key: %s', key)
        return self.niofs_client.generate_presigned_url(
            self.bucket, key, expiration=expiration, internet=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-label eval videos via OnBucketChat background tasks.')
    parser.add_argument('--bucket', type=str, default=DEFAULT_BUCKET)
    parser.add_argument('--bucket-prefix', type=str, default=DEFAULT_BUCKET_PREFIX,
                        help='Key prefix under which eval_videos clips are stored.')
    parser.add_argument('--prompt-file', type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model-name', type=str, default='Kimi')
    parser.add_argument('--user-name', type=str, default=os.getenv('ADVIBE_USER_NAME', ''))
    parser.add_argument('--api-key', type=str, default=os.getenv('ADVIBE_API_KEY', ''))
    parser.add_argument('--query', type=str, default=DEFAULT_QUERY)
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--submit-workers', type=int, default=4)
    parser.add_argument('--poll-interval', type=float, default=5.0)
    parser.add_argument('--use-on-bucket', action='store_true',
                        help='使用 IntranetOnBucketChat（SDK 封装的对象存储对话）。'
                             '默认使用兼容模式：BaseChat + generate_presigned_url(internet=False)')
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


def discover_clips(niofs_client: niofs.Client, bucket: str, prefix: str) -> list[dict[str, str]]:
    """List clips from the bucket. Each clip is the first .mp4 under a UUID subdirectory.

    Handles POSIX buckets (e.g. share-global) where the SDK adds an internal prefix
    (e.g. 'idc1/') to returned keys. The returned object_key is the logical key
    without the internal prefix, suitable for generate_presigned_url().
    """
    prefix = prefix.rstrip('/') + '/'
    resp = niofs_client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, MaxKeys=1000, Delimiter='/'
    )
    raw_dirs = [cp['Prefix'] for cp in (resp.get('CommonPrefixes') or [])]

    # Detect SDK-internal prefix (e.g. 'idc1/') prepended to returned keys
    sdk_prefix = ''
    if raw_dirs:
        sample = raw_dirs[0]
        idx = sample.find(prefix)
        if idx > 0:
            sdk_prefix = sample[:idx]

    clips = []
    for raw_dir in sorted(raw_dirs):
        logical_dir = raw_dir[len(sdk_prefix):]   # strip internal prefix
        clip_id = logical_dir.rstrip('/').rsplit('/', 1)[-1]
        r = niofs_client.list_objects_v2(
            Bucket=bucket, Prefix=logical_dir, MaxKeys=20, Delimiter=''
        )
        mp4s = sorted(
            obj['Key'][len(sdk_prefix):]           # strip internal prefix from object key
            for obj in (r.get('Contents') or [])
            if obj['Key'].lower().endswith('.mp4')
        )
        if mp4s:
            clips.append({'clip_id': clip_id, 'object_key': mp4s[0]})
    return clips


def submit_one_basechat(clip: dict[str, str], niofs_client: niofs.Client, base_chat: BaseChat,
                        query: str, fps: int, bucket: str) -> dict[str, Any]:
    """兼容模式：BaseChat + generate_presigned_url(internet=False)"""
    url = niofs_client.generate_presigned_url(
        bucket, clip['object_key'], expiration=12 * 3600, internet=False
    )
    video = VideoInput(url=url, fps=fps)
    task = base_chat.chat_background(
        query=query,
        video=video,
        extra_info=clip['clip_id'],
    )
    return {
        'clip_id': clip['clip_id'],
        'object_key': clip['object_key'],
        'task_id': task.task_id,
        'submit_success': task.success,
        'submit_status': task.status,
        'submit_error': task.error_message,
        'submitted_at': int(time.time()),
    }


def submit_one_onbucket(clip: dict[str, str], bucket_chat: IntranetOnBucketChat,
                        query: str, fps: int) -> dict[str, Any]:
    """SDK 模式：IntranetOnBucketChat + BucketVideo(key=...)"""
    video = BucketVideo(key=clip['object_key'])
    task = bucket_chat.chat_video_background(
        query=query,
        video_key=video,
        extra_info=clip['clip_id'],
    )
    return {
        'clip_id': clip['clip_id'],
        'object_key': clip['object_key'],
        'task_id': task.task_id,
        'submit_success': task.success,
        'submit_status': task.status,
        'submit_error': task.error_message,
        'submitted_at': int(time.time()),
    }


def poll_task(task_client: GetTask, task_row: dict[str, Any]) -> dict[str, Any]:
    result = task_client.get_task(task_row['task_id'])
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
        'clip_id', 'object_key', 'task_id', 'status',
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
                'object_key': row.get('object_key', ''),
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


def write_summary(path: Path, tasks: dict[str, dict], results: dict[str, dict]) -> None:
    summary = {
        'task_count': len(tasks),
        'result_count': len(results),
        'submit_status': dict(Counter(r.get('submit_status') or 'unknown' for r in tasks.values())),
        'result_status': dict(Counter(r.get('status') or 'unknown' for r in results.values())),
        'updated_at': int(time.time()),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> int:
    args = parse_args()
    if not args.user_name:
        raise SystemExit('Missing user name. Pass --user-name or set ADVIBE_USER_NAME.')
    if not args.api_key:
        raise SystemExit('Missing api key. Pass --api-key or set ADVIBE_API_KEY.')
    if args.submit_only and args.poll_only:
        raise SystemExit('--submit-only and --poll-only cannot be used together.')

    prompt = args.prompt_file.read_text(encoding='utf-8')
    run_name = sanitize_name(f"{args.model_name}_{args.prompt_file.stem}_on_bucket")
    tasks_path = args.output_dir / f'tasks_{run_name}.jsonl'
    results_path = args.output_dir / f'results_{run_name}.jsonl'
    predictions_path = args.output_dir / f'predictions_{run_name}.csv'
    summary_path = args.output_dir / f'summary_{run_name}.json'

    tasks = {} if args.overwrite else read_jsonl_map(tasks_path, 'clip_id')
    results = {} if args.overwrite else read_jsonl_map(results_path, 'clip_id')

    # Discover clips via niofs SDK
    niofs_client = niofs.Client(NIOFS_AK, NIOFS_SK)
    clips = discover_clips(niofs_client, args.bucket, args.bucket_prefix)
    if args.sample > 0:
        clips = clips[:args.sample]
    print(f'discovered_clips={len(clips)} from {args.bucket}/{args.bucket_prefix}')

    if not args.poll_only:
        to_submit = [
            c for c in clips
            if args.overwrite or c['clip_id'] not in tasks or not tasks[c['clip_id']].get('task_id')
        ]
        print(f'to_submit={len(to_submit)}')

        if to_submit:
            # 初始化对话客户端：
            # --use-on-bucket 时用 IntranetOnBucketChat，
            # 否则用兼容模式 BaseChat + generate_presigned_url
            if args.use_on_bucket:
                print('mode=IntranetOnBucketChat')
                chat_client = IntranetOnBucketChat(
                    bucket=args.bucket,
                    niofs_ak=NIOFS_AK,
                    niofs_sk=NIOFS_SK,
                    user_name=args.user_name,
                    api_key=args.api_key,
                    prompt=prompt,
                    model_name=args.model_name,
                    fps=args.fps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    model_extra_params={'enable_thinking': True} if args.enable_thinking else None,
                )
            else:
                print('mode=BaseChat (BaseChat + presigned_url internet=False)')
                chat_client = BaseChat(
                    user_name=args.user_name,
                    api_key=args.api_key,
                    prompt=prompt,
                    model_name=args.model_name,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    model_extra_params={'enable_thinking': True} if args.enable_thinking else None,
                )

            with ThreadPoolExecutor(max_workers=max(1, args.submit_workers)) as pool:
                futures: dict = {}
                for clip in to_submit:
                    if args.use_on_bucket:
                        f = pool.submit(submit_one_onbucket, clip, chat_client, args.query, args.fps)
                    else:
                        f = pool.submit(submit_one_basechat, clip, niofs_client, chat_client, args.query, args.fps, args.bucket)
                    futures[f] = clip
                for future in as_completed(futures):
                    clip = futures[future]
                    try:
                        task_row = future.result()
                    except Exception as exc:
                        task_row = {
                            'clip_id': clip['clip_id'],
                            'object_key': clip['object_key'],
                            'task_id': None,
                            'submit_success': False,
                            'submit_status': 'failed',
                            'submit_error': str(exc),
                            'submitted_at': int(time.time()),
                        }
                    tasks[clip['clip_id']] = task_row
                    print(
                        f"submit {clip['clip_id']}: "
                        f"task_id={task_row.get('task_id')} "
                        f"status={task_row.get('submit_status')} "
                        f"success={task_row.get('submit_success')}"
                    )
                    write_jsonl(tasks_path, [tasks[k] for k in sorted(tasks)])
                    write_summary(summary_path, tasks, results)

    if args.submit_only:
        print(f'submit phase done: {tasks_path}')
        return 0

    task_client = GetTask(user_name=args.user_name, api_key=args.api_key)
    pending = {
        clip_id: row for clip_id, row in tasks.items()
        if row.get('task_id') and (
            clip_id not in results or results[clip_id].get('status') not in FINAL_STATUSES
        )
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
