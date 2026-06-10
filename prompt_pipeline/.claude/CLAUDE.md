# prompt_pipeline — Claude Code 说明

## 项目概述

多模态视频标注流水线，用于自动驾驶安全场景检测。通过 DashScope/Qwen 视觉模型对视频片段进行批量推理并评估 P/R/F1，支持自动 prompt 迭代优化。

## 目录结构

```
scenes/          — 每个场景一个 YAML 配置（数据路径、推理参数、迭代策略）
prompts/         — 各场景的 prompt 版本（v1.md, v2.md...）及 metrics.json
pipeline/        — 核心代码（data 加载、推理、评估、优化）
scripts/         — 辅助脚本（视频下载等）
data_uuid/       — UUID 列表文件
flows/           — Prefect 流编排（可选）
run.py           — 主入口（自动加载 .env）
.env             — API key 等密钥（不提交 git）
```

## 快速开始

```bash
cd /home/huajiang.sun/llmtagger/prompt_pipeline

# .env 已配置好，直接运行：
python3 run.py --scene bus_lane --sample 5

# 全量推理
python3 run.py --scene bus_lane

# 开启 prompt 迭代优化
python3 run.py --scene bus_lane --iterate
```

## API Key 配置

`.env` 文件在项目根目录，`run.py` 启动时自动加载：

```bash
# .env 内容
DASHSCOPE_API_KEY=sk-4cc9e93494ca4b8e88f2f7b071e57db3
ADW_USER=dayun.shen
ADW_PROD_PASS=ORSGVU6EF9
ADW_STG_PASS=G4IDTL2YJW
```

`.env` 已加入 `.gitignore`，不会提交到 git。

## 已有场景

| 场景名 | 描述 | 数据路径 |
|--------|------|----------|
| `blind_curve` | 大曲率盲区检测 | `/home/huajiang.sun/model_muse/positive_data` |
| `bus_lane` | 公交车道检测 | `/home/huajiang.sun/model_muse/bus_lane_pos` (150 视频) |

## 从 ADW 下载视频

使用通用下载脚本 `scripts/download_adw_videos.py`，支持任意 UUID 列表文件：

```bash
# 下载公交车道数据（ADW_PROD_PASS 从 .env 自动读取）
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python3 scripts/download_adw_videos.py \
    --uuid-file /home/huajiang.sun/公交车道-150clip.txt \
    --output-dir /home/huajiang.sun/model_muse/bus_lane_pos \
    --workers 6

# 测试 5 条
python3 scripts/download_adw_videos.py \
    --uuid-file data_uuid/bus_lane_true.txt \
    --output-dir /tmp/test_output \
    --sample 5
```

**输出格式**（local_loader 兼容）：
```
{output_dir}/{uuid}/{uuid}_{camera}.mp4
```

### 下载脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--uuid-file` | UUID 列表文本文件（必填） | - |
| `--output-dir` | 输出目录（必填） | - |
| `--camera` | 相机名称 | `Front30` |
| `--workers` | 并发数 | `6` |
| `--sample` | 只处理前 N 条（测试用） | - |
| `--adw-user` | ADW 用户名 | `$ADW_USER` |
| `--adw-prod-pass` | ADW 生产密码 | `$ADW_PROD_PASS` |

## adwsdk 安装与兼容性

adwsdk 原本只为 Python 3.8 安装。在 Python 3.12 下使用的解决方案：

```bash
# 1. 复制 site-packages（一次性操作，已完成）
cp -r ~/.local/lib/python3.8/site-packages/adwsdk ~/.local/lib/python3.12/site-packages/
cp -r ~/.local/lib/python3.8/site-packages/niofs ~/.local/lib/python3.12/site-packages/
cp -r ~/.local/lib/python3.8/site-packages/{crcmod,boto3,botocore,s3transfer,jmespath,elasticsearch,elasticsearch_dsl} \
    ~/.local/lib/python3.12/site-packages/
```

**必须设置的环境变量**（已在脚本内自动设置）：
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 已修复的 adwsdk Bug

**文件**：`~/.local/lib/python3.12/site-packages/adwsdk/cmm.py`

**问题**：新版 protobuf 中 map 字段返回 `RepeatedCompositeFieldContainer` 而非 dict，调用 `.items()` 会失败。

**修复**（`__trans_map` 和 `__trans_data` 方法）：
```python
def __trans_map(self, protomap):
    m = {}
    if not protomap:
        return m
    try:
        items = protomap.items()
    except AttributeError:
        items = ((item.key, item.value) for item in protomap)
    for k, v in items:
        m[k] = get_value(v)
    return m
```

### adwsdk 启动慢

adwsdk 导入耗时约 41 秒（protobuf pure-python 模式），属正常现象，每次进程启动只发生一次。

## 视频下载常见问题

### API 400 错误："String value length exceeded"

**原因**：MP4 文件太大，base64 后超过 DashScope API 字符串长度限制（约 28MB）。

**解决方案**：下载后将 4K H265 重新编码为 720p H264：
```
ffmpeg -i input.h265 -vf scale=1280:720 -vcodec libx264 -crf 38 -preset veryfast -an output.mp4
```
原始 4K H265 约 31MB，720p H264 约 1-2MB，满足 API 限制。

### h264_nvenc 不可用

A100 是计算卡（A100-SXM4），没有 NVENC 硬件编码器。不能用 `h264_nvenc`，使用 CPU `libx264` + `veryfast` 预设（约 7 秒/文件）。

### vcodec copy 文件太大

`-vcodec copy` 直接封装 H265 不重新编码，产生 ~31MB 文件，超过 API 限制。不能用 copy 模式。

### 残留损坏 MP4 被误跳过

若 ffmpeg 中途被杀，会留下不完整的 MP4 文件（几字节到几 MB）。脚本已加入大小验证：
```python
if mp4_dest.exists() and mp4_dest.stat().st_size > 100_000:  # 才算有效
    skip
```

### adwsdk 不能使用 boto3/niofs 模块

botocore（Python 3.8 版本）与 Python 3.12 的 urllib3 不兼容（`cannot import name 'DEFAULT_CIPHERS'`）。
**解决**：只使用 `adwsdk.adw_client.AdwClient` 和 `adwsdk.cmm`，不 import `adwsdk.dlb` 或 `adwsdk.discovery`。

## 新增场景流程

1. 准备视频到本地目录，按 `{uuid}/{uuid}_{camera}.mp4` 组织
2. 在 `scenes/` 创建 `{scene}.yaml`（参考 `bus_lane.yaml`）
3. 在 `prompts/{scene}/` 创建 `v1.md`
4. 运行：`python3 run.py --scene {scene} --sample 10`

## 关键配置项

| 字段 | 说明 |
|------|------|
| `data.positive_dir` / `negative_dir` | 视频目录；空路径自动跳过 |
| `inference.model` | 当前使用 `qwen3.7-plus` |
| `inference.workers` | 并发推理数，建议 3-5 |
| `iteration.max_rounds` | prompt 自动优化轮数 |

## 输出

- 推理结果：`prompts/{scene}/metrics.json`
- 迭代历史：`prompts/{scene}/history.json`
- 优化后 prompt：`prompts/{scene}/v2.md`, `v3.md` ...
