# prompt_pipeline — Claude Code 说明

## 项目概述

多模态视频标注流水线，用于自动驾驶安全场景检测。通过 DashScope/Qwen 视觉模型对视频片段进行批量推理并评估 P/R/F1，支持自动 prompt 生成与迭代优化。

## 目录结构

```
scenes/                   — 每个场景一个 YAML 配置（数据路径、推理参数、迭代策略）
prompts/                  — 各场景的 prompt 版本（v1.md, v2.md...）及 metrics.json / history.json
pipeline/                 — 核心代码
  ├── config.py           — YAML → dataclass 配置加载（含 ${VAR} 环境变量占位符）
  ├── data/               — 数据加载（local_loader / stream_loader / modality 多模态编码）
  ├── inference/          — 推理客户端（dashscope_client / vllm_client / batch_runner 并发调度）
  ├── evaluate/           — P/R/F1 评估、混淆矩阵、metrics.json 持久化
  ├── improve/            — Prompt 自动生成与迭代优化（初始生成 / 失败归因 / Top-K 规则修改 / Markdown 解析）
  └── monitor/            — Token 计数与预算告警（控制台 / 飞书 / 邮件）
skills/                   — 可复用规则与模板库（road_geometry / traffic_elements / prompt_templates）
scripts/                  — 辅助脚本（ADW 视频下载、eval_benchmark 标注等）
flows/                    — Prefect 流编排（可选，无 Prefect 时降级为顺序执行）
run.py                    — 主入口（自动加载 .env）
.env                      — API key 等密钥（不提交 git）
```

## 核心模块说明

| 模块 | 职责 |
|------|------|
| `pipeline/data/local_loader` | 扫描 `positive_dir`/`negative_dir` 下的 `.mp4` 或 `frame_*.jpg`，构造 `Sample` 列表 |
| `pipeline/data/modality/` | 多模态输入编码：image（PIL 缩放 base64）、video（ffmpeg 关键帧提取）、bev（鸟瞰图）、pointcloud（点云渲染）、topic（传感器元数据） |
| `pipeline/inference/dashscope_client` | DashScope API 客户端，含重试、token 统计、JSON 响应解析 |
| `pipeline/inference/vllm_client` | 本地 vLLM 客户端，用 `file://` URL 避 base64 开销 |
| `pipeline/inference/batch_runner` | `ThreadPoolExecutor` 并发推理、超时检测、结果缓存（JSON 持久化） |
| `pipeline/evaluate/metrics` | 计算 P/R/F1/混淆矩阵，支持多版本对比和 best_version 选取 |
| `pipeline/improve/failure_analyzer` | 区分 FP（误报）/ FN（漏报），生成 `FailureReport` |
| `pipeline/improve/prompt_initializer` | LLM 自动生成初始 Prompt，接入 skills 规则库作为上下文 |
| `pipeline/improve/topk_modifier` | 基于失败案例调 LLM 重写 Top-K 规则，生成 `v(n+1).md` 并追加 `history.json` |
| `pipeline/improve/prompt_parser` | 将 Markdown Prompt 解析为 Section/Rule 结构（支持按规则定位替换） |
| `pipeline/monitor/token_tracker` | 线程安全 token 计数，支持轮次/全局预算阈值告警 |
| `pipeline/monitor/alert` | 统一告警分发（控制台 + 飞书 + 邮件） |
| `skills/` | 规则集与 Prompt 模板（road_geometry / traffic_elements / prompt_templates） |
| `flows/` | Prefect 编排：`scene_flow`（单场景评估+迭代）、`multi_scene_flow`（多场景并发+汇总） |

## 快速开始

```bash
cd /home/huajiang.sun/llmtagger/prompt_pipeline

# 方式一：全自动流程（推荐）
# 1. 写好 scenes/{scene}.yaml（只需数据路径 + 模型配置）
# 2. LLM 自动生成初始 prompt
python3 run.py --scene {scene} --init-prompt --description "场景描述，如：检测视频中的动物"
# 3. 跑评估
python3 run.py --scene {scene} --sample 10
# 4. 开启 prompt 迭代优化
python3 run.py --scene {scene} --iterate

# 方式二：手动写 prompts/{scene}/v1.md，直接评估
python3 run.py --scene bus_lane --sample 5

# 全量推理
python3 run.py --scene bus_lane

# 开启 prompt 迭代优化
python3 run.py --scene bus_lane --iterate
```

### `--init-prompt` 自动生成 Prompt

用户只需提供一句场景描述，LLM 会基于 `skills/` 规则库自动生成结构完整的初始 Prompt（任务 → 概念定义 → 关键视觉特征 → 判断逻辑 → 注意事项 → 输出格式），无需手写。

```bash
# 生成初始 prompt（默认版本 v1，已存在则需 --force 覆盖）
python3 run.py --scene animal --init-prompt --description "检测行车记录仪视频中是否出现动物"

# 覆盖已存在的 v1.md
python3 run.py --scene animal --init-prompt --description "..." --force

# 生成后可直接迭代
python3 run.py --scene animal --iterate --max-rounds 3
```

规则库接入逻辑：
- 有专属规则的场景（如 `blind_curve`）：LLM 拿到道路几何 + 交通元素规则作为参考
- 无专属规则的场景（如 `animal`、天气类）：LLM 拿到通用交通规则，自行生成场景特定规则

## API Key 配置

`.env` 文件在项目根目录，`run.py` 启动时自动加载。**切勿在本文档或任何源码中写入真实密钥**，所有密钥只在 `.env` 中维护：

```bash
# .env 内容示例（实际值请填入 .env，勿提交到 git）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
ADW_USER=<your_adw_user>
ADW_PROD_PASS=<your_adw_prod_pass>
ADW_STG_PASS=<your_adw_stg_pass>
```

`.env` 已加入 `.gitignore`，不会提交到 git。

## 已有场景

| 场景名 | 描述 | 引擎/模型 |
|--------|------|-----------|
| `animal` | 动物检测 | dashscope / qwen3.7-plus |
| `blind_curve` | 大曲率盲区检测 | dashscope / qwen3.7-plus |
| `blind_curve_vllm` | 大曲率盲区检测（本地 vLLM） | vllm / Qwen2.5-VL-3B-Instruct |
| `bus_lane` | 公交车道检测 | dashscope / qwen3.7-plus |
| `duoyun` | 多云检测 | dashscope / qwen3.7-plus |
| `qiangfanshe` | 强反射（白天逆光）检测 | dashscope / qwen3.7-plus |
| `qingtian` | 晴天检测 | dashscope / qwen3.7-plus |
| `wutian` | 雾天检测 | dashscope / qwen3.7-plus |
| `xuetian` | 雪天检测 | dashscope / qwen3.7-plus |
| `yutian` | 雨天检测 | dashscope / qwen3.7-plus |
| `you_feijidongchedao` | 有非机动车道检测 | dashscope / qwen3.7-plus |

## 从 ADW 下载视频

使用通用下载脚本 `scripts/download_adw_videos.py`，支持任意 UUID 列表文件：

**默认模式：提取高分辨率帧（推荐）**

```bash
# 下载并提取 5 帧高分辨率 JPEG（原生分辨率）
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python3 scripts/download_adw_videos.py \
    --uuid-file /home/huajiang.sun/公交车道-150clip.txt \
    --output-dir /home/huajiang.sun/model_muse/bus_lane_pos \
    --workers 6

# 将已下载的 MP4 转为帧模式（保留 MP4）
python3 scripts/download_adw_videos.py \
    --output-dir /home/huajiang.sun/data/晴天 \
    --from-mp4 --frame-count 5 --keep-mp4

# 将已下载的 MP4 转为帧模式（删除 MP4）
python3 scripts/download_adw_videos.py \
    --output-dir /home/huajiang.sun/data/晴天 \
    --from-mp4 --frame-count 5

# 旧模式：720p H264 MP4 视频
python3 scripts/download_adw_videos.py \
    --uuid-file /home/huajiang.sun/公交车道-150clip.txt \
    --output-dir /home/huajiang.sun/model_muse/bus_lane_pos \
    --mp4 --workers 6
```

**输出格式**：
- 帧模式（默认）：`{output_dir}/{uuid}/frame_01.jpg ... frame_05.jpg` + `_meta.json`
- MP4 模式（`--mp4`）：`{output_dir}/{uuid}/{uuid}_{camera}.mp4`

### 下载脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--uuid-file` | UUID 列表文本文件（下载模式必填，--from-mp4 不需要） | - |
| `--output-dir` | 输出目录（必填） | - |
| `--camera` | 相机名称 | `Front30` |
| `--workers` | 并发数 | `6` |
| `--sample` | 只处理前 N 条（测试用） | - |
| `--frame-count` | 每个视频提取帧数 | `5` |
| `--mp4` | 使用旧的 720p MP4 模式（而非帧提取） | 关闭 |
| `--from-mp4` | 从已有 MP4 文件提取帧（不下载） | 关闭 |
| `--keep-mp4` | 提取帧后保留原始 MP4 文件 | 关闭 |
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
3. 自动生成初始 prompt：`python3 run.py --scene {scene} --init-prompt --description "场景描述"`
4. 运行：`python3 run.py --scene {scene} --sample 10`
5. 或直接迭代优化：`python3 run.py --scene {scene} --iterate`

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
