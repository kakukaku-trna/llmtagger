# prompt_pipeline

多模态视频标注 Pipeline，基于 DashScope API（Qwen 系列模型），支持自动化批量推理、P/R/F1 评估、Prompt 迭代优化和多场景并发编排。

## 目录结构

```
prompt_pipeline/
├── pipeline/                        # 核心逻辑
│   ├── config.py                    # YAML → dataclass 配置加载
│   ├── data/
│   │   ├── local_loader.py          # 本地视频扫描与采样
│   │   ├── stream_loader.py         # 远程流式拉取（通过 UUID 查询数据集 API）
│   │   └── modality/
│   │       ├── image.py             # 图像加载、缩放、base64 编码
│   │       ├── video.py             # 关键帧提取（ffmpeg）、视频压缩
│   │       ├── bev.py               # BEV 鸟瞰图加载与编码
│   │       ├── pointcloud.py        # 点云 → BEV/前视渲染（matplotlib）
│   │       └── topic.py             # 传感器 Topic 元数据提取与文本化
│   ├── inference/
│   │   ├── dashscope_client.py      # DashScope API 客户端（含 token 统计、重试）
│   │   ├── vllm_client.py           # 本地 vLLM 客户端
│   │   └── batch_runner.py          # 并发批量推理、超时检测、结果缓存
│   ├── evaluate/
│   │   └── metrics.py               # P/R/F1/混淆矩阵，持久化到 metrics.json
│   ├── improve/
│   │   ├── failure_analyzer.py      # FP/FN 归因分析
│   │   ├── topk_modifier.py         # LLM 驱动 Top-K 规则修改，生成新版本 Prompt
│   │   └── prompt_parser.py         # Prompt Markdown 结构化解析（Section/Rule/Diff）
│   └── monitor/
│       ├── token_tracker.py         # 线程安全 token 计数与预算告警
│       ├── alert.py                 # 统一告警分发（控制台 + 飞书 + 邮件）
│       ├── feishu_alert.py          # 飞书卡片 webhook 发送
│       └── email_alert.py           # SMTP 邮件发送
│
├── skills/                          # 可复用规则与模板库
│   ├── road_geometry/
│   │   └── rules.py                 # 盲弯/窄路/路沿/待行区几何规则集
│   ├── traffic_elements/
│   │   └── rules.py                 # 交通标志/信号灯/标线/动态元素规则集
│   └── prompt_templates/
│       └── base_templates.py        # 检测任务通用模板、CoT 模板、快捷 builder
│
├── flows/                           # Prefect 编排（无 Prefect 时自动降级为顺序执行）
│   ├── scene_flow.py                # 单场景评估 + 迭代 flow
│   └── multi_scene_flow.py          # 多场景并发 flow + 汇总报告
│
├── scenes/
│   └── blind_curve.yaml             # 场景配置（数据路径、模型、目标、迭代策略）
│
├── prompts/
│   └── {scene}/
│       ├── v1.md                    # 初始 Prompt
│       ├── metrics.json             # 每个版本的 P/R/F1 记录
│       └── history.json             # Prompt 迭代变更历史
│
├── tests/                           # pytest 单元测试 + 集成测试（52 个用例）
├── run.py                           # 统一命令行入口
└── requirements.txt
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
# 可选（按需安装）:
pip install pillow numpy matplotlib   # modality/ 模块依赖
pip install prefect                   # flows/ Prefect 编排
```

### 单场景评估（全量样本）

```bash
cd prompt_pipeline
python run.py --scene blind_curve
```

### 限定样本数

```bash
python run.py --scene blind_curve --sample 10
```

### Prompt 迭代优化

```bash
python run.py --scene blind_curve --iterate --max-rounds 3
```

### 通过 Prefect Flow 运行单场景

```bash
python flows/scene_flow.py --scene blind_curve --sample 20
```

### 多场景并发评估

```bash
python flows/multi_scene_flow.py --scenes blind_curve waitzone_left --sample 20
```

### 运行测试

```bash
python3.12 -m pytest tests/ -v
# 期望: 52 passed
```

---

## 模块说明

### `pipeline/data/`

| 文件 | 用途 |
|------|------|
| `local_loader.py` | 扫描本地 `positive_dir`/`negative_dir` 下的 `.mp4` 文件，打标签"是"/"否" |
| `stream_loader.py` | 按 UUID 列表从远程数据集 API 拉取视频（需实现 `_api_get_video_url()`） |
| `modality/image.py` | PIL 图像加载、最长边缩放、base64 编码 |
| `modality/video.py` | ffmpeg 关键帧提取（可配置 fps / max_frames）、视频压缩 |
| `modality/bev.py` | 从样本目录加载 BEV 鸟瞰图 |
| `modality/pointcloud.py` | `.bin`（KITTI float32）/ `.pcd`（ASCII）点云加载，BEV 渲染为 JPEG base64 |
| `modality/topic.py` | 读取 `topic.json`/`meta.yaml`，提取自车速度、信号灯状态等，格式化为文本上下文 |

### `pipeline/improve/`

| 文件 | 用途 |
|------|------|
| `failure_analyzer.py` | 区分 FP（误报）/ FN（漏报），返回 `FailureReport` |
| `topk_modifier.py` | 将失败案例 + 当前指标发给 LLM，生成新版本 Prompt，写入 `v(n+1).md` |
| `prompt_parser.py` | 将 Markdown Prompt 解析为 `ParsedPrompt`，支持按 Section/Rule 精确定位和替换 |

### `pipeline/monitor/`

| 文件 | 用途 |
|------|------|
| `token_tracker.py` | 线程安全计数，支持轮次/全局预算阈值告警 |
| `alert.py` | 统一入口，自动分发到控制台、飞书、邮件 |
| `feishu_alert.py` | 发送飞书卡片消息（HTTP POST + JSON Card 格式） |
| `email_alert.py` | SMTP 发送 HTML 邮件（`EmailConfig` 为空时打印预览） |

### `skills/`

| 模块 | 用途 |
|------|------|
| `road_geometry/rules.py` | 盲弯、窄路、路沿、待行区的几何判断规则集，按场景筛选 |
| `traffic_elements/rules.py` | 交通标志、信号灯、路面标线、动态元素规则集 |
| `prompt_templates/base_templates.py` | 通用检测 Prompt 模板、CoT 步骤模板、`build_detection_prompt()` / `blind_curve_prompt()` 等快捷 builder |

### `flows/`

| 文件 | 用途 |
|------|------|
| `scene_flow.py` | 单场景评估 + 迭代，支持 Prefect `@flow`/`@task` 或纯函数降级 |
| `multi_scene_flow.py` | 多场景并发（Prefect `.submit()`）或顺序执行，输出跨场景汇总表 |

---

## 场景配置（`scenes/{scene}.yaml`）

| 字段 | 说明 |
|------|------|
| `data.positive_dir` | 正样本目录（label="是"） |
| `data.negative_dir` | 负样本目录（label="否"） |
| `inference.engine` | `dashscope` 或 `vllm` |
| `inference.model` | 如 `qwen3.7-plus` |
| `inference.workers` | 并发线程数 |
| `target.precision/recall` | 迭代停止目标（%） |
| `iteration.max_rounds` | 最大迭代轮次 |
| `iteration.improve_topk` | 每轮最多修改的规则条数 |
| `token_budget.*` | 预算上限与告警阈值 |

## 扩展新场景

1. 新建 `scenes/{scene}.yaml`，填写数据路径和推理参数
2. 新建 `prompts/{scene}/v1.md`，写初始 Prompt
   - 可用 `skills/prompt_templates/base_templates.py` 中的 builder 快速生成
   - 可从 `skills/road_geometry/rules.py` 或 `skills/traffic_elements/rules.py` 组合规则
3. 运行：`python run.py --scene {scene}`
