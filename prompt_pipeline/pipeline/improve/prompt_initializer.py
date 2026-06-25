"""Generate initial prompt for a scene using LLM, with skills rule library as context."""
from __future__ import annotations

import re
from pathlib import Path

from openai import OpenAI

from pipeline.config import SceneConfig
from pipeline.data.local_loader import _discover_media
from skills.road_geometry.rules import get_rules_for_scene as get_road_rules
from skills.traffic_elements.rules import get_rules_for_scene as get_traffic_rules


_SYSTEM_PROMPT = """\
你是一位自动驾驶场景标注 Prompt 设计专家。你的任务是根据场景描述，生成一个完整的二分类检测 Prompt。

你会同时拿到以下多种线索：
- scene id
- display_name
- 用户额外描述
- positive_dir / negative_dir 的目录名和样本语义
- 少量样本文件名示例

请先在内部完成“任务消歧”：
- `positive_dir` 下样本表示应判“是”
- `negative_dir` 下样本表示应判“否”
- 如果 `display_name`、scene id、用户描述、目录命名之间存在冲突，优先采用与正负样本目录命名更一致的“目标概念”
- 如果 `display_name` 或目录名里带有“多摄 / multicam / Front120 / Front30 / Rear / SideView”等信息，应把它们理解为“输入模态或机位组织方式”的线索
- 不要因为某个字段残留旧名字，就生成完全无关的 Prompt

Prompt 必须严格包含以下结构（Markdown 格式）：

# 任务
角色设定 + 检测任务一句话描述。

## 概念定义
明确目标场景的定义，1-2 句话。

## 关键视觉特征（满足以下任一条件即判"是"）
编号列出 3-5 条判断依据，每条用 **粗体标题** + 冒号 + 说明。

## 判断逻辑
- **判"是"**：满足条件
- **判"否"**：不满足条件

## 注意事项
3-5 条边界情况、误判陷阱、多段路况处理、恶劣天气等说明。

## 输出格式
纯净JSON（无markdown标记、无多余文字）：
{"result": "是/否", "reason": "简要描述判断依据，20字以内"}

直接返回完整的 Prompt 文本，不要包含任何额外说明。
"""

_USER_TEMPLATE = """\
场景基础信息：
{scene_context}

{rules_context}

请为以上场景生成一个完整的检测 Prompt。
"""


def generate_initial_prompt(
    scene_cfg: SceneConfig,
    description: str = "",
) -> str:
    """Use LLM to generate an initial prompt for the scene.

    Args:
        scene_cfg: Scene configuration (uses display_name, name, inference config).
        description: Optional user-provided scene description; falls back to display_name.

    Returns:
        Generated prompt text (Markdown).
    """
    desc = description or scene_cfg.display_name or scene_cfg.name
    rules_ctx = _build_rules_context(scene_cfg.name)
    scene_ctx = _build_scene_context(scene_cfg, desc)

    user_msg = _USER_TEMPLATE.format(
        scene_context=scene_ctx,
        rules_context=rules_ctx,
    )

    client = OpenAI(
        api_key=scene_cfg.inference.api_key,
        base_url=scene_cfg.inference.api_base,
        timeout=120,
    )
    resp = client.chat.completions.create(
        model=scene_cfg.inference.model,
        temperature=0.3,
        extra_body={"enable_thinking": False},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    prompt_text = resp.choices[0].message.content.strip()

    prompt_text = re.sub(r"^```[a-z]*\n?", "", prompt_text)
    prompt_text = re.sub(r"\n?```$", "", prompt_text)
    return prompt_text.strip()


def _build_rules_context(scene: str) -> str:
    """Build rules context string from skills library for the given scene."""
    road_rules = get_road_rules(scene)
    traffic_rules = get_traffic_rules(scene)

    sections = []

    if road_rules.rules:
        lines = [r.text for r in road_rules.rules]
        sections.append("道路几何规则参考：\n" + "\n".join(f"  - {l}" for l in lines))

    if traffic_rules:
        lines = [r.text for r in traffic_rules]
        sections.append("交通元素规则参考：\n" + "\n".join(f"  - {l}" for l in lines))

    if sections:
        return "以下是规则库中与该场景相关的参考规则（可酌情采用或修改）：\n\n" + "\n\n".join(sections)

    return "（规则库中无该场景的预置规则，请根据场景描述自行生成判断规则。）"


def _build_scene_context(scene_cfg: SceneConfig, description: str) -> str:
    """Build rich scene context for init-prompt generation."""
    lines = [
        f"- scene id: {scene_cfg.name}",
        f"- display_name: {scene_cfg.display_name or '(空)'}",
        f"- 用户描述: {description}",
        f"- 数据模式: {scene_cfg.data.mode}",
    ]

    pos_dir = scene_cfg.data.positive_dir
    neg_dir = scene_cfg.data.negative_dir
    pos_name = Path(pos_dir).name if pos_dir else ""
    neg_name = Path(neg_dir).name if neg_dir else ""

    lines.extend(
        [
            "- 标签语义：positive_dir 下样本应判“是”，negative_dir 下样本应判“否”",
            f"- positive_dir: {pos_dir or '(空)'}",
            f"- negative_dir: {neg_dir or '(空)'}",
            f"- positive_dir basename: {pos_name or '(空)'}",
            f"- negative_dir basename: {neg_name or '(空)'}",
        ]
    )

    modality_hints = _infer_modality_hints(scene_cfg, description)
    if modality_hints:
        lines.append(f"- 输入模态线索: {modality_hints}")

    data_summary = _build_data_summary(scene_cfg)
    if data_summary:
        lines.append(data_summary)

    lines.append(
        "- 任务解析要求：请综合 scene id、display_name、用户描述、正负样本目录命名和样本名示例，先推断“什么情况应该判是、什么情况应该判否”，再生成 Prompt。"
    )
    return "\n".join(lines)


def _infer_modality_hints(scene_cfg: SceneConfig, description: str) -> str:
    """Infer likely modality hints from scene metadata text."""
    text = " ".join(
        filter(
            None,
            [
                scene_cfg.name,
                scene_cfg.display_name,
                description,
                scene_cfg.data.positive_dir,
                scene_cfg.data.negative_dir,
            ],
        )
    ).lower()

    hints: list[str] = []
    if "multicam" in text or "多摄" in text:
        hints.append("多摄或拼接视频输入")
    if "front120" in text:
        hints.append("主视角可能是 Front120")
    if "front30" in text:
        hints.append("主视角可能是 Front30")
    if "sideview" in text or "rear" in text:
        hints.append("可能包含侧视或后视机位")
    if not hints:
        return ""
    return "；".join(hints)


def _build_data_summary(scene_cfg: SceneConfig, sample_limit: int = 3) -> str:
    """Summarize positive/negative local dataset naming hints."""
    if scene_cfg.data.mode != "local":
        return ""

    summaries: list[str] = []
    for label, directory in (
        ("正样本", scene_cfg.data.positive_dir),
        ("负样本", scene_cfg.data.negative_dir),
    ):
        if not directory or not Path(directory).exists():
            continue
        try:
            media = _discover_media(directory)
        except Exception:
            continue
        examples = [_format_media_name(p) for p in media[:sample_limit]]
        example_str = "，".join(examples) if examples else "无"
        summaries.append(
            f"- {label}目录中发现约 {len(media)} 条样本，样本名示例：{example_str}"
        )

    return "\n".join(summaries)


def _format_media_name(path: str) -> str:
    """Format media path into compact sample name for LLM context."""
    p = Path(path)
    if p.is_dir():
        return f"{p.name}/(frames)"
    return p.name
