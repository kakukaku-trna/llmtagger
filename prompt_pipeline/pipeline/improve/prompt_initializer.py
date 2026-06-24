"""Generate initial prompt for a scene using LLM, with skills rule library as context."""
from __future__ import annotations

import re

from openai import OpenAI

from pipeline.config import SceneConfig
from skills.road_geometry.rules import get_rules_for_scene as get_road_rules
from skills.traffic_elements.rules import get_rules_for_scene as get_traffic_rules


_SYSTEM_PROMPT = """\
你是一位自动驾驶场景标注 Prompt 设计专家。你的任务是根据场景描述，生成一个完整的二分类检测 Prompt。

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
场景名称：{scene_name}
场景描述：{scene_description}

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
    scene_name = scene_cfg.display_name or scene_cfg.name
    desc = description or scene_cfg.display_name or scene_cfg.name

    rules_ctx = _build_rules_context(scene_cfg.name)

    user_msg = _USER_TEMPLATE.format(
        scene_name=scene_name,
        scene_description=desc,
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
