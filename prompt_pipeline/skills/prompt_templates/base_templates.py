"""Base prompt templates for video detection tasks.

Templates use {placeholders} compatible with Python str.format().
Build a complete prompt by calling build_detection_prompt() or by
composing the fragment strings manually.

Design principles:
- Chain-of-thought by default (模型先分析再给出结论)
- Strict JSON output format to simplify parse_response()
- Rules section is isolated so topk_modifier can target it precisely
"""
from __future__ import annotations

from typing import List, Optional


# ─────────────────────────────────────────────────────────────
# Output format fragments
# ─────────────────────────────────────────────────────────────

OUTPUT_FORMAT_BINARY = """\
## 输出格式

请以 JSON 格式输出，不要包含任何额外内容：
```json
{{"result": "是/否", "reason": "简洁的判断依据，不超过100字"}}
```"""

OUTPUT_FORMAT_WITH_SCORE = """\
## 输出格式

请以 JSON 格式输出，不要包含任何额外内容：
```json
{{
  "result": "是/否",
  "confidence": 0.0-1.0,
  "reason": "简洁的判断依据，不超过100字",
  "key_evidence": ["最主要的证据1", "证据2"]
}}
```"""

# ─────────────────────────────────────────────────────────────
# Chain-of-thought instruction
# ─────────────────────────────────────────────────────────────

COT_TEMPLATE = """\
## 分析步骤

请按以下步骤思考（内部推理，不需要输出）：
1. 观察视频整体场景（时间段、天气、道路类型）
2. 识别与 {task_name} 相关的视觉特征
3. 逐条对照判断规则，标记符合和不符合的条目
4. 综合所有证据得出结论"""

# ─────────────────────────────────────────────────────────────
# Full detection template
# ─────────────────────────────────────────────────────────────

DETECTION_TEMPLATE = """\
你是一名自动驾驶场景标注专家，擅长从行车记录仪视频中识别特定场景。

## 任务描述

请判断视频片段中是否存在 **{scene_display_name}**。

{task_description}

## 判断规则

{rules}

{cot_section}

{output_format}
"""


def build_detection_prompt(
    scene_display_name: str,
    task_description: str,
    rules: List[str],
    use_cot: bool = True,
    output_format: str = OUTPUT_FORMAT_BINARY,
    task_name: str = "",
) -> str:
    """Assemble a complete detection prompt from components.

    Args:
        scene_display_name: Chinese display name, e.g. "盲弯"
        task_description: 1-3 sentence description of what to detect
        rules: list of numbered rule strings (without the number prefix)
        use_cot: whether to include the chain-of-thought section
        output_format: output format fragment string
        task_name: used in COT template placeholder

    Returns:
        Complete prompt string ready to send to the LLM.
    """
    numbered_rules = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))

    cot_section = ""
    if use_cot:
        cot_section = COT_TEMPLATE.format(
            task_name=task_name or scene_display_name
        )

    return DETECTION_TEMPLATE.format(
        scene_display_name=scene_display_name,
        task_description=task_description,
        rules=numbered_rules,
        cot_section=cot_section,
        output_format=output_format,
    ).strip()


# ─────────────────────────────────────────────────────────────
# Scene-specific quick builders
# ─────────────────────────────────────────────────────────────

def blind_curve_prompt(extra_rules: Optional[List[str]] = None) -> str:
    """Build a blind_curve detection prompt from base rules + optional extras."""
    from skills.road_geometry.rules import get_rules_for_scene
    ruleset = get_rules_for_scene("blind_curve")
    rules = [r.text for r in ruleset.rules]
    if extra_rules:
        rules.extend(extra_rules)
    return build_detection_prompt(
        scene_display_name="盲弯",
        task_description=(
            "盲弯是指由于地形、建筑或植被遮挡，驾驶员无法提前看清弯道后方路况的弯道。"
            "请根据视频中的视觉线索判断当前路段是否为盲弯。"
        ),
        rules=rules,
        task_name="盲弯",
    )


def waitzone_prompt(direction: str = "left") -> str:
    """Build a waitzone detection prompt for left/right/straight direction."""
    from skills.road_geometry.rules import get_rules_for_scene
    cn_map = {"left": "左转", "right": "右转", "straight": "直行"}
    scene = f"waitzone_{direction}"
    cn = cn_map.get(direction, direction)
    ruleset = get_rules_for_scene(scene)
    rules = [r.text for r in ruleset.rules]
    return build_detection_prompt(
        scene_display_name=f"{cn}待行区",
        task_description=(
            f"待行区是路口允许车辆在红灯期间提前驶入等候的区域。"
            f"请判断视频中是否存在 {cn} 方向的待行区标线或相关信号灯。"
        ),
        rules=rules,
        task_name=f"{cn}待行区",
    )
