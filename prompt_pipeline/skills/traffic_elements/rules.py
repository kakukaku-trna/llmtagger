"""Reusable traffic element detection rules.

Covers traffic signs, signals, road markings, and dynamic elements
(vehicles, pedestrians, cyclists).

Compose with road_geometry rules to build richer scene prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrafficRule:
    id: str
    category: str   # sign | signal | marking | dynamic
    text: str
    scenes: List[str] = field(default_factory=list)
    weight: float = 1.0


TRAFFIC_ELEMENT_RULES: Dict[str, TrafficRule] = {

    # ── Signs (标志) ─────────────────────────────────────────
    "sign_speed_limit": TrafficRule(
        id="sign_speed_limit",
        category="sign",
        text="可见限速标志牌，数值清晰可读",
        weight=0.9,
    ),
    "sign_no_passing": TrafficRule(
        id="sign_no_passing",
        category="sign",
        text="可见禁止超车/禁止通行标志，双向交通受限",
        scenes=["blind_curve"],
        weight=0.8,
    ),
    "sign_curve_warning": TrafficRule(
        id="sign_curve_warning",
        category="sign",
        text="可见弯道警告标志（黄色菱形/三角形，含弯曲箭头）",
        scenes=["blind_curve"],
        weight=0.9,
    ),
    "sign_pedestrian": TrafficRule(
        id="sign_pedestrian",
        category="sign",
        text="可见注意行人或人行横道标志",
        weight=0.7,
    ),
    "sign_construction": TrafficRule(
        id="sign_construction",
        category="sign",
        text="可见施工警告标志或施工区域锥桶",
        scenes=["construction_curb"],
        weight=1.0,
    ),

    # ── Signals (信号灯) ──────────────────────────────────────
    "signal_red": TrafficRule(
        id="signal_red",
        category="signal",
        text="交通信号灯当前显示红灯，自车须停车等候",
        scenes=["waitzone_left", "waitzone_right", "waitzone_straight"],
        weight=1.0,
    ),
    "signal_green": TrafficRule(
        id="signal_green",
        category="signal",
        text="交通信号灯当前显示绿灯或绿色箭头",
        scenes=["waitzone_left", "waitzone_right", "waitzone_straight"],
        weight=1.0,
    ),
    "signal_arrow_left": TrafficRule(
        id="signal_arrow_left",
        category="signal",
        text="可见左转绿色箭头信号灯",
        scenes=["waitzone_left"],
        weight=1.0,
    ),

    # ── Road markings (路面标线) ───────────────────────────────
    "marking_solid_yellow": TrafficRule(
        id="marking_solid_yellow",
        category="marking",
        text="路面中心线为黄色实线，禁止跨越超车",
        scenes=["blind_curve"],
        weight=0.9,
    ),
    "marking_stop_line": TrafficRule(
        id="marking_stop_line",
        category="marking",
        text="路面可见白色停车线，标示交叉口停车位置",
        scenes=["waitzone_left", "waitzone_right", "waitzone_straight"],
        weight=0.8,
    ),
    "marking_crosswalk": TrafficRule(
        id="marking_crosswalk",
        category="marking",
        text="可见斑马线（人行横道）标线",
        weight=0.7,
    ),
    "marking_arrow": TrafficRule(
        id="marking_arrow",
        category="marking",
        text="路面可见方向导向箭头，指示允许通行方向",
        weight=0.8,
    ),

    # ── Dynamic elements (动态元素) ───────────────────────────
    "dynamic_oncoming": TrafficRule(
        id="dynamic_oncoming",
        category="dynamic",
        text="视频中可见对向来车，增加盲弯风险评估难度",
        scenes=["blind_curve"],
        weight=0.7,
    ),
    "dynamic_pedestrian": TrafficRule(
        id="dynamic_pedestrian",
        category="dynamic",
        text="视频中可见行人在路面或路侧活动",
        weight=0.6,
    ),
    "dynamic_cyclist": TrafficRule(
        id="dynamic_cyclist",
        category="dynamic",
        text="视频中可见骑行者（自行车/电动车）",
        weight=0.6,
    ),
}


def get_rules_for_scene(scene: str) -> List[TrafficRule]:
    """Return traffic rules applicable to the given scene."""
    return [r for r in TRAFFIC_ELEMENT_RULES.values()
            if not r.scenes or scene in r.scenes]


def get_rule(rule_id: str) -> Optional[TrafficRule]:
    return TRAFFIC_ELEMENT_RULES.get(rule_id)
