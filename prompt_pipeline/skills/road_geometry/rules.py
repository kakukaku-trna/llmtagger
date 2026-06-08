"""Reusable road geometry detection rules.

These rule fragments are designed to be composed into scene-specific prompts.
Each rule is a self-contained Chinese natural-language sentence describing a
geometric feature of the road that may or may not be present in the video.

Usage:
    from skills.road_geometry.rules import get_rules_for_scene
    rules = get_rules_for_scene("blind_curve")
    prompt = "请判断视频中是否存在盲弯。\\n判断规则：\\n" + rules.as_numbered_list()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Rule:
    id: str
    text: str
    scenes: List[str] = field(default_factory=list)   # applicable scenes
    weight: float = 1.0                                 # for top-K selection


@dataclass
class RuleSet:
    scene: str
    rules: List[Rule]

    def as_numbered_list(self, prefix: str = "") -> str:
        lines = []
        for i, r in enumerate(self.rules, 1):
            lines.append(f"{i}. {prefix}{r.text}")
        return "\n".join(lines)

    def top_k(self, k: int) -> "RuleSet":
        sorted_rules = sorted(self.rules, key=lambda r: r.weight, reverse=True)
        return RuleSet(scene=self.scene, rules=sorted_rules[:k])


# ─────────────────────────────────────────────────────────────
# Rule library
# ─────────────────────────────────────────────────────────────

ROAD_GEOMETRY_RULES: Dict[str, Rule] = {

    # ── Blind curve (盲弯) ────────────────────────────────────
    "bc_radius": Rule(
        id="bc_radius",
        text="弯道曲率半径小，转弯半径明显短于普通道路，驾驶员视线受限",
        scenes=["blind_curve"],
        weight=1.0,
    ),
    "bc_sight_blocked": Rule(
        id="bc_sight_blocked",
        text="弯道外侧存在山体、建筑、植被或高路堤等遮挡物，使弯道后方路段不可见",
        scenes=["blind_curve"],
        weight=1.0,
    ),
    "bc_road_marking": Rule(
        id="bc_road_marking",
        text="路面设有盲弯警示标线（黄色实线、双黄线）或减速标志",
        scenes=["blind_curve"],
        weight=0.8,
    ),
    "bc_guardrail": Rule(
        id="bc_guardrail",
        text="弯道外侧设有护栏或防撞墩，表明该处为事故多发危险路段",
        scenes=["blind_curve"],
        weight=0.7,
    ),

    # ── Narrow road (窄路) ───────────────────────────────────
    "narrow_width": Rule(
        id="narrow_width",
        text="车道宽度不足两车并行，单车道或路面明显狭窄",
        scenes=["narrow_curb"],
        weight=1.0,
    ),
    "narrow_edge": Rule(
        id="narrow_edge",
        text="道路两侧无明显路肩，路基边缘直接临近路面",
        scenes=["narrow_curb"],
        weight=0.8,
    ),

    # ── Curb / road edge (路沿/路边) ─────────────────────────
    "curb_height": Rule(
        id="curb_height",
        text="路沿石高度明显，与机动车道存在明确的物理分隔",
        scenes=["curb", "lateral_curb"],
        weight=1.0,
    ),
    "curb_continuity": Rule(
        id="curb_continuity",
        text="路沿石连续延伸，无明显缺口或破损，限制车辆轨迹",
        scenes=["curb", "lateral_curb"],
        weight=0.9,
    ),
    "curb_lateral": Rule(
        id="curb_lateral",
        text="侧方路沿石位于自车侧边而非前方，对变道或靠边停车有影响",
        scenes=["lateral_curb"],
        weight=1.0,
    ),

    # ── Intersection / wait zone (待行区) ────────────────────
    "wz_marking": Rule(
        id="wz_marking",
        text="路面存在前方待行区标线（白色框线 + 前方字样）",
        scenes=["waitzone_left", "waitzone_right", "waitzone_straight"],
        weight=1.0,
    ),
    "wz_signal": Rule(
        id="wz_signal",
        text="可见待行区专用信号灯（绿色箭头或独立小灯），允许驶入待行区",
        scenes=["waitzone_left", "waitzone_right", "waitzone_straight"],
        weight=1.0,
    ),
    "wz_direction_left": Rule(
        id="wz_direction_left",
        text="待行区为左转方向，路面箭头或标牌指向左转",
        scenes=["waitzone_left"],
        weight=1.0,
    ),
    "wz_direction_right": Rule(
        id="wz_direction_right",
        text="待行区为右转方向，路面箭头或标牌指向右转",
        scenes=["waitzone_right"],
        weight=1.0,
    ),
    "wz_direction_straight": Rule(
        id="wz_direction_straight",
        text="待行区为直行方向，路面直行箭头指示",
        scenes=["waitzone_straight"],
        weight=1.0,
    ),
}


def get_rules_for_scene(scene: str) -> RuleSet:
    """Return all rules applicable to the given scene name."""
    rules = [r for r in ROAD_GEOMETRY_RULES.values() if scene in r.scenes]
    return RuleSet(scene=scene, rules=rules)


def get_rule(rule_id: str) -> Optional[Rule]:
    return ROAD_GEOMETRY_RULES.get(rule_id)
