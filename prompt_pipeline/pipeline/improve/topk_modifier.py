"""Top-K prompt modifier. Uses LLM to revise the K most relevant rules based on failures."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from pipeline.config import InferenceConfig, SceneConfig
from pipeline.evaluate.metrics import Metrics
from pipeline.improve.failure_analyzer import FailureReport


_SYSTEM_PROMPT = """\
你是一位 Prompt 优化专家。你的任务是根据失败案例分析，对现有 Prompt 进行最小化修改。

规则：
1. 只允许修改 Prompt 中最关键的 Top-{topk} 条判断规则/描述，其余内容保持不变
2. 不要改变输出格式要求
3. 修改要针对具体失败模式：FP（误报）说明判断过宽，FN（漏报）说明判断过严
4. 返回完整的修改后 Prompt 文本，不要包含任何额外说明

直接返回修改后的 Prompt 文本（Markdown格式）。
"""

_USER_TEMPLATE = """\
当前 Prompt：
```
{current_prompt}
```

评估指标（当前版本 {version}）：
- Precision: {precision:.1f}%  Recall: {recall:.1f}%  F1: {f1:.1f}%
- TP={TP}  FP={FP}  FN={FN}  TN={TN}

失败案例分析（Top-{topk} 个最典型）：

误报 FP（实为"否"，模型判为"是"）—— 说明判断条件过宽：
{fp_cases}

漏报 FN（实为"是"，模型判为"否"）—— 说明判断条件过严：
{fn_cases}

请修改 Prompt，只改最关键的 {topk} 条规则，其余保持不变。
"""


def generate_next_prompt(
    current_prompt: str,
    failures: FailureReport,
    metrics: Metrics,
    current_version: str,
    topk: int,
    config: InferenceConfig,
) -> str:
    """Call LLM to produce an improved prompt based on failure analysis."""
    fp_text = _format_cases(failures.fp_cases[:topk])
    fn_text = _format_cases(failures.fn_cases[:topk])

    user_msg = _USER_TEMPLATE.format(
        current_prompt=current_prompt,
        version=current_version,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        TP=metrics.TP, FP=metrics.FP, FN=metrics.FN, TN=metrics.TN,
        topk=topk,
        fp_cases=fp_text if fp_text else "（无）",
        fn_cases=fn_text if fn_text else "（无）",
    )

    client = OpenAI(api_key=config.api_key, base_url=config.api_base)
    resp = client.chat.completions.create(
        model=config.model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT.format(topk=topk)},
            {"role": "user", "content": user_msg},
        ],
    )
    new_prompt = resp.choices[0].message.content.strip()

    # Strip markdown code fences if present
    new_prompt = re.sub(r"^```[a-z]*\n?", "", new_prompt)
    new_prompt = re.sub(r"\n?```$", "", new_prompt)
    return new_prompt.strip()


def save_new_version(
    scene_cfg: SceneConfig,
    new_prompt: str,
    from_version: str,
    new_version: str,
    failures: FailureReport,
    metrics_delta: Optional[dict] = None,
) -> Path:
    """Save the new prompt to prompts/{scene}/{new_version}.md and record in history.json."""
    prompt_path = scene_cfg.prompt_path(new_version)
    prompt_path.write_text(new_prompt, encoding="utf-8")

    _append_history(
        history_path=scene_cfg.history_path(),
        new_version=new_version,
        from_version=from_version,
        fp_count=len(failures.fp_cases),
        fn_count=len(failures.fn_cases),
        metrics_delta=metrics_delta or {},
    )

    return prompt_path


def _append_history(
    history_path: Path,
    new_version: str,
    from_version: str,
    fp_count: int,
    fn_count: int,
    metrics_delta: dict,
) -> None:
    history = []
    if history_path.exists() and history_path.stat().st_size > 2:
        with open(history_path, encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    entry = {
        "version": new_version,
        "from": from_version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "failure_count": {"FP": fp_count, "FN": fn_count},
        "metrics_delta": metrics_delta,
    }
    history.append(entry)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def next_version_name(current: str) -> str:
    """Increment version string: v1 → v2, v3 → v4."""
    m = re.match(r"v(\d+)$", current)
    if m:
        return f"v{int(m.group(1)) + 1}"
    return current + "_next"


def _format_cases(cases) -> str:
    lines = []
    for i, c in enumerate(cases, 1):
        name = Path(c.video_path).parent.name[:20]
        lines.append(f"  {i}. [{name}] 模型理由: {c.reason[:80]}")
    return "\n".join(lines)
