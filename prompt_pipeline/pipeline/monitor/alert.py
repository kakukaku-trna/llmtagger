"""Unified alert dispatcher.

Routes alerts to console, Feishu webhook, and/or email.
Console output is always active. Feishu and email are sent only when
the respective configuration is provided.
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Optional


class AlertLevel(Enum):
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


def emit_alert(
    level: AlertLevel,
    event: str,
    detail: str,
    scene: str = "",
    progress: str = "",
    tokens_used: int = 0,
    token_budget: int = 0,
    feishu_webhook: Optional[str] = None,
    email: Optional[str] = None,
    mention_id: Optional[str] = None,
) -> None:
    """Emit a structured alert to all configured channels."""
    _print_alert(level, event, detail, scene, progress, tokens_used, token_budget)

    fields = {}
    if scene:
        fields["场景"] = scene
    if progress:
        fields["进度"] = progress
    if tokens_used > 0:
        budget_str = f" / {token_budget:,}" if token_budget > 0 else ""
        fields["Token 消耗"] = f"{tokens_used:,}{budget_str}"
    fields["时间"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if feishu_webhook:
        _send_feishu(feishu_webhook, level, event, detail, fields, mention_id)

    if email:
        _send_email(email, level, event, detail, fields)


# ─────────────────────────────────────────────────────────────
# Console
# ─────────────────────────────────────────────────────────────

def _print_alert(level, event, detail, scene, progress, tokens_used, token_budget):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    icon = {"INFO": "✅", "WARNING": "⚠️ ", "ERROR": "🔴"}.get(level.value, "🔔")
    print(f"\n{'─'*45}")
    print(f"{icon} LLMTagger [{level.value}]")
    print(f"{'─'*45}")
    if scene:
        print(f"  场景: {scene}")
    print(f"  事件: {event}")
    print(f"  详情: {detail}")
    if progress:
        print(f"  进度: {progress}")
    if tokens_used > 0:
        budget_str = f" / {token_budget:,}" if token_budget > 0 else ""
        print(f"  Token 消耗: {tokens_used:,}{budget_str}")
    print(f"  时间: {now}")
    print(f"{'─'*45}\n")


# ─────────────────────────────────────────────────────────────
# Feishu
# ─────────────────────────────────────────────────────────────

def _send_feishu(webhook, level, event, detail, fields, mention_id):
    from pipeline.monitor.feishu_alert import FeishuLevel, send_feishu_alert
    mapping = {
        AlertLevel.INFO:    FeishuLevel.INFO,
        AlertLevel.WARNING: FeishuLevel.WARNING,
        AlertLevel.ERROR:   FeishuLevel.ERROR,
    }
    ok = send_feishu_alert(
        webhook=webhook,
        level=mapping.get(level, FeishuLevel.INFO),
        title=event,
        body=detail,
        fields=fields,
        mention_id=mention_id,
    )
    if ok:
        print(f"  ↗ 飞书告警已发送")
    else:
        print(f"  ✗ 飞书告警发送失败，请检查 webhook URL")


# ─────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────

def _send_email(to, level, event, detail, fields):
    from pipeline.monitor.email_alert import build_html_body, default_email_config, send_email_alert
    subject = f"[LLMTagger {level.value}] {event}"
    html = build_html_body(title=event, body=detail, fields=fields)
    cfg = default_email_config()
    ok = send_email_alert(cfg=cfg, to=to, subject=subject, body=detail, html_body=html)
    if ok:
        print(f"  ↗ 邮件告警已发送至 {to}")
    elif cfg is None:
        pass  # stub mode already printed preview
    else:
        print(f"  ✗ 邮件告警发送失败，请检查 SMTP 配置")
