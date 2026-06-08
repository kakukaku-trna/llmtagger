"""Feishu (Lark) webhook alert sender.

Sends structured card messages to a Feishu Bot webhook.
Feishu card format docs:
https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/feishu-cards/card-components

Usage:
    from pipeline.monitor.feishu_alert import send_feishu_alert, FeishuLevel
    send_feishu_alert(
        webhook="https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
        level=FeishuLevel.WARNING,
        title="Token 预算告警",
        body="已消耗 85% 预算",
        fields={"场景": "blind_curve", "已用 Token": "170,000"},
    )
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional


class FeishuLevel(Enum):
    INFO    = ("green",  "✅ INFO")
    WARNING = ("yellow", "⚠️ WARNING")
    ERROR   = ("red",    "🔴 ERROR")


def send_feishu_alert(
    webhook: str,
    level: FeishuLevel,
    title: str,
    body: str,
    fields: Optional[Dict[str, str]] = None,
) -> bool:
    """POST a card message to a Feishu webhook.

    Returns True if the request succeeds (HTTP 200 with code=0), False otherwise.
    Raises no exceptions — failures are printed to stderr only.
    """
    import json
    try:
        import urllib.request
        payload = _build_card(level, title, body, fields or {})
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            if resp.status == 200 and resp_body.get("code", -1) == 0:
                return True
            import sys
            print(f"[FeishuAlert] non-zero response: {resp_body}", file=sys.stderr)
            return False
    except Exception as e:
        import sys
        print(f"[FeishuAlert] send failed: {e}", file=sys.stderr)
        return False


def _build_card(
    level: FeishuLevel,
    title: str,
    body: str,
    fields: Dict[str, str],
) -> dict:
    color, prefix = level.value
    elements = [
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": body},
        }
    ]
    if fields:
        field_elems = [
            {
                "tag": "column",
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"**{k}**\n{v}"}}
                ],
            }
            for k, v in fields.items()
        ]
        elements.append({
            "tag": "column_set",
            "flex_mode": "none",
            "background_style": "grey",
            "columns": field_elems[:4],   # Feishu supports max 4 columns
        })
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"{prefix}  {title}"},
                "template": color,
            },
            "elements": elements,
        },
    }
