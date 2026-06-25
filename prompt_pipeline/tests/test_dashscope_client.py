"""Tests for pipeline/inference/dashscope_client.py parsing helpers."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.inference.dashscope_client import _parse_response


def test_parse_json_response():
    result, reason = _parse_response('{"result": "是", "reason": "晴天"}')
    assert result == "是"
    assert reason == "晴天"


def test_parse_text_response():
    text = "视频开始，本车在中间车道匀速前行，前方车流适中。"
    result, reason = _parse_response(text, response_mode="text")
    assert result == text
    assert reason == ""
