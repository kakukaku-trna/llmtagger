"""DashScope API inference client. Adapted from detect_blind_curve.py."""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Tuple

from openai import OpenAI

from pipeline.config import InferenceConfig
from pipeline.data.local_loader import Sample


@dataclass
class InferResult:
    video_path: str
    label: str
    result: str         # "是" | "否" | ""
    reason: str
    status: str         # success | parse_error | failed
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed: float
    error: str = ""


class DashScopeClient:
    def __init__(self, config: InferenceConfig):
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
        )

    def infer(self, sample: Sample, prompt: str) -> InferResult:
        """Run inference on a single video sample."""
        import datetime
        t0 = datetime.datetime.now()

        out = InferResult(
            video_path=sample.video_path,
            label=sample.label,
            result="",
            reason="",
            status="failed",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            elapsed=0.0,
        )

        try:
            b64 = _encode_video(sample.video_path)
            raw, pt, ct = self._call_api(b64, prompt)
            result, reason = _parse_response(raw)
            out.result = result
            out.reason = reason
            out.status = "success"
            out.prompt_tokens = pt
            out.completion_tokens = ct
            out.total_tokens = pt + ct
        except (json.JSONDecodeError, ValueError) as e:
            out.status = "parse_error"
            out.error = str(e)
        except Exception as e:
            out.error = str(e)

        out.elapsed = (datetime.datetime.now() - t0).total_seconds()
        return out

    def _call_api(self, b64: str, prompt: str) -> Tuple[str, int, int]:
        """Call DashScope API with retries. Returns (content, prompt_tokens, completion_tokens)."""
        cfg = self._config
        last_err = ""
        for attempt in range(1, cfg.max_retries + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=cfg.model,
                    temperature=0.0,
                    extra_body={
                        "enable_thinking": True,
                        "thinking_budget": cfg.thinking_budget,
                    },
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                                "fps": cfg.fps,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }],
                )
                content = completion.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("API returned empty content")
                usage = completion.usage
                pt = usage.prompt_tokens if usage else 0
                ct = usage.completion_tokens if usage else 0
                return content, pt, ct
            except Exception as e:
                last_err = str(e)
                if attempt < cfg.max_retries:
                    time.sleep(2)
        raise RuntimeError(f"API failed after {cfg.max_retries} retries: {last_err}")


def _encode_video(video_path: str) -> str:
    """Base64-encode a video file."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_response(content: str) -> Tuple[str, str]:
    """Extract result and reason from LLM JSON response."""
    text = content
    if isinstance(text, list):
        text = text[0].get("text", str(text[0])) if text else ""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {text[:200]}")

    data = json.loads(match.group(0))
    result = data.get("result", "")
    # Normalize English yes/no to Chinese
    result = {"yes": "是", "Yes": "是", "YES": "是",
              "no": "否", "No": "否", "NO": "否"}.get(result, result)
    return result, data.get("reason", "")
