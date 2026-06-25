"""Local vLLM inference client. Uses file:// URL to avoid base64 overhead."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

from pipeline.config import InferenceConfig
from pipeline.data.local_loader import Sample
from pipeline.inference.dashscope_client import InferResult, _parse_response


class VLLMClient:
    """Calls a locally running vLLM server (e.g., Qwen2.5-VL)."""

    def __init__(self, config: InferenceConfig):
        self._config = config

    def infer(self, sample: Sample, prompt: str) -> InferResult:
        import datetime

        t0 = datetime.datetime.now()
        raw = ""

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
            raw, pt, ct = self._call_api(sample.video_path, prompt)
            result, reason = _parse_response(raw, self._config.response_mode)
            out.raw_response = raw  # 保存原始响应文本
            out.result = result
            out.reason = reason
            out.status = "success"
            out.prompt_tokens = pt
            out.completion_tokens = ct
            out.total_tokens = pt + ct
        except (json.JSONDecodeError, ValueError) as e:
            out.status = "parse_error"
            out.raw_response = raw  # 即使解析失败也保存原始响应
            out.error = str(e)
        except Exception as e:
            out.error = str(e)

        out.elapsed = (datetime.datetime.now() - t0).total_seconds()
        return out

    def _call_api(self, video_path: str, prompt: str):
        cfg = self._config
        payload = {
            "model": cfg.model,
            "temperature": 0.0,
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": f"file://{video_path}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        last_err = ""
        for attempt in range(1, cfg.max_retries + 1):
            try:
                resp = requests.post(
                    cfg.vllm_url, json=payload, timeout=cfg.timeout_per_sample
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return (
                    content,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                )
            except Exception as e:
                last_err = str(e)
                if attempt < cfg.max_retries:
                    time.sleep(1)
        raise RuntimeError(f"vLLM failed after {cfg.max_retries} retries: {last_err}")

    def health_check(self) -> bool:
        """Check if vLLM server is alive."""
        try:
            resp = requests.get(
                self._config.vllm_url.replace("/v1/chat/completions", "/health"),
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False
