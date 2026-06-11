"""Raw response logger – append-only JSONL for LLM outputs."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline.inference.dashscope_client import InferResult


class RawLogWriter:
    """Thread-safe writer that appends raw LLM responses to a single JSONL file.

    Each pipeline run (single or iterative) gets one log file so the full
    conversation history is easy to grep or stream-analyse later.
    """

    def __init__(self, log_path: str | Path):
        self._path = Path(log_path)
        self._lock = threading.Lock()
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Create (or truncate) the file and write an empty line so readers
        # can tail it immediately.
        self._path.write_text("", encoding="utf-8")

    # ─────────────────────────────────────────────────────────
    def write(
        self,
        result: InferResult,
        *,
        scene: str = "",
        version: str = "",
        round_n: int = 0,
    ) -> None:
        """Append one record to the JSONL log."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scene": scene,
            "version": version,
            "round": round_n,
            "video_path": result.video_path,
            "video_name": Path(result.video_path).name,
            "raw_response": result.raw_response,
            "status": result.status,
            "parsed_result": result.result,
            "parsed_reason": result.reason,
            "elapsed": result.elapsed,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "error": result.error,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def make_log_path(base_dir: str | Path, scene: str) -> Path:
        """Generate a timestamped log path under *base_dir/scene/*."""
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(base_dir) / scene / f"raw_responses_{now}.jsonl"
