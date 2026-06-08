"""Reusable prompt template fragments."""
from skills.prompt_templates.base_templates import (
    DETECTION_TEMPLATE,
    COT_TEMPLATE,
    OUTPUT_FORMAT_BINARY,
    OUTPUT_FORMAT_WITH_SCORE,
    build_detection_prompt,
)

__all__ = [
    "DETECTION_TEMPLATE",
    "COT_TEMPLATE",
    "OUTPUT_FORMAT_BINARY",
    "OUTPUT_FORMAT_WITH_SCORE",
    "build_detection_prompt",
]
