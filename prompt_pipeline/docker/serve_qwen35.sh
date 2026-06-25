#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data-algorithm/model_cache/Qwen/Qwen3.5-9B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
LOG_LEVEL="${LOG_LEVEL:-info}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model path not found: $MODEL_PATH" >&2
  echo "Mount the downloaded Qwen3.5-9B directory into the container." >&2
  exit 1
fi

exec transformers serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --reasoning off \
  --trust-remote-code \
  --log-level "$LOG_LEVEL"
