#!/bin/bash
# Qwen3-1.7B, single GPU. Usage: bash serve_qwen3-mini.sh [--save|--load]
# No mode = plain SGLang (also warms the JIT caches before the first --save).
# Archive: ./foundry_archive (foundry.toml). SGL_MODEL / SGL_EXTRA_ARGS override.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/persist.sh"
PERSIST=()
case "${1:-}" in
  --save) foundry_persist save || exit 1 ;;
  --load) foundry_persist load || exit 1 ;;
  "") ;;
  *) echo "Usage: $0 [--save|--load]"; exit 1 ;;
esac
source "$SCRIPT_DIR/set_env.sh"
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3-1.7B}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --tp-size 1 --mem-fraction-static 0.6 --disable-radix-cache \
  --attention-backend flashinfer --cuda-graph-max-bs-decode 512 \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
