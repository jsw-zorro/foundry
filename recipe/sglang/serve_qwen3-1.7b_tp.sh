#!/bin/bash
# Qwen3-1.7B, tensor parallel (torch symm-mem allreduce in the decode graphs).
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash serve_qwen3-1.7b_tp.sh <tp_size> [--save|--load]
# No mode = plain SGLang. Archive: ./foundry_archive (foundry.toml).
# Decode graphs for every bs 1..MAXBS (default 256) per rank, no padding; prefill graphs off.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/persist.sh"
TP_SIZE=${1:?Usage: $0 <tp_size> [--save|--load]}
PERSIST=()
case "${2:-}" in
  --save) foundry_persist save || exit 1 ;;
  --load) foundry_persist load || exit 1 ;;
  "") ;;
  *) echo "Usage: $0 <tp_size> [--save|--load]"; exit 1 ;;
esac
source "$SCRIPT_DIR/set_env.sh"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP_SIZE - 1)))
exec sglang serve \
  --model-path "${SGL_MODEL:-Qwen/Qwen3-1.7B}" \
  --trust-remote-code --host 0.0.0.0 --port "${PORT:-12000}" \
  --tp-size "$TP_SIZE" --mem-fraction-static 0.6 --disable-radix-cache \
  --attention-backend flashinfer \
  --cuda-graph-max-bs-decode "${MAXBS:-256}" --disable-cuda-graph-padding --cuda-graph-backend-prefill disabled \
  --disable-custom-all-reduce --enable-torch-symm-mem \
  "${PERSIST[@]}" ${SGL_EXTRA_ARGS:-}
