# Start and stop a vLLM server inside a Slurm job. Source after common.sh.
#   vllm_start <log> <model> <revision> <served-name> <port> [extra vllm args...]
#   vllm_wait_ready <port> [max seconds, default 3600]
#   vllm_stop
# Deterministic (batch-invariant) mode is on unless BATCH_INVARIANT=0. The date gpt-oss sees is
# pinned to SYSTEM_START_DATE (common.sh), never the server's clock.

vllm_start() {
  local log="$1" model="$2" rev="$3" served="$4" port="$5"
  shift 5
  local parsers=()
  case "$model" in
    # FlashAttention drifts when several long-prompt requests decode together, even in
    # batch-invariant mode; the Triton backend does not.
    openai/gpt-oss-*) parsers=(--tool-call-parser openai --enable-auto-tool-choice --attention-backend TRITON_ATTN) ;;
    Qwen/Qwen3*)      parsers=(--tool-call-parser hermes --enable-auto-tool-choice --reasoning-parser qwen3) ;;
    # Mistral's own format (consolidated.safetensors, params.json, tekken.json), as Mistral recommends.
    mistralai/*)      parsers=(--tokenizer-mode mistral --config-format mistral --load-format mistral) ;;
  esac

  # --containall leaves only a tiny temporary home and /tmp, which kernel compilers overflow;
  # caches go to scratch (one folder per vLLM release), /tmp to node-local disk.
  local cache="$VLLM_CACHE_DIR/$VLLM_TAG"
  mkdir -p "$cache/home"
  VLLM_JOB_TMP="$(mktemp -d -p "${TMPDIR:-/tmp}" vllm-XXXX)"

  APPTAINERENV_HF_HOME=/hf \
  APPTAINERENV_HF_HUB_OFFLINE=1 \
  APPTAINERENV_VLLM_BATCH_INVARIANT="${BATCH_INVARIANT:-1}" \
  APPTAINERENV_VLLM_SYSTEM_START_DATE="$SYSTEM_START_DATE" \
  APPTAINERENV_XDG_CACHE_HOME=/cache \
  APPTAINERENV_TRITON_CACHE_DIR=/cache/triton \
  APPTAINERENV_VLLM_CACHE_ROOT=/cache/vllm \
  APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  apptainer exec --nv --containall --home "$cache/home" --bind "$VLLM_JOB_TMP:/tmp" \
      --bind "$HF_HOME:/hf" --bind "$cache:/cache" "$VLLM_SIF" \
    vllm serve "$model" --revision "$rev" --served-model-name "$served" \
      --host 127.0.0.1 --port "$port" --seed 0 "${parsers[@]}" \
      --no-enable-prefix-caching --gpu-memory-utilization 0.90 "$@" \
      > "$log" 2>&1 &
  VLLM_PID=$!
}

vllm_wait_ready() {
  local port="$1" max="${2:-3600}" waited=0
  while (( waited < max )); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && { echo "vLLM ready after ${waited}s"; return 0; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM exited during startup"; return 1; }
    sleep 10
    waited=$((waited + 10))
  done
  echo "vLLM not ready after ${max}s"
  return 1
}

vllm_stop() {
  kill "$VLLM_PID" 2>/dev/null
  wait "$VLLM_PID" 2>/dev/null
  rm -rf "$VLLM_JOB_TMP"
}
