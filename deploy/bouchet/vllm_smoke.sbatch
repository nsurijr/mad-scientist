#!/bin/bash
#SBATCH --job-name=ms-vllm-smoke
#SBATCH --account=pi_smd26
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
# Determinism check: serve one model, record fixed probes alone and under load, restart the
# server, record again, and require bit-identical replies throughout.
#
# Settings come from the environment (sbatch passes it through):
#   SMOKE_MODEL / SMOKE_REV   model and revision (default: the pinned pipeline model)
#   BATCH_INVARIANT           1 (default) or 0
#   VLLM_TAG / VLLM_DIGEST    vLLM release (see common.sh)
# e.g. SMOKE_MODEL=Qwen/Qwen3-32B SMOKE_REV=<sha> sbatch deploy/bouchet/vllm_smoke.sbatch
#      sbatch -p gpu_b200 deploy/bouchet/vllm_smoke.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/common.sh

MODEL="${SMOKE_MODEL:-$PIPELINE_MODEL}"
REV="${SMOKE_REV:-$PIPELINE_MODEL_REV}"
SERVED="${MODEL##*/}"
case "$MODEL" in
  openai/gpt-oss-*) PARSERS=(--tool-call-parser openai --enable-auto-tool-choice --attention-backend TRITON_ATTN) ;;
  Qwen/Qwen3*)      PARSERS=(--tool-call-parser hermes --enable-auto-tool-choice --reasoning-parser qwen3) ;;
  *)                PARSERS=() ;;
esac
PORT=$((20000 + SLURM_JOB_ID % 20000))
OUT="logs/vllm-smoke-$SLURM_JOB_ID"
mkdir -p "$OUT"

export APPTAINERENV_HF_HOME=/hf
export APPTAINERENV_HF_HUB_OFFLINE=1
# vLLM 0.31.0 has no batch-invariant MXFP4 MoE kernel for gpt-oss on Hopper, so
# the mode is selectable: BATCH_INVARIANT=0 measures how often replies differ without it.
export APPTAINERENV_VLLM_BATCH_INVARIANT="${BATCH_INVARIANT:-1}"

# --containall gives the container only a small temporary home, which Triton and torch.compile
# caches overflow. Keep compile caches in scratch, one folder per vLLM release.
CACHE="$VLLM_CACHE_DIR/$VLLM_TAG"
mkdir -p "$CACHE/home"
# Some kernel JITs (vLLM 0.32's HUMMING MoE backend) also write to /tmp and $HOME, which
# --containall keeps tiny: give the container a node-local /tmp and a real home.
JOB_TMP="$(mktemp -d -p "${TMPDIR:-/tmp}" vllm-smoke-XXXX)"
trap 'rm -rf "$JOB_TMP"' EXIT
CONTAINER=(apptainer exec --nv --containall --home "$CACHE/home" --bind "$JOB_TMP:/tmp"
           --bind "$HF_HOME:/hf" --bind "$CACHE:/cache" "$VLLM_SIF")
export APPTAINERENV_XDG_CACHE_HOME=/cache
export APPTAINERENV_TRITON_CACHE_DIR=/cache/triton
export APPTAINERENV_VLLM_CACHE_ROOT=/cache/vllm
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/cache/inductor

echo "model=$MODEL@${REV:0:12} partition=$SLURM_JOB_PARTITION vllm=$VLLM_TAG batch_invariant=$APPTAINERENV_VLLM_BATCH_INVARIANT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
apptainer exec --bind "$CACHE:/cache" "$VLLM_SIF" python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"

serve() {
  "${CONTAINER[@]}" \
    vllm serve "$MODEL" --revision "$REV" --served-model-name "$SERVED" \
      --host 127.0.0.1 --port "$PORT" --seed 0 "${PARSERS[@]}" \
      --no-enable-prefix-caching --gpu-memory-utilization 0.90 \
      > "$OUT/server-$1.log" 2>&1 &
  SERVER_PID=$!
  # Batch-invariant cold starts compile for ~15 min (measured on Bouchet); allow up to an hour.
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "server $1 ready after $SECONDS s"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 10
  done
  echo "server $1 failed to start"
  grep -E "Error|error:" "$OUT/server-$1.log" | grep -v "^\s*File" | tail -5 | cut -c1-600
  return 1
}

stop() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }

status=0
serve a || exit 1
python3 deploy/bouchet/determinism_check.py run --port "$PORT" --model "$SERVED" --out "$OUT/a.json" || status=1
stop

serve b || exit 1
python3 deploy/bouchet/determinism_check.py run --port "$PORT" --model "$SERVED" --out "$OUT/b.json" || status=1
stop

python3 deploy/bouchet/determinism_check.py compare "$OUT/a.json" "$OUT/b.json" || status=1
echo "overall: $([ $status -eq 0 ] && echo PASS || echo FAIL)"
exit $status
