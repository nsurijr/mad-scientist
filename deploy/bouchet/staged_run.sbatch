#!/bin/bash
#SBATCH --job-name=ms-staged
#SBATCH --account=pi_smd26
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Staged pipeline (S1-S5, harness/runner.py) on one or more seeds against one vLLM server, then
# an organizer-side report against the sealed truth (deploy/bouchet/staged_report.py).
#
#   SEEDS="s0" sbatch deploy/bouchet/staged_run.sbatch
#   SEEDS="s0 s6 s7" PARALLEL=3 MAX_CALLS=30 sbatch deploy/bouchet/staged_run.sbatch
#   RUNS="s0 s0:S3:null1" ...   a run is a seed, or seed:STAGE:KIND for a null variant (harness/runner.py
#                                NULL_VARIANTS: whitespace changes to that stage's brief); folder s0-S3-null1
#
# RECORD_NS=<namespace> routes every LLM call and code execution through the gateway
# (harness/gateway.py), which records them in cas/<namespace>. Replay the batch with
# staged_replay.sbatch.
#
# BACKEND=ilaas sends model calls to iLaaS instead of a local vLLM (backend_lib.sh): no GPU
# (submit with -p day), and RECORD_NS is required, since only the gateway holds the key.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/env.sh
source deploy/bouchet/common.sh
source deploy/bouchet/vllm_lib.sh
source deploy/bouchet/gateway_lib.sh
source deploy/bouchet/backend_lib.sh

SEEDS="${SEEDS:-s0}"
RUNS="${RUNS:-$SEEDS}"
PARALLEL="${PARALLEL:-1}"
MAX_CALLS="${MAX_CALLS:-30}"
RECORD_NS="${RECORD_NS:-}"
PORT=$((20000 + SLURM_JOB_ID % 20000))
BATCH="runs/staged-$SLURM_JOB_ID"
mkdir -p "$BATCH"
echo "batch=$BATCH runs=[$RUNS] parallel=$PARALLEL max_calls=$MAX_CALLS vllm=$VLLM_TAG image=$(cut -c1-12 "$SIF_DIR/pipeline.sif.sha256") date=$SYSTEM_START_DATE record_ns=${RECORD_NS:-none}"

if [ "$BACKEND" = ilaas ] && [ -z "$RECORD_NS" ]; then
  echo "BACKEND=ilaas needs RECORD_NS: iLaaS is reached only through the gateway"
  exit 1
fi
backend_start "$BATCH" "$PORT" || exit 1
base_url="http://127.0.0.1:$PORT/v1"
cache_args=()
background=$BACKEND_JOBS
if [ -n "$RECORD_NS" ]; then
  gateway_start record "$RECORD_NS" "$((PORT + 1))" "$BATCH/gateway.jsonl" "${UPSTREAM_ARGS[@]}" \
    || { backend_stop; exit 1; }
  base_url="http://127.0.0.1:$((PORT + 1))/v1"
  cache_args=(--tool-cache-url "$base_url/tools")
  background=$((BACKEND_JOBS + 1))
fi

for spec in $RUNS; do
  IFS=: read -r seed null_stage null_kind <<< "$spec"
  name="$seed"
  null_args=()
  if [ -n "$null_kind" ]; then
    name="$seed-$null_stage-$null_kind"
    null_args=(--null "$null_stage:$null_kind")
  fi
  while (( $(jobs -rp | wc -l) >= PARALLEL + background )); do sleep 10; done
  (
    start=$SECONDS
    .venv/bin/python -m harness.runner --seed "$seed" --run-dir "$BATCH/$name" "${null_args[@]}" \
      --base-url "$base_url" --image "$SIF_DIR/pipeline.sif" --max-calls "$MAX_CALLS" "${cache_args[@]}" \
      > "$BATCH/$name.log" 2>&1
    echo "$name: runner exit=$? after $((SECONDS - start))s"
  ) &
done
while (( $(jobs -rp | wc -l) > background )); do sleep 10; done
if [ -n "$RECORD_NS" ]; then
  gateway_stop
  .venv/bin/python -m harness.gateway summary "$BATCH/gateway.jsonl"
fi
backend_stop

.venv/bin/python deploy/bouchet/staged_report.py "$BATCH" sealed | tee "$BATCH/report.txt"
