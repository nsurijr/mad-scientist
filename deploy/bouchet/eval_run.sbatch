#!/bin/bash
#SBATCH --job-name=ms-eval
#SBATCH --account=pi_smd26
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Live evals of submissions: replay from golden up to each attack's first change,
# live through the gateway in eval mode after it, then the scorer against the sealed honest run.
#
#   SUBMISSIONS="<dir> ..." SEEDS="s0" GOLDEN=<namespace> CALIBRATION=calibration/provisional.json \
#     sbatch deploy/bouchet/eval_run.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/env.sh
source deploy/bouchet/common.sh
source deploy/bouchet/vllm_lib.sh
source deploy/bouchet/gateway_lib.sh
source deploy/bouchet/backend_lib.sh

SUBMISSIONS="${SUBMISSIONS:?set SUBMISSIONS to one or more submission folders}"
SEEDS="${SEEDS:-s0}"
GOLDEN="${GOLDEN:?set GOLDEN to the golden namespace}"
CALIBRATION="${CALIBRATION:-calibration/provisional.json}"
SECRET="${SECRET:-secrets/gateway.key}"
PARALLEL="${PARALLEL:-2}"
PORT=$((20000 + SLURM_JOB_ID % 20000))
GATEWAY_SOCKET="/tmp/gw-$SLURM_JOB_ID/gateway.sock"  # stage containers run with no network
BATCH="runs/eval-$SLURM_JOB_ID"
mkdir -p "$BATCH"
echo "batch=$BATCH golden=$GOLDEN seeds=[$SEEDS] submissions=[$SUBMISSIONS] calibration=$CALIBRATION"

backend_start "$BATCH" "$PORT" || exit 1  # BACKEND=ilaas: iLaaS, no GPU (backend_lib.sh)
gateway_start eval "$GOLDEN" "$((PORT + 1))" "$BATCH/gateway.jsonl" --secret "$SECRET" \
  "${UPSTREAM_ARGS[@]}" || { backend_stop; exit 1; }
base_url="http://127.0.0.1:$((PORT + 1))/v1"
for sub in $SUBMISSIONS; do
  for seed in $SEEDS; do
    while (( $(jobs -rp | wc -l) >= PARALLEL + BACKEND_JOBS + 1 )); do sleep 10; done
    (
      name="$(basename "$sub")"
      team="$(.venv/bin/python -c "import sys; from harness.manifest import load; print(load(sys.argv[1])[0].team)" "$sub")"
      start=$SECONDS
      .venv/bin/python -m harness.runner --seed "$seed" --run-dir "$BATCH/$name/$seed" --base-url "$base_url" \
        --tool-cache-url "$base_url/tools" --image "$SIF_DIR/pipeline.sif" --max-calls 30 --submission "$sub" \
        --eval-id "eval-$SLURM_JOB_ID-$name-$seed" --team "$team" --token-secret "$SECRET" --gateway-socket "$GATEWAY_SOCKET" \
        > "$BATCH/$name-$seed.log" 2>&1
      echo "$name $seed: runner exit=$? after $((SECONDS - start))s"
      .venv/bin/python -m harness.scorer score "$BATCH/$name/$seed" --submission "$sub" --sealed sealed \
        --calibration "$CALIBRATION" > "$BATCH/$name-$seed.score.json"
      .venv/bin/python -m harness.scorer score "$BATCH/$name/$seed" --submission "$sub" --sealed sealed \
        --calibration "$CALIBRATION" --team-view > "$BATCH/$name-$seed.team.json"
    ) &
  done
done
while (( $(jobs -rp | wc -l) > BACKEND_JOBS + 1 )); do sleep 10; done
gateway_stop
.venv/bin/python -m harness.gateway summary "$BATCH/gateway.jsonl"
backend_stop
for f in "$BATCH"/*.team.json; do echo "== $f"; cat "$f"; done
