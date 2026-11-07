#!/bin/bash
#SBATCH --job-name=ms-staged-replay
#SBATCH --account=pi_smd26
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Strict replay of a recorded staged batch (acceptance test 1). The gateway serves only
# cas/$NS, with no model server and no GPU, and every LLM call and code execution must be a
# recorded hit. deploy/bouchet/replay_check.py then compares the replay with the recording.
#
#   NS=<namespace> RECORDED=runs/staged-<job> sbatch deploy/bouchet/staged_replay.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/env.sh
source deploy/bouchet/common.sh
source deploy/bouchet/gateway_lib.sh

NS="${NS:?set NS to the recorded namespace}"
RECORDED="${RECORDED:?set RECORDED to the recorded batch folder}"
PORT=$((20000 + SLURM_JOB_ID % 20000))
BATCH="runs/replay-$SLURM_JOB_ID"
mkdir -p "$BATCH"
echo "batch=$BATCH namespace=$NS recorded=$RECORDED date=$SYSTEM_START_DATE"

gateway_start replay "$NS" "$PORT" "$BATCH/gateway.jsonl" || exit 1
base_url="http://127.0.0.1:$PORT/v1"
for run in "$RECORDED"/*/run.json; do
  seed="$(basename "$(dirname "$run")")"
  # Same budget and threads as the recording: they shape the trajectory.
  read -r max_calls threads < <(python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['settings']; \
print(s['max_calls'], s['exec_threads'])" "$run")
  start=$SECONDS
  .venv/bin/python -m harness.runner --seed "$seed" --run-dir "$BATCH/$seed" --base-url "$base_url" \
    --tool-cache-url "$base_url/tools" --image "$SIF_DIR/pipeline.sif" --max-calls "$max_calls" \
    --exec-threads "$threads" > "$BATCH/$seed.log" 2>&1
  echo "$seed: runner exit=$? after $((SECONDS - start))s"
done
gateway_stop

.venv/bin/python deploy/bouchet/replay_check.py "$RECORDED" "$BATCH" | tee "$BATCH/replay_check.txt"
