#!/bin/bash
#SBATCH --job-name=ms-acc2
#SBATCH --account=pi_smd26
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Acceptance test 2, clean divergence: an attack at stage k replays everything upstream of
# its first change from golden, with no strict-mode miss, and the first call that needs inference
# comes right after that change. Preview evals, so no model server and no GPU: the gateway (eval
# mode) stops each eval at its first live call.
#
#   GOLDEN=<namespace with an honest s0> [TEAM=<fresh team name>] sbatch deploy/bouchet/acceptance2.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/env.sh
source deploy/bouchet/common.sh
source deploy/bouchet/gateway_lib.sh

GOLDEN="${GOLDEN:?set GOLDEN to the namespace holding the honest s0 run}"
SECRET="${SECRET:-secrets/gateway.key}"
ATTACKS="${ATTACKS:-s4_context s4_tool_result}"
TEAM="${TEAM:-acc2}"  # its namespace must not hold these attacks already, or nothing needs inference
PORT=$((20000 + SLURM_JOB_ID % 20000))
GATEWAY_SOCKET="/tmp/gw-$SLURM_JOB_ID/gateway.sock"  # stage containers run with no network
BATCH="runs/acc2-$SLURM_JOB_ID"
mkdir -p "$BATCH"
echo "batch=$BATCH golden=$GOLDEN attacks=[$ATTACKS] date=$SYSTEM_START_DATE"

gateway_start eval "$GOLDEN" "$PORT" "$BATCH/gateway.jsonl" --secret "$SECRET" || exit 1
base_url="http://127.0.0.1:$PORT/v1"
for attack in $ATTACKS; do
  # The run folder is named after the seed, so eval tags match golden's ("s0/S4").
  start=$SECONDS
  .venv/bin/python -m harness.runner --seed s0 --run-dir "$BATCH/$attack/s0" --base-url "$base_url" \
    --tool-cache-url "$base_url/tools" --image "$SIF_DIR/pipeline.sif" --max-calls 30 \
    --submission "tests/acceptance/attacks/$attack" --eval-id "acc2-$SLURM_JOB_ID-$attack" --team "$TEAM" \
    --token-secret "$SECRET" --gateway-socket "$GATEWAY_SOCKET" --preview > "$BATCH/$attack.log" 2>&1
  echo "$attack: runner exit=$? after $((SECONDS - start))s"
done
gateway_stop

.venv/bin/python deploy/bouchet/acceptance2_check.py "$BATCH" $ATTACKS | tee "$BATCH/acceptance2.txt"
