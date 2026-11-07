#!/bin/bash
#SBATCH --job-name=ms-preview
#SBATCH --account=pi_smd26
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# What a team gets for free before spending tokens: the validator, then a preview eval
# on the public seed against its golden recording, then the preview report. No model server, no GPU.
#
#   SUBMISSIONS="<dir> ..." GOLDEN=<namespace> GOLDEN_RUN=runs/<batch>/s0 sbatch deploy/bouchet/preview.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/env.sh
source deploy/bouchet/common.sh
source deploy/bouchet/gateway_lib.sh

SUBMISSIONS="${SUBMISSIONS:?set SUBMISSIONS to one or more submission folders}"
GOLDEN="${GOLDEN:?set GOLDEN to the golden namespace}"
GOLDEN_RUN="${GOLDEN_RUN:?set GOLDEN_RUN to the golden run folder of the public seed}"
SECRET="${SECRET:-secrets/gateway.key}"
PORT=$((20000 + SLURM_JOB_ID % 20000))
GATEWAY_SOCKET="/tmp/gw-$SLURM_JOB_ID/gateway.sock"  # stage containers run with no network
BATCH="runs/preview-$SLURM_JOB_ID"
mkdir -p "$BATCH"
echo "batch=$BATCH golden=$GOLDEN golden_run=$GOLDEN_RUN submissions=[$SUBMISSIONS]"

gateway_start eval "$GOLDEN" "$PORT" "$BATCH/gateway.jsonl" --secret "$SECRET" || exit 1
base_url="http://127.0.0.1:$PORT/v1"
for sub in $SUBMISSIONS; do
  name="$(basename "$sub")"
  echo "== $name"
  if ! .venv/bin/python -m harness.validate check "$sub" --golden-run "$GOLDEN_RUN" --image "$SIF_DIR/pipeline.sif" \
      --json "$BATCH/$name.check.json"; then
    continue  # rejected: no preview
  fi
  team="$(.venv/bin/python -c "import sys; from harness.manifest import load; print(load(sys.argv[1])[0].team)" "$sub")"
  .venv/bin/python -m harness.runner --seed s0 --run-dir "$BATCH/$name/s0" --base-url "$base_url" \
    --tool-cache-url "$base_url/tools" --image "$SIF_DIR/pipeline.sif" --max-calls 30 --submission "$sub" \
    --eval-id "preview-$SLURM_JOB_ID-$name" --team "$team" --token-secret "$SECRET" --gateway-socket "$GATEWAY_SOCKET" --preview \
    > "$BATCH/$name.log" 2>&1
  .venv/bin/python -m harness.validate preview-report "$BATCH/$name/s0" --golden-run "$GOLDEN_RUN" \
    --json "$BATCH/$name.preview.json"
done
gateway_stop
