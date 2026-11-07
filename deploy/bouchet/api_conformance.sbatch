#!/bin/bash
#SBATCH --job-name=ms-conformance
#SBATCH --account=pi_smd26
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# The API conformance check (deploy/bouchet/api_conformance.py) against iLaaS, or
# against our own vLLM as the baseline to compare it with. It sends recorded requests of one stage
# from a namespace (default stab3, tags matching */S1) and writes runs/conformance-<backend>-<job>.json.
#
#   BACKEND=ilaas sbatch -p day --gpus=0 --mem=8G deploy/bouchet/api_conformance.sbatch   # no GPU
#   sbatch deploy/bouchet/api_conformance.sbatch                                          # vLLM baseline
#   NAMESPACE=stab3 TAG="s7/S4" EXTRA="--long 100000 --burst 24" ...
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/env.sh
source deploy/bouchet/common.sh
source deploy/bouchet/vllm_lib.sh

NAMESPACE="${NAMESPACE:-stab3}"
TAG="${TAG:-*/S1}"
EXTRA="${EXTRA:-}"
PORT=$((20000 + SLURM_JOB_ID % 20000))
OUT="runs/conformance-$BACKEND-$SLURM_JOB_ID.json"
key=()
case "$BACKEND" in
  vllm)
    vllm_start "logs/conformance-vllm-$SLURM_JOB_ID.log" "$PIPELINE_MODEL" "$PIPELINE_MODEL_REV" gpt-oss-120b "$PORT"
    vllm_wait_ready "$PORT" || exit 1
    base_url="http://127.0.0.1:$PORT/v1"
    ;;
  ilaas)
    base_url="$ILAAS_BASE_URL"
    key=(--key-file "$ILAAS_KEY_FILE")
    ;;
  *)
    echo "BACKEND must be vllm or ilaas"
    exit 1
    ;;
esac
echo "conformance: backend=$BACKEND url=$base_url namespace=$NAMESPACE tag=$TAG"
# shellcheck disable=SC2086  # EXTRA is a list of options
PYTHONPATH=. .venv/bin/python deploy/bouchet/api_conformance.py --base-url "$base_url" "${key[@]}" \
  --from-cas "$REPO/cas" --namespace "$NAMESPACE" --tag "$TAG" --out "$OUT" $EXTRA
status=$?
[ "$BACKEND" = vllm ] && vllm_stop
echo "report: $OUT"
exit $status
