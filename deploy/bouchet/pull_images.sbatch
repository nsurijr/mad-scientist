#!/bin/bash
#SBATCH --job-name=ms-pull-images
#SBATCH --account=pi_smd26
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Pull the pinned vLLM image and the sandbox test image as SIF files.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/common.sh
mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}"

[ -f "$SANDBOX_SIF" ] || apptainer pull "$SANDBOX_SIF" docker://python:3.12-slim
[ -f "$VLLM_SIF" ] || apptainer pull "$VLLM_SIF" "docker://vllm/vllm-openai@$VLLM_DIGEST"

ls -lh "$SIF_DIR"
mkdir -p "$VLLM_CACHE_DIR/$VLLM_TAG"
apptainer exec --bind "$VLLM_CACHE_DIR/$VLLM_TAG:/cache" "$VLLM_SIF" python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
