#!/bin/bash
#SBATCH --job-name=ms-download-models
#SBATCH --account=pi_smd26
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Download the pinned pipeline and monitor models into the shared Hugging Face cache.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/env.sh
source deploy/bouchet/common.sh

# The Xet downloader buffers chunks in memory; at the defaults it went past 16 GB on the
# gpt-oss shards (measured on Bouchet). Fewer parallel files and range requests keep it bounded.
export HF_XET_NUM_CONCURRENT_RANGE_GETS=8
DL=(.venv/bin/hf download --max-workers 4)

# gpt-oss ships duplicate weight copies under original/ and metal/; vLLM needs neither.
"${DL[@]}" "$PIPELINE_MODEL" --revision "$PIPELINE_MODEL_REV" --exclude "original/*" --exclude "metal/*"
"${DL[@]}" "$MONITOR_MODEL" --revision "$MONITOR_MODEL_REV"

du -sh "$HF_HOME"/hub/models--openai--gpt-oss-120b "$HF_HOME"/hub/models--Qwen--Qwen3-32B
