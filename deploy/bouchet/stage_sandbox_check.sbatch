#!/bin/bash
#SBATCH --job-name=ms-stage-sandbox
#SBATCH --account=pi_smd26
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# A stage container with no network, as harness/runner.py builds it for evals (--gateway-socket):
# it must reach the gateway through its Unix socket and nothing else (deploy/bouchet/stage_sandbox_check.py).
#
#   sbatch deploy/bouchet/stage_sandbox_check.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/env.sh
source deploy/bouchet/common.sh
PYTHONPATH=. .venv/bin/python deploy/bouchet/stage_sandbox_check.py "$SIF_DIR/pipeline.sif"
