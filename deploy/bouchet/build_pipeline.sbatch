#!/bin/bash
#SBATCH --job-name=ms-build-pipeline
#SBATCH --account=pi_smd26
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Build containers/pipeline.sif from containers/pipeline.def (patch P7) and record its hash.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/bouchet/common.sh
mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${TMPDIR:-/tmp}"

out="$SIF_DIR/pipeline.sif"
# Unprivileged builds need user namespaces; fall back to --fakeroot if the plain build is refused.
apptainer build --force "$out" containers/pipeline.def \
  || apptainer build --force --fakeroot "$out" containers/pipeline.def

sha256sum "$out" | tee "$out.sha256"
# --containall, as the pipeline runs it: otherwise ~/.local packages leak in.
apptainer exec --containall "$out" python -c \
  "import sys, numpy, sklearn, openai, matplotlib; print(sys.version.split()[0], 'numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'openai', openai.__version__, 'matplotlib', matplotlib.__version__)"
