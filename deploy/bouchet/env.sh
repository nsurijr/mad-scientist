# Source before running anything from .venv on Bouchet (login node or Slurm job):
#   source deploy/bouchet/env.sh
# The venv's interpreter links against the module's libpython, so the module must be loaded.
module load Python/3.12.3-GCCcore-13.3.0 uv/0.13.4
