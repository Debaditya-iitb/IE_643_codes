#!/bin/bash
#SBATCH --job-name=uni_prep
#SBATCH --cpus-per-task=32
#SBATCH --time=0-23:00:00
#SBATCH --output=JOBS/prep.%j.out
#SBATCH --error=JOBS/prep.%j.err
set -euo pipefail
umask 077
# Submit from the repository root, or set ROOT.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set +u; source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-uniasr}"; set -u
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT"
cd "$ROOT"
python data/preprocess.py --workers 32 --tag default
