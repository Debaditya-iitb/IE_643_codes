#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-23:00:00
#SBATCH --output=JOBS/%x.%j.out
#SBATCH --error=JOBS/%x.%j.err
# sbatch -p <p> --qos=<p> -J <name> scripts/run_py_slurm.sh script.py args...
set -euo pipefail
umask 077
# Submit from the repository root, or set ROOT.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set +u; source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-uniasr}"; set -u
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false
cd "$ROOT"
python -u "$@"
