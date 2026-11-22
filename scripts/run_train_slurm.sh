#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-23:00:00
#SBATCH --output=JOBS/%x.%j.out
#SBATCH --error=JOBS/%x.%j.err
# sbatch -p <partition> --qos=<partition> -J <name> scripts/run_train_slurm.sh <stage> <config> [--set k=v ...]
set -euo pipefail
umask 077
STAGE=$1; shift
# Submit from the repository root, or set ROOT.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set +u; source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-uniasr}"; set -u
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false
cd "$ROOT"
echo "node $(hostname) job ${SLURM_JOB_ID:-none} $(date -Is) gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
python -u experiments/train_stage$STAGE.py "$@"
