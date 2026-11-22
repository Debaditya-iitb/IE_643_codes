#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-04:00:00
#SBATCH --output=JOBS/%x.%j.out
#SBATCH --error=JOBS/%x.%j.err
# Starts the web page on a GPU node. Submit from the repository root:
#   sbatch -p <partition> --qos=<partition> -J serve scripts/serve_slurm.sh
# Then read the link:   grep OPEN JOBS/serve.<jobid>.out
# From your own computer:   ssh -L 8765:<node>:8765 <you>@<login-host>   and open http://localhost:8765/?t=<token>
set -euo pipefail
umask 077
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set +u; source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-uniasr}"; set -u
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false
cd "$ROOT"
python -u app/server.py --host 0.0.0.0 --port "${PORT:-8765}"
