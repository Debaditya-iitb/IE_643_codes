#!/bin/bash
#SBATCH --job-name=uni_score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00
#SBATCH --output=JOBS/%x.%j.out
#SBATCH --error=JOBS/%x.%j.err
# sbatch -p <partition> --qos=<partition> -J score_<run> scripts/run_score_slurm.sh <run>
# Dumps test emissions for a finished run, then scores them, including beam decoding.
set -euo pipefail
umask 077
RUN=$1
# Submit from the repository root, or set ROOT.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set +u; source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-uniasr}"; set -u
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false
cd "$ROOT"
python -u experiments/dump_emissions.py "$RUN" --cap 2000
python -u experiments/evaluate_all.py score "$RUN" --workers 24 2>&1 | grep -v -E "Unigrams and labels|Found entries of length"
echo SCORE_DONE
