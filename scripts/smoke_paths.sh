#!/bin/bash
#SBATCH --job-name=uni_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=0-03:00:00
#SBATCH --output=JOBS/%x.%j.out
#SBATCH --error=JOBS/%x.%j.err
# Runs each code path the overfit gate did not reach: LoRA, splices with forced alignment,
# outlier exposure, resume, emission dump, scoring. Short runs, so PER values mean nothing here.
set -euo pipefail
umask 077
# Submit from the repository root, or set ROOT.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set +u; source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-uniasr}"; set -u
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false
cd "$ROOT"
COMMON="init_from=null steps=40 eval_every=20 log_every=10 dev_per_lang=5 workers=4 splice_rows=200 outlier_share=0.03 warmup=5"
LANGS='langs=[hi,mr,bn,gu,ta,te,en,hi_en,bn_en]'
DEV='dev_langs=[hi,en,hi_en]'
for bb in xlsr53:lora16 mms1b:lora8; do
  b=${bb%%:*}; s=${bb##*:}
  echo "=== train $b $s"
  python -u experiments/train_stage2.py experiments/configs/${b}_e1_${s}.yaml --set name=smoke_$b $COMMON "$LANGS" "$DEV"
  echo "=== resume $b"
  python -u experiments/train_stage2.py experiments/configs/${b}_e1_${s}.yaml --set name=smoke_$b $COMMON "$LANGS" "$DEV" steps=60
  echo "=== dump $b"
  python -u experiments/dump_emissions.py smoke_$b --cap 20
  echo "=== score $b"
  python -u experiments/evaluate_all.py score smoke_$b
done
echo SMOKE_DONE
