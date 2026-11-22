#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-04:00:00
#SBATCH --output=JOBS/%x.%j.out
#SBATCH --error=JOBS/%x.%j.err
# Transcribe your own audio with the final system.
#
#   sbatch -p l40 --qos=l40 -J transcribe scripts/transcribe_slurm.sh /path/a.wav /path/b.mp3 /path/folder/
#
# Each argument is an audio file or a folder (every .wav .flac .mp3 .ogg .m4a inside it).
# For each file you get, in results/transcripts/:
#   <name>.txt   one line per interval: start, end, language, transcript in that language's script
#   <name>.json  the same with phones and confidences
# Paths must be on the shared filesystem (not /tmp), because the job runs on a compute node.
set -euo pipefail
umask 077
# Submit from the repository root, or set ROOT.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set +u; source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-uniasr}"; set -u
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false
cd "$ROOT"
OUT=results/transcripts
mkdir -p "$OUT"
[ "$#" -gt 0 ] || { echo "usage: sbatch ... scripts/transcribe_slurm.sh <audio file or folder> ..."; exit 1; }
for arg in "$@"; do
  if [ -d "$arg" ]; then
    find "$arg" -maxdepth 1 -type f \( -iname '*.wav' -o -iname '*.flac' -o -iname '*.mp3' -o -iname '*.ogg' -o -iname '*.m4a' \) | sort
  else
    echo "$arg"
  fi
done | while IFS= read -r f; do
  name=$(basename "${f%.*}")
  python -u transcribe.py "$f" --out "$OUT/$name.json" | tee "$OUT/$name.txt"
done
echo "DONE, outputs in $PWD/$OUT"
