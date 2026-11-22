#!/bin/bash
#SBATCH --job-name=uni_build
#SBATCH --cpus-per-task=32
#SBATCH --time=0-12:00:00
#SBATCH --output=JOBS/build.%j.out
#SBATCH --error=JOBS/build.%j.err
set -uo pipefail
umask 077
# Submit from the repository root, or set ROOT.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
set +u; source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-uniasr}"; set -u
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT"
cd "$ROOT"
if [ ! -f data/raw/nptel/.extracted ]; then
  mkdir -p data/raw/nptel/train
  # A partial split-tar part: gzip reports unexpected EOF at the cut, everything before it is valid.
  head -c 9000000000 data/raw/nptel/nptel-train.tar.gz.partaa | tar -xz -C data/raw/nptel/train 2>JOBS/nptel_tar.log
  touch data/raw/nptel/.extracted
fi
find data/raw/nptel/train -maxdepth 3 -type d | head; find data/raw/nptel/train -name '*.wav' | wc -l
set -e
python g2p/tests/test_phonemizer.py 2>&1 | tail -n 1
python data/sources.py 2>&1 | grep -E '^(en |total)'
python data/preprocess.py --workers 32 --tag default 2>&1 | grep -v -E 'Using the latest|Found the latest|Warn|warn|pkg_res|^[0-9]+$'
python data/build_manifests.py --workers 32 --tag default 2>&1 | grep -v -E 'Warn|warn|pkg_res|nltk'
python decoding/build_lm.py --workers 9 2>&1 | grep -v -E 'Warn|warn|pkg_res|nltk'
echo BUILD_DONE
