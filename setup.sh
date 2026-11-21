#!/usr/bin/env bash
# One-time setup for macOS and Linux (and Windows through WSL2). Needs Python 3.10 to 3.12 and an internet connection.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

"$PY" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "need Python 3.10 to 3.12, found " + sys.version'
"$PY" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip

if [ "$(uname)" = "Darwin" ]; then
  pip install torch
else
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt

# kenlm is compiled during install: it needs a C++ compiler (macOS: xcode-select --install; Ubuntu: sudo apt install build-essential python3-dev)
pip install kenlm || pip install https://github.com/kpu/kenlm/archive/master.zip

for f in results/checkpoints/xlsr53_s3e/best.pt decoding/lm/mix.bin decoding/lm/mix.unigrams.txt; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
done
echo "Setup finished. Start the page with:  ./run.sh"
