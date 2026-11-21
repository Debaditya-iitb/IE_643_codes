#!/usr/bin/env bash
# Starts the page on this computer and opens it in the browser. Stop it with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source venv/bin/activate
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# --device cpu works everywhere. If you have an NVIDIA GPU with CUDA PyTorch installed, remove it to use the GPU.
exec python app/server.py --device cpu --open "$@"
