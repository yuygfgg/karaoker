#!/usr/bin/env bash
set -euo pipefail

# Bootstrap dependencies under this repo:
# - Create a conda env at `./.conda/env` and install MFA + Python deps
# - Clone/build whisper.cpp and python-audio-separator under `./third_party`

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_PREFIX="${ROOT}/.conda/env"
PKGS_DIR="${ROOT}/.conda/pkgs"
PIP_CACHE="${ROOT}/.pip-cache"
TP_DIR="${ROOT}/third_party"

mkdir -p "${PKGS_DIR}" "${PIP_CACHE}" "${TP_DIR}"

export CONDA_PKGS_DIRS="${PKGS_DIR}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -p "${ENV_PREFIX}" -y python=3.11 pip
fi

conda install -p "${ENV_PREFIX}" -y -c conda-forge montreal-forced-aligner

PIP_CMD=("${ENV_PREFIX}/bin/python" -m pip)
export PIP_CACHE_DIR="${PIP_CACHE}"

"${PIP_CMD[@]}" install -e "${ROOT}[dev,textgrid]"

if [[ ! -d "${TP_DIR}/python-audio-separator/.git" ]]; then
  git clone --depth 1 https://github.com/nomadkaraoke/python-audio-separator "${TP_DIR}/python-audio-separator"
fi
"${PIP_CMD[@]}" install -e "${TP_DIR}/python-audio-separator"

if [[ ! -d "${TP_DIR}/whisper.cpp/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp "${TP_DIR}/whisper.cpp"
fi
make -C "${TP_DIR}/whisper.cpp" -j"$(sysctl -n hw.ncpu || echo 4)"

echo "Done."
echo "Activate: conda activate ${ENV_PREFIX}"
echo "whisper.cpp binary: ${TP_DIR}/whisper.cpp/build/bin/main"
echo "MFA binary: ${ENV_PREFIX}/bin/mfa"
echo "audio-separator binary: ${ENV_PREFIX}/bin/audio-separator"
