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
SOFA_DIR="${TP_DIR}/SOFA"
SOFA_MODELS_DIR="${ROOT}/models/sofa"
SOFA_MODEL_URL="https://github.com/colstone/SOFA_Models/releases/download/JPN-V0.0.2b/SOFA_model_JPN_Ver0.0.2_Beta.zip"
SOFA_MODEL_ZIP="${SOFA_MODELS_DIR}/SOFA_model_JPN_Ver0.0.2_Beta.zip"

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

if [[ ! -d "${SOFA_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/qiuqiao/SOFA "${SOFA_DIR}"
fi

if [[ -f "${SOFA_DIR}/requirements.txt" ]]; then
  "${PIP_CMD[@]}" install -r "${SOFA_DIR}/requirements.txt"
else
  echo "warning: SOFA requirements.txt not found at ${SOFA_DIR}/requirements.txt"
fi

mkdir -p "${SOFA_MODELS_DIR}"
if [[ ! -f "${SOFA_MODEL_ZIP}" ]]; then
  tmp="${SOFA_MODEL_ZIP}.part"
  rm -f "${tmp}"
  curl -L --fail --retry 5 --retry-delay 1 --output "${tmp}" "${SOFA_MODEL_URL}"
  mv "${tmp}" "${SOFA_MODEL_ZIP}"
fi

# Extract the model if we don't see any likely checkpoints/dictionaries yet.
have_ckpt="$(find "${SOFA_MODELS_DIR}" -type f \( -iname "*.ckpt" -o -iname "*.pt" -o -iname "*.pth" \) -print -quit 2>/dev/null || true)"
have_dict="$(find "${SOFA_MODELS_DIR}" -type f \( -iname "*dict*.txt" -o -iname "*dictionary*.txt" -o -iname "*.dict" \) -print -quit 2>/dev/null || true)"
if [[ -z "${have_ckpt}" || -z "${have_dict}" ]]; then
  unzip -q -o "${SOFA_MODEL_ZIP}" -d "${SOFA_MODELS_DIR}"
fi

echo "Done."
echo "Activate: conda activate ${ENV_PREFIX}"
echo "whisper.cpp binary: ${TP_DIR}/whisper.cpp/build/bin/main"
echo "MFA binary: ${ENV_PREFIX}/bin/mfa"
echo "audio-separator binary: ${ENV_PREFIX}/bin/audio-separator"
echo "SOFA repo: ${SOFA_DIR}"
echo "SOFA models: ${SOFA_MODELS_DIR}"
