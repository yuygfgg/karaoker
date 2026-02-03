#!/usr/bin/env bash
set -euo pipefail

# Run `karaoker` from the repo-local conda env and keep tool state under this repo.
#
# Behavior tweaks:
# - Set `MFA_ROOT_DIR` to avoid MFA writing to `~/Documents/MFA`
# - Set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid a known OpenMP crash on macOS (audio-separator/torch)
# - Clear proxy env vars that can break downloads when a local proxy is configured

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ROOT}/.conda/env"

die() { echo "error: $*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/karaoker_run.sh --input <audio> --workdir <dir> [options]

Options:
  --lrc <path>                 Use an .lrc file as lyrics (recommended). Skips ASR.
  --separate                   Enable vocal separation (python-audio-separator).
  --no-separate                Disable separation (default).
  --download-models            Download required MFA models and separation model before running.

  --kana-output <katakana|hiragana>   Default: katakana
  --mfa-acoustic-model <name|path>    Default: japanese_mfa
  --mfa-dict <name|path>              Default: (auto via G2P)

  --audio-separator-model <filename>  Default: model_bs_roformer_ep_317_sdr_12.9755.ckpt
  --whisper-model <name>              Optional. Used only if you don't pass --lrc.
                                      Default: large-v3. Auto-downloads if missing.
USAGE
}

INPUT=""
LRC=""
WORKDIR=""
SEPARATE=0
DOWNLOAD_MODELS=0
KANA_OUTPUT="katakana"
MFA_ACOUSTIC_MODEL="japanese_mfa"
MFA_DICT=""  # empty => auto-g2p dict
AUDIO_SEP_MODEL="model_bs_roformer_ep_317_sdr_12.9755.ckpt"
WHISPER_MODEL="large-v3" # model name (e.g. large-v3, medium, base, small, tiny)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --input) INPUT="${2:-}"; shift 2 ;;
    --lrc) LRC="${2:-}"; shift 2 ;;
    --workdir) WORKDIR="${2:-}"; shift 2 ;;
    --separate) SEPARATE=1; shift ;;
    --no-separate) SEPARATE=0; shift ;;
    --download-models) DOWNLOAD_MODELS=1; shift ;;
    --kana-output) KANA_OUTPUT="${2:-}"; shift 2 ;;
    --mfa-acoustic-model) MFA_ACOUSTIC_MODEL="${2:-}"; shift 2 ;;
    --mfa-dict) MFA_DICT="${2:-}"; shift 2 ;;
    --audio-separator-model) AUDIO_SEP_MODEL="${2:-}"; shift 2 ;;
    --whisper-model) WHISPER_MODEL="${2:-}"; shift 2 ;;
    *) die "unknown arg: $1 (use --help)" ;;
  esac
done

[[ -n "${INPUT}" ]] || die "--input is required"
[[ -n "${WORKDIR}" ]] || die "--workdir is required"

[[ -x "${ENV_PREFIX}/bin/python" ]] || die "conda env not found at ${ENV_PREFIX} (run ./scripts/bootstrap_local.sh)"
[[ -x "${ENV_PREFIX}/bin/karaoker" ]] || die "karaoker not installed in ${ENV_PREFIX}"

# Normalize whisper model argument: allow "ggml-large-v3.bin" or "large-v3" etc.
WHISPER_MODEL="${WHISPER_MODEL#ggml-}"
WHISPER_MODEL="${WHISPER_MODEL%.bin}"

FFMPEG="${ENV_PREFIX}/bin/ffmpeg"
MFA="${ENV_PREFIX}/bin/mfa"
AUDIO_SEP="${ENV_PREFIX}/bin/audio-separator"
WHISPER_CLI="${ROOT}/third_party/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODELS_DIR="${ROOT}/third_party/whisper.cpp/models"
WHISPER_DOWNLOAD="${WHISPER_MODELS_DIR}/download-ggml-model.sh"

[[ -x "${FFMPEG}" ]] || die "ffmpeg not found in env at ${FFMPEG}"
[[ -x "${MFA}" ]] || die "mfa not found in env at ${MFA}"

# Avoid MFA defaulting to ~/Documents/MFA (often blocked).
export MFA_ROOT_DIR="${ROOT}/.mfa"
mkdir -p "${MFA_ROOT_DIR}"

# Work around duplicate OpenMP runtime crash (macOS) for audio-separator / torch.
export KMP_DUPLICATE_LIB_OK=TRUE

# Avoid inheriting proxy envs that point to a local proxy not reachable from the sandbox.
export NO_PROXY='*'
export HTTP_PROXY=''
export HTTPS_PROXY=''
export http_proxy=''
export https_proxy=''

ensure_mfa_models() {
  # These commands require network if the models aren't present yet.
  "${MFA}" model list acoustic | grep -q "japanese_mfa" || "${MFA}" model download acoustic japanese_mfa
  "${MFA}" model list dictionary | grep -q "japanese_mfa" || "${MFA}" model download dictionary japanese_mfa
  "${MFA}" model list g2p | grep -q "japanese_katakana_mfa" || "${MFA}" model download g2p japanese_katakana_mfa
}

whisper_model_path() {
  # whisper.cpp uses filenames like "ggml-large-v3.bin"
  echo "${WHISPER_MODELS_DIR}/ggml-${WHISPER_MODEL}.bin"
}

ensure_whisper_model() {
  [[ -x "${WHISPER_CLI}" ]] || die "whisper-cli not found at ${WHISPER_CLI} (run ./scripts/bootstrap_local.sh)"
  local mp
  mp="$(whisper_model_path)"
  if [[ -f "${mp}" ]]; then
    return 0
  fi
  [[ -x "${WHISPER_DOWNLOAD}" ]] || die "whisper.cpp download script missing at ${WHISPER_DOWNLOAD}"
  echo "Whisper model not found: ${mp}"
  echo "Downloading whisper.cpp model: ${WHISPER_MODEL}"

  # First try upstream whisper.cpp helper (defaults to huggingface.co).
  # On some networks, huggingface.co can be intercepted and present a wrong TLS cert.
  set +e
  NO_PROXY='*' HTTP_PROXY='' HTTPS_PROXY='' http_proxy='' https_proxy='' \
    bash "${WHISPER_DOWNLOAD}" "${WHISPER_MODEL}" "${WHISPER_MODELS_DIR}"
  rc=$?
  set -e
  if [[ $rc -eq 0 && -f "${mp}" ]]; then
    return 0
  fi

  # Fallback: download from hf-mirror.com (Hugging Face mirror).
  # URL pattern mirrors the upstream: /<repo>/resolve/main/<file>
  local url="https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-${WHISPER_MODEL}.bin"
  echo "Upstream download failed; trying mirror:"
  echo "  ${url}"

  local tmp="${mp}.part"
  rm -f "${tmp}"
  NO_PROXY='*' HTTP_PROXY='' HTTPS_PROXY='' http_proxy='' https_proxy='' \
    curl -L --fail --retry 5 --retry-delay 1 --output "${tmp}" "${url}"
  mv "${tmp}" "${mp}"
  [[ -s "${mp}" ]] || die "download finished but model file is empty: ${mp}"
}

ensure_audio_separator_model() {
  local model_dir="${ROOT}/models/audio_separator"
  mkdir -p "${model_dir}"
  # audio-separator will also fetch a small JSON index; allow network for first run.
  "${AUDIO_SEP}" --model_file_dir "${model_dir}" --download_model_only -m "${AUDIO_SEP_MODEL}"
}

if [[ "${DOWNLOAD_MODELS}" -eq 1 ]]; then
  ensure_mfa_models
  if [[ "${SEPARATE}" -eq 1 ]]; then
    [[ -x "${AUDIO_SEP}" ]] || die "audio-separator not found in env at ${AUDIO_SEP}"
    ensure_audio_separator_model
  fi
  # Whisper model is downloaded on-demand when ASR is used (no --lrc).
fi

mkdir -p "${WORKDIR}"

cmd=(
  "${ENV_PREFIX}/bin/karaoker" run
  --input "${INPUT}"
  --workdir "${WORKDIR}"
  --ffmpeg "${FFMPEG}"
  --mfa "${MFA}"
  --kana-output "${KANA_OUTPUT}"
  --mfa-acoustic-model "${MFA_ACOUSTIC_MODEL}"
)

if [[ -n "${LRC}" ]]; then
  cmd+=( --lyrics-lrc "${LRC}" )
else
  # ASR path (offline via whisper.cpp). We accept a model *name* and download if missing.
  ensure_whisper_model
  cmd+=( --whisper-cpp "${WHISPER_CLI}" --whisper-model "$(whisper_model_path)" )
fi

if [[ -n "${MFA_DICT}" ]]; then
  cmd+=( --mfa-dict "${MFA_DICT}" )
fi

if [[ "${SEPARATE}" -eq 1 ]]; then
  [[ -x "${AUDIO_SEP}" ]] || die "audio-separator not found in env at ${AUDIO_SEP}"
  cmd+=( --audio-separator "${AUDIO_SEP}" )
fi

echo "+ ${cmd[*]}"
exec "${cmd[@]}"
