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
  --asr-backend <whispercpp|gemini>    Default: whispercpp. Ignored when --lrc is set.
  --kana-backend <mecab|gemini>        Default: mecab.
  --gemini-model <name>                Default: gemini-3-flash-preview.
                                       Gemini backends require `GEMINI_API_KEY` and `pip install -e ".[gemini]"`.

  --aligner-backend <mfa|sofa>        Default: mfa
  --sofa-root <path>                  Default: ./third_party/SOFA
  --sofa-model-dir <path>             Default: ./models/sofa
  --sofa-python <path>                Default: repo conda env python
  --sofa-ckpt <path>                  Optional. Overrides auto-detected SOFA checkpoint.
  --sofa-dict <path>                  Optional. Overrides auto-detected SOFA dictionary.

  --lrc <path>                 Use an .lrc file as lyrics (recommended). Skips ASR.
  --separate                   Enable vocal separation (python-audio-separator) (default).
  --no-separate                Disable separation.
  --download-models            Download required MFA/SOFA/separation/VAD models before running.

  --kana-output <katakana|hiragana>   Default: katakana
  --mfa-acoustic-model <name|path>    Default: japanese_mfa
  --mfa-dict <name|path>              Default: (auto via G2P)
  --mfa-f0 <none|constant|flatten>    Optional. Flatten pitch (F0) with WORLD/pyworld before MFA.
                                      Default: none.
  --mfa-f0-constant-hz <hz>           When --mfa-f0 constant. Default: 150
  --mfa-f0-flatten-factor <0..1>      When --mfa-f0 flatten. Default: 0
  --no-mfa-f0-preserve-unvoiced       When set, also forces unvoiced frames (F0=0) to the flattened value.

  --audio-separator-model <filename>  Default: model_bs_roformer_ep_317_sdr_12.9755.ckpt
  --dereverb-model <filename>         Default: dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt
  --no-dereverb                       Disable de-reverb (default enabled).
  --no-silero-vad                     Disable Silero VAD gating (default enabled).
  --whisper-model <name>              Optional. Used only if you don't pass --lrc.
                                      Default: large-v2. Auto-downloads if missing.
USAGE
}

INPUT=""
LRC=""
WORKDIR=""
ASR_BACKEND="whispercpp"
KANA_BACKEND="mecab"
GEMINI_MODEL="gemini-3-flash-preview"
SEPARATE=1
DOWNLOAD_MODELS=0
KANA_OUTPUT="katakana"
MFA_ACOUSTIC_MODEL="japanese_mfa"
MFA_DICT=""  # empty => auto-g2p dict
MFA_F0="none"
MFA_F0_CONSTANT_HZ="150"
MFA_F0_FLATTEN_FACTOR="0.0"
MFA_F0_PRESERVE_UNVOICED=1
AUDIO_SEP_MODEL="model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEREVERB_MODEL="dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"
DEREVERB=1
SILERO_VAD=1
WHISPER_MODEL="large-v2" # model name (e.g. large-v2, medium, base, small, tiny)
ALIGNER_BACKEND="mfa"
SOFA_ROOT="${ROOT}/third_party/SOFA"
SOFA_MODEL_DIR="${ROOT}/models/sofa"
SOFA_MODEL_URL="https://github.com/colstone/SOFA_Models/releases/download/JPN-V0.0.2b/SOFA_model_JPN_Ver0.0.2_Beta.zip"
SOFA_MODEL_ZIP="${SOFA_MODEL_DIR}/SOFA_model_JPN_Ver0.0.2_Beta.zip"
SOFA_PYTHON="${ENV_PREFIX}/bin/python"
SOFA_CKPT=""
SOFA_DICT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --input) INPUT="${2:-}"; shift 2 ;;
    --lrc) LRC="${2:-}"; shift 2 ;;
    --workdir) WORKDIR="${2:-}"; shift 2 ;;
    --asr-backend) ASR_BACKEND="${2:-}"; shift 2 ;;
    --kana-backend) KANA_BACKEND="${2:-}"; shift 2 ;;
    --gemini-model) GEMINI_MODEL="${2:-}"; shift 2 ;;
    --separate) SEPARATE=1; shift ;;
    --no-separate) SEPARATE=0; shift ;;
    --download-models) DOWNLOAD_MODELS=1; shift ;;
    --aligner-backend) ALIGNER_BACKEND="${2:-}"; shift 2 ;;
    --sofa-root) SOFA_ROOT="${2:-}"; shift 2 ;;
    --sofa-model-dir)
      SOFA_MODEL_DIR="${2:-}"
      SOFA_MODEL_ZIP="${SOFA_MODEL_DIR}/SOFA_model_JPN_Ver0.0.2_Beta.zip"
      shift 2
      ;;
    --sofa-python) SOFA_PYTHON="${2:-}"; shift 2 ;;
    --sofa-ckpt) SOFA_CKPT="${2:-}"; shift 2 ;;
    --sofa-dict) SOFA_DICT="${2:-}"; shift 2 ;;
    --kana-output) KANA_OUTPUT="${2:-}"; shift 2 ;;
    --mfa-acoustic-model) MFA_ACOUSTIC_MODEL="${2:-}"; shift 2 ;;
    --mfa-dict) MFA_DICT="${2:-}"; shift 2 ;;
    --mfa-f0) MFA_F0="${2:-}"; shift 2 ;;
    --mfa-f0-constant-hz) MFA_F0_CONSTANT_HZ="${2:-}"; shift 2 ;;
    --mfa-f0-flatten-factor) MFA_F0_FLATTEN_FACTOR="${2:-}"; shift 2 ;;
    --mfa-f0-preserve-unvoiced) MFA_F0_PRESERVE_UNVOICED=1; shift ;;
    --no-mfa-f0-preserve-unvoiced) MFA_F0_PRESERVE_UNVOICED=0; shift ;;
    --audio-separator-model) AUDIO_SEP_MODEL="${2:-}"; shift 2 ;;
    --dereverb-model) DEREVERB_MODEL="${2:-}"; shift 2 ;;
    --no-dereverb) DEREVERB=0; shift ;;
    --no-silero-vad) SILERO_VAD=0; shift ;;
    --whisper-model) WHISPER_MODEL="${2:-}"; shift 2 ;;
    *) die "unknown arg: $1 (use --help)" ;;
  esac
done

[[ -n "${INPUT}" ]] || die "--input is required"
[[ -n "${WORKDIR}" ]] || die "--workdir is required"

[[ -x "${ENV_PREFIX}/bin/python" ]] || die "conda env not found at ${ENV_PREFIX} (run ./scripts/bootstrap_local.sh)"
[[ -x "${ENV_PREFIX}/bin/karaoker" ]] || die "karaoker not installed in ${ENV_PREFIX}"

# Normalize whisper model argument: allow "ggml-large-v2.bin" or "large-v2" etc.
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

sofa_guess_ckpt() {
  find "${SOFA_MODEL_DIR}" -type f \( -iname "*.ckpt" -o -iname "*.pt" -o -iname "*.pth" \) \
    2>/dev/null | sort | head -n 1 || true
}

sofa_guess_dict() {
  find "${SOFA_MODEL_DIR}" -type f \( -iname "*dict*.txt" -o -iname "*dictionary*.txt" -o -iname "*.dict" \) \
    2>/dev/null | sort | head -n 1 || true
}

ensure_sofa_repo() {
  [[ -f "${SOFA_ROOT}/infer.py" ]] || die "SOFA repo missing at ${SOFA_ROOT} (run ./scripts/bootstrap_local.sh)"
}

ensure_sofa_model() {
  mkdir -p "${SOFA_MODEL_DIR}"

  # If we already have a checkpoint + dictionary, do nothing.
  local ckpt dict
  ckpt="$(sofa_guess_ckpt)"
  dict="$(sofa_guess_dict)"
  if [[ -n "${ckpt}" && -n "${dict}" ]]; then
    return 0
  fi

  # Download + extract the official Japanese beta model.
  if [[ ! -f "${SOFA_MODEL_ZIP}" ]]; then
    echo "Downloading SOFA model zip:"
    echo "  ${SOFA_MODEL_URL}"
    local tmp="${SOFA_MODEL_ZIP}.part"
    rm -f "${tmp}"
    NO_PROXY='*' HTTP_PROXY='' HTTPS_PROXY='' http_proxy='' https_proxy='' \
      curl -L --fail --retry 5 --retry-delay 1 --output "${tmp}" "${SOFA_MODEL_URL}"
    mv "${tmp}" "${SOFA_MODEL_ZIP}"
  fi

  echo "Extracting SOFA model zip -> ${SOFA_MODEL_DIR}"
  unzip -q -o "${SOFA_MODEL_ZIP}" -d "${SOFA_MODEL_DIR}"

  ckpt="$(sofa_guess_ckpt)"
  dict="$(sofa_guess_dict)"
  [[ -n "${ckpt}" ]] || die "SOFA model extracted but no checkpoint found under ${SOFA_MODEL_DIR}"
  [[ -n "${dict}" ]] || die "SOFA model extracted but no dictionary found under ${SOFA_MODEL_DIR}"
}

ensure_mfa_models() {
  # These commands require network if the models aren't present yet.
  "${MFA}" model list acoustic | grep -q "japanese_mfa" || "${MFA}" model download acoustic japanese_mfa
  "${MFA}" model list dictionary | grep -q "japanese_mfa" || "${MFA}" model download dictionary japanese_mfa
  "${MFA}" model list g2p | grep -q "japanese_katakana_mfa" || "${MFA}" model download g2p japanese_katakana_mfa
}

whisper_model_path() {
  # whisper.cpp uses filenames like "ggml-large-v2.bin"
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
  if [[ "${DEREVERB}" -eq 1 ]]; then
    "${AUDIO_SEP}" --model_file_dir "${model_dir}" --download_model_only -m "${DEREVERB_MODEL}"
  fi
}

ensure_silero_vad_model() {
  local model_dir="${ROOT}/models/silero_vad"
  mkdir -p "${model_dir}"

  # Trigger torch.hub to download the Silero VAD repo + weights into our repo-local cache.
  SILERO_DIR="${model_dir}" KMP_DUPLICATE_LIB_OK=TRUE \
    "${ENV_PREFIX}/bin/python" - <<'PY'
import os
from pathlib import Path

import torch

model_dir = Path(os.environ["SILERO_DIR"])
model_dir.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(model_dir))

# Download/extract without importing hubconf.py (which can require torchaudio).
try:
    repo_dir = torch.hub._get_cache_or_reload(  # pyright: ignore[reportPrivateUsage]
        "snakers4/silero-vad",
        force_reload=False,
        trust_repo=True,
        calling_fn="karaoker",
    )
except TypeError:
    repo_dir = torch.hub._get_cache_or_reload(  # pyright: ignore[reportPrivateUsage]
        "snakers4/silero-vad",
        force_reload=False,
    )

model_path = Path(repo_dir) / "src" / "silero_vad" / "data" / "silero_vad.jit"
assert model_path.exists(), f"missing silero model: {model_path}"
torch.jit.load(str(model_path), map_location=torch.device("cpu")).eval()

print("Silero VAD cached at:", model_dir)
PY
}

if [[ "${DOWNLOAD_MODELS}" -eq 1 ]]; then
  ensure_mfa_models
  if [[ "${ALIGNER_BACKEND}" == "sofa" ]]; then
    ensure_sofa_repo
    ensure_sofa_model
  fi
  if [[ "${SEPARATE}" -eq 1 ]]; then
    [[ -x "${AUDIO_SEP}" ]] || die "audio-separator not found in env at ${AUDIO_SEP}"
    ensure_audio_separator_model
    if [[ "${SILERO_VAD}" -eq 1 ]]; then
      ensure_silero_vad_model
    fi
  fi
  # Whisper model is downloaded on-demand when ASR backend is whispercpp (no --lrc).
fi

mkdir -p "${WORKDIR}"

cmd=(
  "${ENV_PREFIX}/bin/karaoker" run
  --input "${INPUT}"
  --workdir "${WORKDIR}"
  --asr-backend "${ASR_BACKEND}"
  --kana-backend "${KANA_BACKEND}"
  --gemini-model "${GEMINI_MODEL}"
  --ffmpeg "${FFMPEG}"
  --mfa "${MFA}"
  --aligner-backend "${ALIGNER_BACKEND}"
  --kana-output "${KANA_OUTPUT}"
  --mfa-acoustic-model "${MFA_ACOUSTIC_MODEL}"
  --mfa-f0 "${MFA_F0}"
  --mfa-f0-constant-hz "${MFA_F0_CONSTANT_HZ}"
  --mfa-f0-flatten-factor "${MFA_F0_FLATTEN_FACTOR}"
)

if [[ "${ALIGNER_BACKEND}" == "sofa" ]]; then
  ensure_sofa_repo
  [[ -d "${SOFA_MODEL_DIR}" ]] || mkdir -p "${SOFA_MODEL_DIR}"
  if [[ -z "${SOFA_CKPT}" ]]; then
    SOFA_CKPT="$(sofa_guess_ckpt)"
  fi
  if [[ -z "${SOFA_DICT}" ]]; then
    SOFA_DICT="$(sofa_guess_dict)"
  fi

  if [[ -z "${SOFA_CKPT}" || -z "${SOFA_DICT}" ]]; then
    ensure_sofa_model
    SOFA_CKPT="${SOFA_CKPT:-$(sofa_guess_ckpt)}"
    SOFA_DICT="${SOFA_DICT:-$(sofa_guess_dict)}"
  fi

  [[ -n "${SOFA_CKPT}" ]] || die "missing SOFA ckpt (pass --sofa-ckpt)"
  [[ -n "${SOFA_DICT}" ]] || die "missing SOFA dict (pass --sofa-dict)"

  cmd+=(
    --sofa-root "${SOFA_ROOT}"
    --sofa-python "${SOFA_PYTHON}"
    --sofa-ckpt "${SOFA_CKPT}"
    --sofa-dict "${SOFA_DICT}"
  )
fi

if [[ -n "${LRC}" ]]; then
  cmd+=( --lyrics-lrc "${LRC}" )
else
  case "${ASR_BACKEND}" in
    whispercpp)
      # ASR path (offline via whisper.cpp). We accept a model *name* and download if missing.
      ensure_whisper_model
      cmd+=( --whisper-cpp "${WHISPER_CLI}" --whisper-model "$(whisper_model_path)" )
      ;;
    gemini)
      # Gemini ASR path. Requires GEMINI_API_KEY in the environment.
      ;;
    *)
      die "unknown --asr-backend: ${ASR_BACKEND} (expected whispercpp|gemini)"
      ;;
  esac
fi

if [[ -n "${MFA_DICT}" ]]; then
  cmd+=( --mfa-dict "${MFA_DICT}" )
fi

if [[ "${MFA_F0_PRESERVE_UNVOICED}" -eq 0 ]]; then
  cmd+=( --no-mfa-f0-preserve-unvoiced )
fi

if [[ "${SEPARATE}" -eq 1 ]]; then
  [[ -x "${AUDIO_SEP}" ]] || die "audio-separator not found in env at ${AUDIO_SEP}"
  cmd+=( --audio-separator "${AUDIO_SEP}" --audio-separator-model "${AUDIO_SEP_MODEL}" )
  if [[ "${DEREVERB}" -eq 0 ]]; then
    cmd+=( --no-dereverb )
  else
    cmd+=( --dereverb-model "${DEREVERB_MODEL}" )
  fi
  if [[ "${SILERO_VAD}" -eq 0 ]]; then
    cmd+=( --no-silero-vad )
  fi
fi

echo "+ ${cmd[*]}"
exec "${cmd[@]}"
