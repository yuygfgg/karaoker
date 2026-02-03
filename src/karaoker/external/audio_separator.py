from __future__ import annotations

import os
import shutil
from pathlib import Path

from karaoker.utils import run_checked


def run_audio_separator(
    *,
    audio_separator: str,
    input_audio: Path,
    output_vocals: Path,
    model_file_dir: Path | None = None,
) -> None:
    """Run the audio-separator CLI and write a single vocal stem to `output_vocals`."""
    output_vocals.parent.mkdir(parents=True, exist_ok=True)
    exe = shutil.which(audio_separator) or audio_separator

    # macOS: torch/OpenMP can abort with a duplicate runtime. This suppresses that check.
    env = {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    if model_file_dir is not None:
        model_file_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "--output_dir",
        str(output_vocals.parent),
        "--output_format",
        "wav",
        "--single_stem",
        "vocals",
    ]
    if model_file_dir is not None:
        cmd += ["--model_file_dir", str(model_file_dir)]
    cmd += [str(input_audio)]

    run_checked(cmd, env={**env, "PATH": str(Path(exe).resolve().parent) + ":" + os.environ.get("PATH", "")})
    _pick_vocals(output_vocals)


def _pick_vocals(output_vocals: Path) -> None:
    """
    Pick a likely vocal stem from audio-separator outputs and copy it to `output_vocals`.
    """
    candidates = sorted(output_vocals.parent.glob("*vocals*.wav")) + sorted(
        output_vocals.parent.glob("*Vocal*.wav")
    )
    if not candidates:
        # Fall back to any wav in the output dir.
        candidates = sorted(output_vocals.parent.glob("*.wav"))
    if not candidates:
        raise FileNotFoundError(
            "audio-separator produced no wav outputs; check its logs/flags."
        )
    output_vocals.write_bytes(candidates[0].read_bytes())
