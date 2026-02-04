from __future__ import annotations

import os
import shutil
from pathlib import Path

from karaoker.utils import run_checked


def run_audio_separator(
    *,
    audio_separator: str,
    input_audio: Path,
    output_audio: Path,
    model_file_dir: Path | None = None,
    model_filename: str | None = None,
    single_stem: str | None = "vocals",
    pick_stem: str | None = None,
) -> None:
    """
    Run the audio-separator CLI and copy a single output WAV to `output_audio`.

    Notes:
    - `single_stem` controls the `--single_stem` flag passed to audio-separator (when not None).
    - `pick_stem` controls which output file we copy from the output directory. If not set, we
      default it to `single_stem`.
    """
    output_audio.parent.mkdir(parents=True, exist_ok=True)
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
        str(output_audio.parent),
        "--output_format",
        "wav",
    ]
    if model_filename is not None:
        cmd += ["--model_filename", model_filename]
    if model_file_dir is not None:
        cmd += ["--model_file_dir", str(model_file_dir)]
    if single_stem is not None:
        cmd += ["--single_stem", single_stem]
    cmd += [str(input_audio)]

    run_checked(cmd, env={**env, "PATH": str(Path(exe).resolve().parent) + ":" + os.environ.get("PATH", "")})
    _pick_output(output_audio, stem=(pick_stem or single_stem))


def _pick_output(output_audio: Path, *, stem: str | None) -> None:
    """
    Pick a likely stem from audio-separator outputs and copy it to `output_audio`.

    This is intentionally heuristic; audio-separator output naming varies by model architecture.
    """
    wavs = sorted(output_audio.parent.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(
            "audio-separator produced no wav outputs; check its logs/flags."
        )

    # Avoid selecting the destination itself (e.g., re-runs in the same directory).
    wavs = [p for p in wavs if p.resolve() != output_audio.resolve()]
    if not wavs:
        raise FileNotFoundError(
            "audio-separator produced wav outputs, but only the destination file exists; "
            "check the output directory selection."
        )

    if stem:
        stem_l = stem.lower()
        stem_compact = stem_l.replace(" ", "").replace("-", "").replace("_", "")

        def matches(p: Path) -> bool:
            name_l = p.name.lower()
            name_compact = name_l.replace(" ", "").replace("-", "").replace("_", "")
            return stem_l in name_l or stem_compact in name_compact

        stem_candidates = [p for p in wavs if matches(p)]
        if stem_candidates:
            wavs = stem_candidates

    shutil.copyfile(wavs[0], output_audio)
