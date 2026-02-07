from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


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
    audio_separator = audio_separator.strip()
    if not audio_separator:
        raise ValueError("audio_separator is empty")

    # Support either:
    # - a plain executable path/name, or
    # - a command string like "python -m audio_separator".
    #
    # If a user passes an explicit path with spaces, shlex-splitting would break it.
    if Path(audio_separator).exists():
        exe_parts = [audio_separator]
    else:
        exe_parts = shlex.split(audio_separator)
        if not exe_parts:
            raise ValueError("audio_separator is empty")
    exe0 = shutil.which(exe_parts[0]) or exe_parts[0]
    exe_dir = str(Path(exe0).resolve().parent)

    # macOS: torch/OpenMP can abort with a duplicate runtime. This suppresses that check.
    env = {
        "KMP_DUPLICATE_LIB_OK": "TRUE",
    }
    if model_file_dir is not None:
        model_file_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe0,
        *exe_parts[1:],
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

    merged_env = os.environ.copy()
    merged_env.update(env)
    # Ensure the executable directory is on PATH in case it relies on sibling binaries.
    merged_env["PATH"] = exe_dir + ":" + merged_env.get("PATH", "")

    proc = subprocess.run(
        cmd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = [
            "audio-separator failed:",
            "  " + " ".join(cmd),
        ]
        if proc.stdout.strip():
            msg.append("--- stdout ---")
            msg.append(proc.stdout.strip())
        if proc.stderr.strip():
            msg.append("--- stderr ---")
            msg.append(proc.stderr.strip())
        raise RuntimeError("\n".join(msg))

    try:
        _pick_output(output_audio, stem=(pick_stem or single_stem))
    except FileNotFoundError as e:
        # audio-separator can log errors but still exit 0; include the logs to aid debugging.
        def _tail(s: str, *, n: int = 200) -> str:
            lines = s.splitlines()
            if len(lines) <= n:
                return s
            return "\n".join(lines[-n:])

        msg = [str(e)]
        if proc.stdout.strip():
            msg.append("--- audio-separator stdout (tail) ---")
            msg.append(_tail(proc.stdout.strip()))
        if proc.stderr.strip():
            msg.append("--- audio-separator stderr (tail) ---")
            msg.append(_tail(proc.stderr.strip()))
        raise RuntimeError("\n".join(msg)) from e


def _pick_output(output_audio: Path, *, stem: str | None) -> None:
    """
    Pick a likely stem from audio-separator outputs and copy it to `output_audio`.

    This is intentionally heuristic; audio-separator output naming varies by model architecture.
    """
    wavs = sorted(
        {*output_audio.parent.glob("*.wav"), *output_audio.parent.glob("*.WAV")}
    )
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

    # Prefer the most recently written file (re-runs can leave old outputs around).
    wavs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    shutil.copyfile(wavs[0], output_audio)
