from __future__ import annotations

from pathlib import Path

from karaoker.utils import run_checked


def ensure_wav_16k_mono(*, ffmpeg: str, input_audio: Path, output_wav: Path) -> None:
    """Convert audio to 16 kHz mono 16-bit PCM WAV."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_checked(cmd)


def cut_wav_segment(
    *,
    ffmpeg: str,
    input_wav: Path,
    start_seconds: float,
    end_seconds: float,
    output_wav: Path,
) -> None:
    """Cut a WAV segment and write it as 16 kHz mono 16-bit PCM."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_wav),
        "-ss",
        f"{start_seconds:.3f}",
        "-to",
        f"{end_seconds:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_checked(cmd)
