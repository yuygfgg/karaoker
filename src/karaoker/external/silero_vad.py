from __future__ import annotations

import array
import os
import sys
import warnings
import wave
from pathlib import Path
from typing import Callable


def zero_outside_segments_s16(
    pcm_s16: array.array,
    *,
    segments: list[tuple[int, int]],
) -> array.array:
    """Return a copy of `pcm_s16` with everything outside `segments` set to 0."""
    out = array.array("h", [0]) * len(pcm_s16)
    n = len(pcm_s16)
    for start, end in segments:
        s = max(0, min(n, int(start)))
        e = max(0, min(n, int(end)))
        if e <= s:
            continue
        out[s:e] = pcm_s16[s:e]
    return out


def _silero_get_speech_timestamps(
    audio,
    model,
    *,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    time_resolution: int = 1,
    visualize_probs: bool = False,
    progress_tracking_callback: Callable[[float], None] | None = None,
    neg_threshold: float | None = None,
    window_size_samples: int = 512,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
) -> list[dict[str, int | float]]:
    """
    A minimal copy of Silero VAD `get_speech_timestamps` that avoids importing torchaudio.

    Upstream Silero's `hubconf.py` imports torchaudio unconditionally, but we already do audio I/O
    ourselves (via the stdlib `wave` module). So we re-implement the timestamp extraction here
    and load the bundled TorchScript model file directly.
    """
    import torch

    # Derived from https://github.com/snakers4/silero-vad (MIT).
    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except Exception as e:  # pragma: no cover
            raise TypeError("Audio cannot be casted to tensor. Cast it manually") from e

    if len(audio.shape) > 1:
        for _ in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process 2ch audio?")

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn("Sampling rate is a multiple of 16000, casting to 16000 manually!", stacklevel=2)
    else:
        step = 1

    if sampling_rate not in [8000, 16000]:
        raise ValueError(
            "Silero VAD supports 8000 and 16000 (or multiple of 16000) sample rates only"
        )

    # NOTE: In upstream code this argument is deprecated/ignored; we match that behavior.
    window_size_samples = 512 if sampling_rate == 16000 else 256

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

        progress = current_start_sample + window_size_samples
        if progress > audio_length_samples:
            progress = audio_length_samples
        if progress_tracking_callback:
            progress_tracking_callback((progress / audio_length_samples) * 100)

    triggered = False
    speeches: list[dict[str, int]] = []
    current_speech: dict[str, int] = {}

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0
    prev_end = next_start = 0
    possible_ends: list[tuple[int, float]] = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = cur_sample
            continue

        if triggered and (cur_sample - current_speech["start"] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = int(prev_end + dur)

                if next_start < prev_end + cur_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech["end"] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if (not use_max_poss_sil_at_max_speech) and (sil_dur_now > min_silence_samples_at_max_speech):
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            current_speech["end"] = temp_end
            if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                speeches.append(current_speech)
            current_speech = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            possible_ends = []
            continue

    if current_speech and (audio_length_samples - current_speech["start"]) > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - speech_pad_samples))
        else:
            speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        out: list[dict[str, int | float]] = []
        for speech_dict in speeches:
            out.append(
                {
                    "start": max(round(speech_dict["start"] / sampling_rate, time_resolution), 0),
                    "end": min(
                        round(speech_dict["end"] / sampling_rate, time_resolution), audio_length_seconds
                    ),
                }
            )
        return out

    if step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    if visualize_probs:
        # Intentionally omitted to keep deps minimal.
        pass

    return speeches


def _load_silero_vad_model(*, model_dir: Path):
    import torch

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(model_dir))

    # Download/extract the repo to get the bundled TorchScript model file.
    # This avoids importing hubconf.py (which depends on torchaudio).
    try:
        repo_dir = torch.hub._get_cache_or_reload(  # pyright: ignore[reportPrivateUsage]
            "snakers4/silero-vad",
            force_reload=False,
            trust_repo=True,
            calling_fn="karaoker",
        )
    except TypeError:  # pragma: no cover - depends on torch version
        repo_dir = torch.hub._get_cache_or_reload(  # pyright: ignore[reportPrivateUsage]
            "snakers4/silero-vad",
            force_reload=False,
        )

    model_path = Path(repo_dir) / "src" / "silero_vad" / "data" / "silero_vad.jit"
    if not model_path.exists():
        raise FileNotFoundError(f"Silero VAD TorchScript model not found: {model_path}")

    model = torch.jit.load(str(model_path), map_location=torch.device("cpu"))
    model.eval()
    return model


def zero_non_speech_with_silero_vad(
    *,
    input_wav: Path,
    output_wav: Path,
    model_dir: Path,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> None:
    """
    Use Silero VAD to zero out non-speech regions (set samples to exactly 0).

    `input_wav` must be 16 kHz mono 16-bit PCM (this is what `ensure_wav_16k_mono` produces).
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    sample_rate, pcm_s16 = _read_wav_s16le_mono(input_wav)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz wav for Silero VAD, got {sample_rate}Hz: {input_wav}")

    try:
        # macOS: torch/OpenMP can abort with a duplicate runtime. This suppresses that check.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch
    except Exception as e:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Silero VAD step requires PyTorch, but it could not be imported. "
            "If you installed karaoker without audio-separator, install torch first."
        ) from e

    # NOTE: This downloads the Silero VAD repo on first run (network required).
    model = _load_silero_vad_model(model_dir=model_dir)

    audio = torch.tensor(pcm_s16, dtype=torch.float32) / 32768.0
    speech = _silero_get_speech_timestamps(
        audio,
        model,  # TorchScript model
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    segments = [(int(x["start"]), int(x["end"])) for x in speech if "start" in x and "end" in x]
    out_pcm = zero_outside_segments_s16(pcm_s16, segments=segments)
    _write_wav_s16le_mono(output_wav, sample_rate=sample_rate, pcm_s16=out_pcm)


def _read_wav_s16le_mono(path: Path) -> tuple[int, array.array]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono wav, got {wf.getnchannels()} channels: {path}")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM wav, got sampwidth={wf.getsampwidth()}: {path}")
        if wf.getcomptype() != "NONE":
            raise ValueError(f"Expected uncompressed PCM wav, got comptype={wf.getcomptype()}: {path}")
        sample_rate = int(wf.getframerate())
        frames = wf.readframes(wf.getnframes())

    pcm = array.array("h")
    pcm.frombytes(frames)
    if sys.byteorder == "big":
        pcm.byteswap()
    return sample_rate, pcm


def _write_wav_s16le_mono(path: Path, *, sample_rate: int, pcm_s16: array.array) -> None:
    pcm = pcm_s16
    if sys.byteorder == "big":
        pcm = array.array("h", pcm_s16)
        pcm.byteswap()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
