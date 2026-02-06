from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


F0FlattenMode = Literal["constant", "flatten"]


@dataclass(frozen=True)
class WorldF0FlattenSettings:
    mode: F0FlattenMode
    constant_hz: float = 150.0
    flatten_factor: float = 0.0
    preserve_unvoiced: bool = True
    frame_period_ms: float = 5.0
    f0_floor_hz: float = 71.0
    f0_ceil_hz: float = 800.0


def world_flatten_f0_wav(
    *,
    input_wav: Path,
    output_wav: Path,
    settings: WorldF0FlattenSettings,
    output_report_json: Path | None = None,
) -> None:
    """
    Flatten pitch (F0) with WORLD, keeping spectral envelope (SP) and aperiodicity (AP).

    This is intended as a pre-processing step for forced alignment (e.g., MFA) on singing voice:
    - Analyze: F0, SP, AP
    - Modify F0 (constant or variance compression)
    - Synthesize with modified F0 + original SP/AP
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
        import pyworld  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "pyworld is required for WORLD-based F0 flattening. "
            'Install it via `pip install -e ".[world]"` (or `pip install pyworld`).'
        ) from e

    sample_rate, pcm_s16 = _read_wav_s16le_mono(input_wav)
    x = np.frombuffer(pcm_s16, dtype="<i2").astype(np.float64) / 32768.0

    # WORLD analysis
    f0, t = pyworld.dio(
        x,
        sample_rate,
        f0_floor=float(settings.f0_floor_hz),
        f0_ceil=float(settings.f0_ceil_hz),
        frame_period=float(settings.frame_period_ms),
    )
    f0 = pyworld.stonemask(x, f0, t, sample_rate)
    sp = pyworld.cheaptrick(x, f0, t, sample_rate)
    ap = pyworld.d4c(x, f0, t, sample_rate)

    f0_new = _modify_f0(
        f0,
        mode=settings.mode,
        constant_hz=float(settings.constant_hz),
        flatten_factor=float(settings.flatten_factor),
        preserve_unvoiced=bool(settings.preserve_unvoiced),
        f0_floor_hz=float(settings.f0_floor_hz),
        f0_ceil_hz=float(settings.f0_ceil_hz),
    )

    y = pyworld.synthesize(
        f0_new,
        sp,
        ap,
        sample_rate,
        frame_period=float(settings.frame_period_ms),
    )

    # Keep duration stable so upstream segment timestamps still line up.
    y = _match_num_samples(y, target=int(x.shape[0]))

    pcm_out = np.clip(y, -1.0, 1.0)
    pcm_out = (pcm_out * 32767.0).astype("<i2")
    _write_wav_s16le_mono(
        output_wav, sample_rate=sample_rate, pcm_s16le=pcm_out.tobytes()
    )

    if output_report_json is not None:
        output_report_json.parent.mkdir(parents=True, exist_ok=True)
        voiced = (f0 > 0).astype(np.int64)
        voiced_new = (f0_new > 0).astype(np.int64)
        report = {
            "input_wav": str(input_wav),
            "output_wav": str(output_wav),
            "sample_rate_hz": int(sample_rate),
            "num_samples": int(x.shape[0]),
            "settings": {
                "mode": settings.mode,
                "constant_hz": float(settings.constant_hz),
                "flatten_factor": float(settings.flatten_factor),
                "preserve_unvoiced": bool(settings.preserve_unvoiced),
                "frame_period_ms": float(settings.frame_period_ms),
                "f0_floor_hz": float(settings.f0_floor_hz),
                "f0_ceil_hz": float(settings.f0_ceil_hz),
            },
            "f0_stats": {
                "num_frames": int(f0.shape[0]),
                "voiced_frames": int(voiced.sum()),
                "voiced_frames_new": int(voiced_new.sum()),
                "mean_hz_voiced": (
                    float(f0[f0 > 0].mean()) if int(voiced.sum()) else None
                ),
                "std_hz_voiced": float(f0[f0 > 0].std()) if int(voiced.sum()) else None,
                "mean_hz_voiced_new": (
                    float(f0_new[f0_new > 0].mean()) if int(voiced_new.sum()) else None
                ),
                "std_hz_voiced_new": (
                    float(f0_new[f0_new > 0].std()) if int(voiced_new.sum()) else None
                ),
            },
        }
        output_report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def _modify_f0(
    f0: "object",
    *,
    mode: F0FlattenMode,
    constant_hz: float,
    flatten_factor: float,
    preserve_unvoiced: bool,
    f0_floor_hz: float,
    f0_ceil_hz: float,
) -> "object":
    import numpy as np  # type: ignore[import-not-found]

    f0_arr = np.asarray(f0, dtype=np.float64)
    out = f0_arr.copy()
    if mode == "constant":
        const = float(constant_hz)
        const = float(np.clip(const, float(f0_floor_hz), float(f0_ceil_hz)))
        if preserve_unvoiced:
            out[f0_arr > 0] = const
        else:
            out[:] = const
        return out

    if mode == "flatten":
        alpha = float(flatten_factor)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"flatten_factor must be in [0,1], got: {alpha}")
        voiced = f0_arr > 0
        if not int(np.count_nonzero(voiced)):
            return out

        # Work in log-Hz space; pitch perception is roughly logarithmic.
        log_f0 = np.log(f0_arr[voiced])
        mu = float(log_f0.mean())
        log_f0_new = mu + (log_f0 - mu) * alpha
        out[voiced] = np.exp(log_f0_new)
        if not preserve_unvoiced:
            out[~voiced] = float(np.exp(mu))

        # Clip only positive values; keep unvoiced frames at 0 if preserve_unvoiced is set.
        pos = out > 0
        out[pos] = np.clip(out[pos], float(f0_floor_hz), float(f0_ceil_hz))
        return out

    raise ValueError(f"Unknown F0 flatten mode: {mode!r}")


def _match_num_samples(y: "object", *, target: int) -> "object":
    import numpy as np  # type: ignore[import-not-found]

    yy = np.asarray(y, dtype=np.float64)
    if int(yy.shape[0]) == int(target):
        return yy
    if int(yy.shape[0]) > int(target):
        return yy[:target]
    pad = int(target) - int(yy.shape[0])
    return np.pad(yy, (0, pad), mode="constant")


def _read_wav_s16le_mono(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(
                f"Expected mono wav, got {wf.getnchannels()} channels: {path}"
            )
        if wf.getsampwidth() != 2:
            raise ValueError(
                f"Expected 16-bit PCM wav, got sampwidth={wf.getsampwidth()}: {path}"
            )
        if wf.getcomptype() != "NONE":
            raise ValueError(
                f"Expected uncompressed PCM wav, got comptype={wf.getcomptype()}: {path}"
            )
        sample_rate = int(wf.getframerate())
        frames = wf.readframes(wf.getnframes())
    return sample_rate, frames


def _write_wav_s16le_mono(path: Path, *, sample_rate: int, pcm_s16le: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_s16le)
