from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest

from karaoker.f0_flatten import WorldF0FlattenSettings, _modify_f0, world_flatten_f0_wav


def test_modify_f0_constant_preserve_unvoiced() -> None:
    f0 = np.asarray([0.0, 100.0, 200.0, 0.0], dtype=np.float64)
    out = _modify_f0(
        f0,
        mode="constant",
        constant_hz=150.0,
        flatten_factor=0.0,
        preserve_unvoiced=True,
        f0_floor_hz=71.0,
        f0_ceil_hz=800.0,
    )
    assert np.allclose(out, [0.0, 150.0, 150.0, 0.0])


def test_modify_f0_constant_no_preserve_unvoiced() -> None:
    f0 = np.asarray([0.0, 100.0, 200.0, 0.0], dtype=np.float64)
    out = _modify_f0(
        f0,
        mode="constant",
        constant_hz=150.0,
        flatten_factor=0.0,
        preserve_unvoiced=False,
        f0_floor_hz=71.0,
        f0_ceil_hz=800.0,
    )
    assert np.allclose(out, 150.0)


def test_modify_f0_flatten_factor_bounds() -> None:
    f0 = np.asarray([100.0, 200.0], dtype=np.float64)
    with pytest.raises(ValueError, match=r"flatten_factor"):
        _modify_f0(
            f0,
            mode="flatten",
            constant_hz=150.0,
            flatten_factor=-0.1,
            preserve_unvoiced=True,
            f0_floor_hz=71.0,
            f0_ceil_hz=800.0,
        )
    with pytest.raises(ValueError, match=r"flatten_factor"):
        _modify_f0(
            f0,
            mode="flatten",
            constant_hz=150.0,
            flatten_factor=1.1,
            preserve_unvoiced=True,
            f0_floor_hz=71.0,
            f0_ceil_hz=800.0,
        )


def test_modify_f0_flatten_factor_zero() -> None:
    f0 = np.asarray([0.0, 100.0, 200.0, 0.0], dtype=np.float64)
    out = _modify_f0(
        f0,
        mode="flatten",
        constant_hz=150.0,
        flatten_factor=0.0,
        preserve_unvoiced=True,
        f0_floor_hz=71.0,
        f0_ceil_hz=800.0,
    )
    geom_mean = float(np.exp(np.mean(np.log([100.0, 200.0]))))
    np.testing.assert_allclose(
        out, [0.0, geom_mean, geom_mean, 0.0], rtol=1e-6, atol=1e-6
    )


def test_modify_f0_flatten_factor_one_noop() -> None:
    f0 = np.asarray([0.0, 100.0, 200.0, 0.0], dtype=np.float64)
    out = _modify_f0(
        f0,
        mode="flatten",
        constant_hz=150.0,
        flatten_factor=1.0,
        preserve_unvoiced=True,
        f0_floor_hz=71.0,
        f0_ceil_hz=800.0,
    )
    np.testing.assert_allclose(out, f0)


def _write_sine_wav(path: Path, *, sr: int = 16000, seconds: float = 0.05) -> None:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    x = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_world_flatten_f0_requires_pyworld(tmp_path: Path) -> None:
    if importlib.util.find_spec("pyworld") is not None:
        pytest.skip(
            "pyworld is installed; this test covers the missing-dependency error path."
        )

    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    _write_sine_wav(in_wav)

    with pytest.raises(ModuleNotFoundError, match=r"pyworld is required"):
        world_flatten_f0_wav(
            input_wav=in_wav,
            output_wav=out_wav,
            settings=WorldF0FlattenSettings(mode="constant", constant_hz=150.0),
        )
