from __future__ import annotations

from pathlib import Path

from karaoker.pipeline import build_default_pipeline
from karaoker.pipeline.stages import KanaStage
from karaoker.pipeline.types import PipelineConfig


def _dummy_config(
    *,
    workdir: Path,
    kana_backend: str,
    audio_separator: str | None,
) -> PipelineConfig:
    return PipelineConfig(
        input_path=workdir / "song.flac",
        workdir=workdir,
        ffmpeg="ffmpeg",
        audio_separator=audio_separator,
        audio_separator_model=None,
        enable_lead_vocals=True,
        lead_vocals_model="UVR-BVE-4B_SN-44100-1.pth",
        lead_vocals_stem="auto",
        enable_dereverb=True,
        dereverb_model="dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
        enable_silero_vad=True,
        silero_vad_threshold=0.5,
        silero_vad_min_speech_ms=250,
        silero_vad_min_silence_ms=100,
        silero_vad_speech_pad_ms=30,
        whisper_cpp=None,
        whisper_model=None,
        mfa="mfa",
        mfa_dict=None,
        mfa_acoustic_model="japanese_mfa",
        kana_output="katakana",
        lyrics_lrc=workdir
        / "lyrics.lrc",  # Ensure build_default_pipeline uses LRC provider.
        asr_backend="whispercpp",
        kana_backend=kana_backend,
        gemini_model="gemini-3-flash-preview",
        mfa_f0_mode="none",
        mfa_f0_constant_hz=150.0,
        mfa_f0_flatten_factor=0.0,
        mfa_f0_preserve_unvoiced=True,
    )


def test_build_default_pipeline_feeds_dry_vocals_to_gemini_kana_when_available(
    tmp_path: Path,
) -> None:
    cfg = _dummy_config(
        workdir=tmp_path / "run",
        kana_backend="gemini",
        audio_separator="python -m audio_separator",
    )
    pipeline = build_default_pipeline(cfg)
    kana_stage = next(s for s in pipeline._stages if isinstance(s, KanaStage))
    converter = kana_stage._converter
    expected = (cfg.workdir / "audio" / "vocals_dry.wav").expanduser().resolve()
    assert getattr(converter, "_input_audio") == expected


def test_build_default_pipeline_does_not_feed_audio_when_separator_disabled(
    tmp_path: Path,
) -> None:
    cfg = _dummy_config(
        workdir=tmp_path / "run",
        kana_backend="gemini",
        audio_separator=None,
    )
    pipeline = build_default_pipeline(cfg)
    kana_stage = next(s for s in pipeline._stages if isinstance(s, KanaStage))
    converter = kana_stage._converter
    assert getattr(converter, "_input_audio") is None
