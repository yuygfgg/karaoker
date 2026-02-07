from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter

from karaoker.aligner import AlignerProvider, MfaAlignerProvider, SofaAlignerProvider
from karaoker.kana_convert import KanaConverter, build_kana_converter
from karaoker.transcript import LrcTranscriptProvider, TranscriptProvider
from karaoker.transcript.asr.base import build_asr_transcript_provider

from .stages import (
    AlignmentStage,
    AudioStage,
    CorpusStage,
    ExportStage,
    KanaStage,
    TranscriptStage,
    WorkspaceStage,
)
from .types import MfaF0FlattenMode, PipelineConfig, PipelineContext, PipelinePaths

__all__ = [
    # API
    "KaraokerPipeline",
    "build_default_pipeline",
    "run_pipeline",
    # Types
    "PipelineConfig",
    "PipelineContext",
    "PipelinePaths",
]

logger = logging.getLogger(__name__)


class KaraokerPipeline:
    def __init__(
        self,
        *,
        transcript_provider: TranscriptProvider,
        kana_converter: KanaConverter,
        aligner: AlignerProvider,
    ) -> None:
        self._stages = [
            WorkspaceStage(),
            AudioStage(),
            TranscriptStage(transcript_provider),
            KanaStage(kana_converter),
            CorpusStage(),
            AlignmentStage(aligner),
            ExportStage(),
        ]

    def run(self, config: PipelineConfig) -> Path:
        paths = PipelinePaths.from_workdir(config.workdir)
        ctx = PipelineContext(config=config, paths=paths)
        logger.info("Workdir: %s", str(config.workdir))
        for stage in self._stages:
            stage_name = stage.__class__.__name__
            logger.info("[%s] start", stage_name)
            t0 = perf_counter()
            stage.run(ctx)
            dt = perf_counter() - t0
            logger.info("[%s] done (%.1fs)", stage_name, dt)
        logger.info("Wrote: %s", str(paths.output_dir / "subtitles.json"))
        return paths.output_dir / "subtitles.json"


def build_default_pipeline(config: PipelineConfig) -> KaraokerPipeline:
    if config.lyrics_lrc is not None:
        transcript_provider: TranscriptProvider = LrcTranscriptProvider(
            lyrics_path=config.lyrics_lrc
        )
    else:
        transcript_provider = build_asr_transcript_provider(
            config.asr_backend,
            model=config.gemini_model,
        )

    # If we have separated dry vocals available, feed them into the Gemini kana converter
    # so it can use sung pronunciation to disambiguate readings.
    kana_input_audio: Path | None = None
    if config.kana_backend.strip().lower() == "gemini" and config.audio_separator:
        paths = PipelinePaths.from_workdir(config.workdir)
        kana_input_audio = paths.audio_dir / "vocals_dry.wav"

    return KaraokerPipeline(
        transcript_provider=transcript_provider,
        kana_converter=build_kana_converter(
            config.kana_backend,
            model=config.gemini_model,
            input_audio=kana_input_audio,
        ),
        aligner=_build_aligner(config),
    )


def _build_aligner(config: PipelineConfig) -> AlignerProvider:
    backend = str(config.aligner_backend).strip().lower()
    if backend == "mfa":
        return MfaAlignerProvider(mfa=config.mfa)
    if backend == "sofa":
        if config.sofa_root is None:
            raise ValueError(
                "aligner_backend=sofa requires sofa_root (path to SOFA repo)."
            )
        return SofaAlignerProvider(
            sofa_python=config.sofa_python,
            sofa_root=Path(config.sofa_root),
        )
    raise ValueError(f"Unknown aligner_backend: {config.aligner_backend}")


def run_pipeline(
    *,
    input_path: Path,
    workdir: Path,
    ffmpeg: str,
    audio_separator: str | None,
    audio_separator_model: str | None = None,
    enable_lead_vocals: bool = True,
    lead_vocals_model: str = "UVR-BVE-4B_SN-44100-1.pth",
    lead_vocals_stem: str = "auto",
    enable_dereverb: bool = True,
    dereverb_model: str = "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    enable_silero_vad: bool = True,
    silero_vad_threshold: float = 0.5,
    silero_vad_min_speech_ms: int = 250,
    silero_vad_min_silence_ms: int = 100,
    silero_vad_speech_pad_ms: int = 30,
    whisper_cpp: str | None = None,
    whisper_model: Path | None = None,
    aligner_backend: str = "mfa",
    mfa: str,
    mfa_dict: str | None,
    mfa_acoustic_model: str,
    sofa_python: str = sys.executable,
    sofa_root: Path | None = None,
    sofa_dict: str | None = None,
    sofa_ckpt: str | None = None,
    kana_output: str,
    lyrics_lrc: Path | None = None,
    asr_backend: str = "whispercpp",
    kana_backend: str = "mecab",
    gemini_model: str = "gemini-3-flash-preview",
    mfa_f0_mode: MfaF0FlattenMode = "none",
    mfa_f0_constant_hz: float = 150.0,
    mfa_f0_flatten_factor: float = 0.0,
    mfa_f0_preserve_unvoiced: bool = True,
) -> None:
    """
    Generate per-kana timing events for a song: audio -> transcript -> forced alignment -> JSON.
    """
    input_path = input_path.expanduser().resolve()
    workdir = workdir.expanduser().resolve()

    config = PipelineConfig(
        input_path=input_path,
        workdir=workdir,
        ffmpeg=ffmpeg,
        audio_separator=audio_separator,
        audio_separator_model=audio_separator_model,
        enable_lead_vocals=enable_lead_vocals,
        lead_vocals_model=lead_vocals_model,
        lead_vocals_stem=lead_vocals_stem,
        enable_dereverb=enable_dereverb,
        dereverb_model=dereverb_model,
        enable_silero_vad=enable_silero_vad,
        silero_vad_threshold=silero_vad_threshold,
        silero_vad_min_speech_ms=silero_vad_min_speech_ms,
        silero_vad_min_silence_ms=silero_vad_min_silence_ms,
        silero_vad_speech_pad_ms=silero_vad_speech_pad_ms,
        whisper_cpp=whisper_cpp,
        whisper_model=whisper_model,
        aligner_backend=aligner_backend,
        mfa=mfa,
        mfa_dict=mfa_dict,
        mfa_acoustic_model=mfa_acoustic_model,
        sofa_python=sofa_python,
        sofa_root=str(sofa_root) if sofa_root is not None else None,
        sofa_dict=sofa_dict,
        sofa_ckpt=sofa_ckpt,
        kana_output=kana_output,
        lyrics_lrc=lyrics_lrc,
        asr_backend=asr_backend,
        kana_backend=kana_backend,
        gemini_model=gemini_model,
        mfa_f0_mode=mfa_f0_mode,
        mfa_f0_constant_hz=mfa_f0_constant_hz,
        mfa_f0_flatten_factor=mfa_f0_flatten_factor,
        mfa_f0_preserve_unvoiced=mfa_f0_preserve_unvoiced,
    )

    pipeline = build_default_pipeline(config)
    pipeline.run(config)
