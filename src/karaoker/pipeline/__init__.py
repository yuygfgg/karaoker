from __future__ import annotations

from pathlib import Path

from karaoker.aligner import MfaAlignerProvider
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
from .types import PipelineConfig, PipelineContext, PipelinePaths

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


class KaraokerPipeline:
    def __init__(
        self,
        *,
        transcript_provider: TranscriptProvider,
        kana_converter: KanaConverter,
        aligner: MfaAlignerProvider,
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
        for stage in self._stages:
            stage.run(ctx)
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

    return KaraokerPipeline(
        transcript_provider=transcript_provider,
        kana_converter=build_kana_converter(config.kana_backend, model=config.gemini_model),
        aligner=MfaAlignerProvider(mfa=config.mfa),
    )


def run_pipeline(
    *,
    input_path: Path,
    workdir: Path,
    ffmpeg: str,
    audio_separator: str | None,
    audio_separator_model: str | None = None,
    enable_dereverb: bool = True,
    dereverb_model: str = "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    enable_silero_vad: bool = True,
    silero_vad_threshold: float = 0.5,
    silero_vad_min_speech_ms: int = 250,
    silero_vad_min_silence_ms: int = 100,
    silero_vad_speech_pad_ms: int = 30,
    whisper_cpp: str | None = None,
    whisper_model: Path | None = None,
    mfa: str,
    mfa_dict: str | None,
    mfa_acoustic_model: str,
    kana_output: str,
    lyrics_lrc: Path | None = None,
    asr_backend: str = "whispercpp",
    kana_backend: str = "pykakasi",
    gemini_model: str = "gemini-3-flash-preview",
) -> None:
    """
    Generate per-kana timing events for a song: audio -> transcript -> MFA alignment -> JSON.
    """
    input_path = input_path.expanduser().resolve()
    workdir = workdir.expanduser().resolve()

    config = PipelineConfig(
        input_path=input_path,
        workdir=workdir,
        ffmpeg=ffmpeg,
        audio_separator=audio_separator,
        audio_separator_model=audio_separator_model,
        enable_dereverb=enable_dereverb,
        dereverb_model=dereverb_model,
        enable_silero_vad=enable_silero_vad,
        silero_vad_threshold=silero_vad_threshold,
        silero_vad_min_speech_ms=silero_vad_min_speech_ms,
        silero_vad_min_silence_ms=silero_vad_min_silence_ms,
        silero_vad_speech_pad_ms=silero_vad_speech_pad_ms,
        whisper_cpp=whisper_cpp,
        whisper_model=whisper_model,
        mfa=mfa,
        mfa_dict=mfa_dict,
        mfa_acoustic_model=mfa_acoustic_model,
        kana_output=kana_output,
        lyrics_lrc=lyrics_lrc,
        asr_backend=asr_backend,
        kana_backend=kana_backend,
        gemini_model=gemini_model,
    )

    pipeline = build_default_pipeline(config)
    pipeline.run(config)

