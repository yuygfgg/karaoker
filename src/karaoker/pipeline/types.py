from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from karaoker.mapping import ScriptUnit

MfaF0FlattenMode = Literal["none", "constant", "flatten"]


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    audio_dir: Path
    asr_dir: Path
    transcript_dir: Path
    alignment_dir: Path
    output_dir: Path

    @staticmethod
    def from_workdir(workdir: Path) -> "PipelinePaths":
        root = workdir
        return PipelinePaths(
            root=root,
            audio_dir=root / "audio",
            asr_dir=root / "asr",
            transcript_dir=root / "transcript",
            alignment_dir=root / "alignment",
            output_dir=root / "output",
        )


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    workdir: Path
    ffmpeg: str
    audio_separator: str | None
    audio_separator_model: str | None
    enable_dereverb: bool
    dereverb_model: str
    enable_silero_vad: bool
    silero_vad_threshold: float
    silero_vad_min_speech_ms: int
    silero_vad_min_silence_ms: int
    silero_vad_speech_pad_ms: int
    whisper_cpp: str | None
    whisper_model: Path | None
    mfa: str
    mfa_dict: str | None
    mfa_acoustic_model: str
    kana_output: str
    lyrics_lrc: Path | None
    asr_backend: str = "whispercpp"
    kana_backend: str = "mecab"
    gemini_model: str = "gemini-3-flash-preview"
    mfa_f0_mode: MfaF0FlattenMode = "none"
    mfa_f0_constant_hz: float = 150.0
    mfa_f0_flatten_factor: float = 0.0
    mfa_f0_preserve_unvoiced: bool = True


@dataclass
class AudioAssets:
    song_wav: Path
    asr_input: Path
    mfa_input: Path
    vocals_wav: Path | None
    vad_speech_segments_ms: list[tuple[int, int]] | None


@dataclass(frozen=True)
class AsrResult:
    provider: str
    json_path: Path | None
    payload: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    ref_kana: str | None = None
    script_units: list[ScriptUnit] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


TranscriptKind = Literal["lrc", "asr", "custom"]


@dataclass
class TranscriptResult:
    kind: TranscriptKind
    segments: list[TranscriptSegment]
    source_path: Path | None
    asr_result: AsrResult | None
    script_text: str | None = None
    ref_kana: str | None = None


@dataclass
class CorpusItem:
    utt_id: str
    wav_path: Path
    lab_path: Path
    segment: TranscriptSegment


@dataclass
class CorpusResult:
    corpus_dir: Path
    items: list[CorpusItem]
    all_kana_tokens: list[str]
    script_text: str | None
    ref_kana: str | None


@dataclass
class AlignmentResult:
    events: list[dict[str, object]]
    script_units_events: list[dict[str, object]] | None


@dataclass
class PipelineContext:
    config: PipelineConfig
    paths: PipelinePaths
    audio: AudioAssets | None = None
    transcript: TranscriptResult | None = None
    corpus: CorpusResult | None = None
    alignment: AlignmentResult | None = None
