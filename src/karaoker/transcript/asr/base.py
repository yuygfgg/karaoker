from __future__ import annotations

from karaoker.transcript.base import TranscriptProvider


class AsrTranscriptProvider(TranscriptProvider):
    """
    Marker base class for ASR-backed transcript providers.

    Implementations should return a `TranscriptResult(kind="asr", ...)` with reasonably
    accurate segment timestamps (seconds) for downstream forced alignment.
    """


def build_asr_transcript_provider(name: str, /, **kwargs) -> AsrTranscriptProvider:
    """
    Convenience factory for ASR providers.

    This keeps CLI/pipeline code decoupled from concrete provider modules and avoids
    import cycles by importing backends lazily.
    """
    key = name.strip().lower()
    if key in {"whispercpp", "whisper.cpp", "whisper_cpp"}:
        from karaoker.transcript.asr.whispercpp import WhisperCppTranscriptProvider

        return WhisperCppTranscriptProvider()
    if key in {"gemini"}:
        from karaoker.transcript.asr.gemini import GeminiTranscriptProvider

        return GeminiTranscriptProvider(**kwargs)

    raise ValueError(f"Unknown ASR transcript provider: {name!r}")
