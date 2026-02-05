from __future__ import annotations

from karaoker.transcript.base import TranscriptProvider


class TextFileTranscriptProvider(TranscriptProvider):
    """Marker base class for transcript providers driven by text files (e.g. LRC)."""
