from __future__ import annotations

import json
from pathlib import Path

from karaoker.lyrics import parse_lrc
from karaoker.pipeline_types import (
    AudioAssets,
    PipelineConfig,
    PipelinePaths,
    TranscriptResult,
    TranscriptSegment,
)
from karaoker.transcript.textfile.base import TextFileTranscriptProvider


class LrcTranscriptProvider(TextFileTranscriptProvider):
    def __init__(self, *, lyrics_path: Path) -> None:
        self._lyrics_path = lyrics_path

    def transcribe(
        self, *, audio: AudioAssets, paths: PipelinePaths, config: PipelineConfig
    ) -> TranscriptResult:
        lrc_lines = parse_lrc(self._lyrics_path)
        segments = [
            TranscriptSegment(text=line.text, start=line.start, end=line.end)
            for line in lrc_lines
        ]

        (paths.asr_dir / "asr.json").write_text(
            json.dumps(
                {
                    "source": "lrc",
                    "num_lines": len(lrc_lines),
                    "path": str(self._lyrics_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return TranscriptResult(
            kind="lrc",
            segments=segments,
            source_path=self._lyrics_path,
            asr_result=None,
        )
