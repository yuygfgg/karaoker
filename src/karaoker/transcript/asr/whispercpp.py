from __future__ import annotations

import json
import re
import shutil
from typing import Any

from karaoker.asr_postprocess import drop_whisper_hallucinations_in_silence
from karaoker.external.whispercpp import run_whisper_cpp
from karaoker.pipeline.types import (
    AsrResult,
    AudioAssets,
    PipelineConfig,
    PipelinePaths,
    TranscriptResult,
    TranscriptSegment,
)
from karaoker.transcript.asr.base import AsrTranscriptProvider


_RE_WHISPER_TIMESTAMP = re.compile(
    r"^(?P<h>\d+):(?P<m>[0-9]{2}):(?P<s>[0-9]{2})(?:[,.](?P<ms>[0-9]{1,3}))?$"
)


def _parse_whisper_timestamp_seconds(ts: str) -> float | None:
    m = _RE_WHISPER_TIMESTAMP.match(ts.strip())
    if not m:
        return None
    h = int(m.group("h"))
    mm = int(m.group("m"))
    ss = int(m.group("s"))
    ms_raw = m.group("ms") or "0"
    ms = int(ms_raw.ljust(3, "0")[:3])
    return h * 3600 + mm * 60 + ss + ms / 1000.0


def _whisper_segment_bounds_seconds(seg: dict[str, Any]) -> tuple[float, float] | None:
    offsets = seg.get("offsets")
    if isinstance(offsets, dict):
        start = offsets.get("from")
        end = offsets.get("to")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return float(start) / 1000.0, float(end) / 1000.0

    ts = seg.get("timestamps")
    if isinstance(ts, dict):
        start = ts.get("from")
        end = ts.get("to")
        if isinstance(start, str) and isinstance(end, str):
            start_s = _parse_whisper_timestamp_seconds(start)
            end_s = _parse_whisper_timestamp_seconds(end)
            if start_s is not None and end_s is not None:
                return start_s, end_s

    return None


class WhisperCppTranscriptProvider(AsrTranscriptProvider):
    def transcribe(
        self, *, audio: AudioAssets, paths: PipelinePaths, config: PipelineConfig
    ) -> TranscriptResult:
        if config.whisper_cpp is None or config.whisper_model is None:
            raise ValueError(
                "Either provide --lyrics-lrc or provide whisper_cpp + whisper_model for ASR."
            )

        asr_json = paths.asr_dir / "asr.json"
        asr_result = run_whisper_cpp(
            whisper_cpp=config.whisper_cpp,
            model_path=config.whisper_model,
            input_wav=audio.asr_input,
            output_json=asr_json,
        )
        if audio.vad_speech_segments_ms:
            asr_result2 = drop_whisper_hallucinations_in_silence(
                asr_result,
                speech_segments_ms=audio.vad_speech_segments_ms,
            )
            if asr_result2 is not asr_result:
                raw_path = asr_json.with_suffix(".raw.json")
                shutil.copyfile(asr_json, raw_path)
                asr_json.write_text(
                    json.dumps(asr_result2, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                asr_result = asr_result2

        transcription = (
            asr_result.get("transcription") if isinstance(asr_result, dict) else None
        )
        if not isinstance(transcription, list):
            keys = (
                list(asr_result.keys())
                if isinstance(asr_result, dict)
                else type(asr_result).__name__
            )
            raise ValueError(
                "Unexpected whisper.cpp JSON schema (expected top-level 'transcription'). "
                f"See: {asr_json} (top-level keys: {keys})"
            )

        segments: list[TranscriptSegment] = []
        for whisper_i, seg in enumerate(transcription):
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            bounds = _whisper_segment_bounds_seconds(seg)
            if bounds is None:
                continue
            start_s, end_s = bounds
            start_s = max(0.0, float(start_s))
            end_s = float(end_s)
            if end_s <= start_s:
                end_s = start_s + 0.5
            segments.append(
                TranscriptSegment(
                    text=text,
                    start=start_s,
                    end=end_s,
                    meta={"whisper_i": whisper_i},
                )
            )

        asr_info = AsrResult(provider="whisper.cpp", json_path=asr_json, payload=asr_result)

        return TranscriptResult(
            kind="asr",
            segments=segments,
            source_path=None,
            asr_result=asr_info,
        )
