from __future__ import annotations

from karaoker.asr_postprocess import drop_whisper_hallucinations_in_silence


def test_drop_whisper_hallucinations_in_silence_drops_fully_silent_segments() -> None:
    whisper_json = {
        "transcription": [
            {"offsets": {"from": 0, "to": 500}, "text": "aaa"},
            {"offsets": {"from": 500, "to": 900}, "text": "bbb"},
            {"offsets": {"from": 900, "to": 1100}, "text": "ccc"},
        ]
    }
    speech_segments_ms = [(800, 1000)]
    out = drop_whisper_hallucinations_in_silence(
        whisper_json, speech_segments_ms=speech_segments_ms
    )
    assert [x["text"] for x in out["transcription"]] == ["bbb", "ccc"]


def test_drop_whisper_hallucinations_in_silence_keeps_partial_overlap() -> None:
    whisper_json = {
        "transcription": [{"offsets": {"from": 900, "to": 1200}, "text": "keep"}]
    }
    speech_segments_ms = [(1000, 1100)]
    out = drop_whisper_hallucinations_in_silence(
        whisper_json, speech_segments_ms=speech_segments_ms
    )
    assert [x["text"] for x in out["transcription"]] == ["keep"]


def test_drop_whisper_hallucinations_in_silence_keeps_segments_without_offsets() -> (
    None
):
    whisper_json = {"transcription": [{"text": "no offsets"}]}
    out = drop_whisper_hallucinations_in_silence(
        whisper_json, speech_segments_ms=[(0, 1)]
    )
    assert out == whisper_json


def test_drop_whisper_hallucinations_in_silence_no_speech_is_noop() -> None:
    whisper_json = {
        "transcription": [{"offsets": {"from": 0, "to": 500}, "text": "aaa"}]
    }
    out = drop_whisper_hallucinations_in_silence(whisper_json, speech_segments_ms=[])
    assert out == whisper_json
