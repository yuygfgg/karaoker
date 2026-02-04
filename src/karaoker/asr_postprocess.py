from __future__ import annotations

from typing import Any


def _merge_segments_ms(segments_ms: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent [start_ms, end_ms) segments."""
    if not segments_ms:
        return []

    segs = sorted((int(s), int(e)) for s, e in segments_ms if int(e) > int(s))
    if not segs:
        return []

    merged: list[tuple[int, int]] = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        # Treat adjacency as mergeable to avoid tiny gaps from rounding.
        if s <= cur_e:
            cur_e = max(cur_e, e)
            continue
        merged.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _has_speech_overlap_ms(
    *,
    segment_ms: tuple[int, int],
    speech_segments_ms: list[tuple[int, int]],
    min_overlap_ms: int = 1,
) -> bool:
    """Return True if `segment_ms` overlaps speech by at least `min_overlap_ms`."""
    start_ms, end_ms = segment_ms
    if end_ms <= start_ms:
        return False

    # `speech_segments_ms` is assumed sorted+merged.
    for s, e in speech_segments_ms:
        if e <= start_ms:
            continue
        if s >= end_ms:
            break
        overlap = min(end_ms, e) - max(start_ms, s)
        if overlap >= min_overlap_ms:
            return True
    return False


def drop_whisper_hallucinations_in_silence(
    whisper_json: dict[str, Any],
    *,
    speech_segments_ms: list[tuple[int, int]],
    min_overlap_ms: int = 1,
) -> dict[str, Any]:
    """
    Drop whisper.cpp transcription segments that fall entirely in VAD non-speech.
    """
    transcription = whisper_json.get("transcription")
    if not isinstance(transcription, list):
        return whisper_json

    speech = _merge_segments_ms(speech_segments_ms)
    if not speech:
        # If VAD returned no speech at all, don't drop anything.
        return whisper_json

    dropped = False
    kept: list[Any] = []
    for seg in transcription:
        if not isinstance(seg, dict):
            kept.append(seg)
            continue

        offsets = seg.get("offsets")
        if not isinstance(offsets, dict):
            kept.append(seg)
            continue

        start = offsets.get("from")
        end = offsets.get("to")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            kept.append(seg)
            continue

        start_ms = int(start)
        end_ms = int(end)
        if _has_speech_overlap_ms(
            segment_ms=(start_ms, end_ms),
            speech_segments_ms=speech,
            min_overlap_ms=min_overlap_ms,
        ):
            kept.append(seg)
        else:
            dropped = True

    if not dropped:
        return whisper_json

    out = dict(whisper_json)
    out["transcription"] = kept
    return out
