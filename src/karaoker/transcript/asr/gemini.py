from __future__ import annotations

from karaoker.external.gemini import run_gemini_asr_with_kana
from karaoker.mapping import ScriptUnit
from karaoker.pipeline.types import (
    AsrResult,
    AudioAssets,
    PipelineConfig,
    PipelinePaths,
    TranscriptResult,
    TranscriptSegment,
)
from karaoker.transcript.asr.base import AsrTranscriptProvider


class GeminiTranscriptProvider(AsrTranscriptProvider):
    """
    Gemini-powered ASR provider that returns BOTH:
        - script text (lyrics)
        - spaced kana tokens (sung pronunciation)

    Notes:
    - Requires `google-genai` installed and `GEMINI_API_KEY` set in the environment.
    - Output tokens are normalized to match karaoker's expected kana tokenization.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
    ) -> None:
        self._model = model

    def transcribe(
        self, *, audio: AudioAssets, paths: PipelinePaths, config: PipelineConfig
    ) -> TranscriptResult:
        kana_output = config.kana_output
        if kana_output not in ("hiragana", "katakana"):
            raise ValueError(
                f"Unsupported kana_output={kana_output!r} (expected hiragana/katakana)."
            )

        asr_json = paths.asr_dir / "asr.json"
        payload = run_gemini_asr_with_kana(
            input_audio=audio.asr_input,
            output_json=asr_json,
            kana_output=kana_output,  # type: ignore[arg-type]
            model=self._model,
        )

        segs_raw = payload.get("segments")
        if not isinstance(segs_raw, list):
            raise ValueError(
                "Unexpected Gemini payload schema (missing 'segments' list)."
            )

        segments: list[TranscriptSegment] = []
        for seg_i, seg in enumerate(segs_raw):
            if not isinstance(seg, dict):
                continue
            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            start = seg.get("start")
            end = seg.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            start_s = max(0.0, float(start))
            end_s = float(end)
            if end_s <= start_s:
                end_s = start_s + 0.5

            kana = str(seg.get("kana", "")).strip()
            tokens = kana.split() if kana else []

            # Minimal mapping: one unit covering the whole line.
            # If Gemini failed to produce kana, leave script_units=None so KanaStage can fall back.
            units: list[ScriptUnit] | None = None
            ref_kana: str | None = None
            if tokens:
                units = [
                    ScriptUnit(
                        i=0,
                        text=text,
                        char_start=0,
                        char_end=len(text),
                        reading=" ".join(tokens),
                        ref_kana_tokens=tuple(tokens),
                        ref_kana_start=0,
                        ref_kana_end=len(tokens),
                    )
                ]
                ref_kana = " ".join(tokens)

            segments.append(
                TranscriptSegment(
                    text=text,
                    start=start_s,
                    end=end_s,
                    ref_kana=ref_kana,
                    script_units=units,
                    meta={"gemini_segment_i": seg_i},
                )
            )

        asr_info = AsrResult(provider="gemini", json_path=asr_json, payload=payload)

        return TranscriptResult(
            kind="asr",
            segments=segments,
            source_path=None,
            asr_result=asr_info,
        )
