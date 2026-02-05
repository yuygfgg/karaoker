from __future__ import annotations

import json
import shutil
from pathlib import Path

from karaoker.aligner import AlignerProvider
from karaoker.external.ffmpeg import cut_wav_segment, ensure_wav_16k_mono
from karaoker.kana import KanaConverter
from karaoker.mapping import ScriptUnit, map_kana_events_to_script
from karaoker.pipeline_types import (
    AlignmentResult,
    AudioAssets,
    CorpusItem,
    CorpusResult,
    PipelineContext,
)
from karaoker.textgrid_parser import textgrid_to_kana_events
from karaoker.transcript import TranscriptProvider


def _project_root() -> Path:
    # repo_root/src/karaoker/pipeline_stages.py -> parents[2] == repo_root
    return Path(__file__).resolve().parents[2]


def _ref_kana_from_units(units: list[ScriptUnit]) -> str:
    tokens: list[str] = []
    for unit in units:
        tokens.extend(unit.ref_kana_tokens)
    return " ".join(tokens)


class WorkspaceStage:
    def run(self, ctx: PipelineContext) -> None:
        ctx.config.workdir.mkdir(parents=True, exist_ok=True)
        for d in [
            ctx.paths.audio_dir,
            ctx.paths.asr_dir,
            ctx.paths.transcript_dir,
            ctx.paths.alignment_dir,
            ctx.paths.output_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


class AudioStage:
    def run(self, ctx: PipelineContext) -> None:
        config = ctx.config
        paths = ctx.paths

        song_wav = paths.audio_dir / "song.wav"
        ensure_wav_16k_mono(
            ffmpeg=config.ffmpeg,
            input_audio=config.input_path,
            output_wav=song_wav,
        )

        vocals_wav: Path | None = None
        vad_speech_segments_ms: list[tuple[int, int]] | None = None
        asr_input = song_wav

        if config.audio_separator:
            from karaoker.external.audio_separator import run_audio_separator

            sep_dir = paths.audio_dir / "audio_separator"
            dereverb_dir = paths.audio_dir / "dereverb"
            vocals_raw = sep_dir / "vocals_raw.wav"

            run_audio_separator(
                audio_separator=config.audio_separator,
                input_audio=config.input_path,
                output_audio=vocals_raw,
                model_filename=config.audio_separator_model,
                # Cache models under the repo so repeated runs reuse downloads.
                model_file_dir=_project_root() / "models" / "audio_separator",
            )

            vocals_dry_raw = vocals_raw
            if config.enable_dereverb:
                vocals_dry_raw = dereverb_dir / "vocals_dry_raw.wav"
                run_audio_separator(
                    audio_separator=config.audio_separator,
                    input_audio=vocals_raw,
                    output_audio=vocals_dry_raw,
                    model_filename=config.dereverb_model,
                    single_stem="noreverb",
                    pick_stem="noreverb",
                    model_file_dir=_project_root() / "models" / "audio_separator",
                )

            vocals_dry_16k = paths.audio_dir / "vocals_dry.wav"
            ensure_wav_16k_mono(
                ffmpeg=config.ffmpeg,
                input_audio=vocals_dry_raw,
                output_wav=vocals_dry_16k,
            )

            vocals_wav = paths.audio_dir / "vocals.wav"
            if config.enable_silero_vad:
                from karaoker.external.silero_vad import zero_non_speech_with_silero_vad

                vad_speech_segments_ms = zero_non_speech_with_silero_vad(
                    input_wav=vocals_dry_16k,
                    output_wav=vocals_wav,
                    model_dir=_project_root() / "models" / "silero_vad",
                    threshold=config.silero_vad_threshold,
                    min_speech_duration_ms=config.silero_vad_min_speech_ms,
                    min_silence_duration_ms=config.silero_vad_min_silence_ms,
                    speech_pad_ms=config.silero_vad_speech_pad_ms,
                    output_segments_json=paths.asr_dir / "vad_speech.json",
                )
            else:
                shutil.copyfile(vocals_dry_16k, vocals_wav)

            asr_input = vocals_wav

        ctx.audio = AudioAssets(
            song_wav=song_wav,
            asr_input=asr_input,
            vocals_wav=vocals_wav,
            vad_speech_segments_ms=vad_speech_segments_ms,
        )


class TranscriptStage:
    def __init__(self, provider: TranscriptProvider) -> None:
        self._provider = provider

    def run(self, ctx: PipelineContext) -> None:
        if ctx.audio is None:
            raise ValueError("Audio stage must run before transcript stage.")
        ctx.transcript = self._provider.transcribe(
            audio=ctx.audio,
            paths=ctx.paths,
            config=ctx.config,
        )


class KanaStage:
    def __init__(self, converter: KanaConverter) -> None:
        self._converter = converter

    def run(self, ctx: PipelineContext) -> None:
        if ctx.transcript is None:
            raise ValueError("Transcript stage must run before kana stage.")

        for seg in ctx.transcript.segments:
            if seg.script_units is None:
                ref_kana, units = self._converter.to_kana(
                    seg.text, output=ctx.config.kana_output
                )
                seg.ref_kana = ref_kana
                seg.script_units = units
            elif seg.ref_kana is None:
                seg.ref_kana = _ref_kana_from_units(seg.script_units)

        if ctx.transcript.ref_kana is None:
            parts = [seg.ref_kana for seg in ctx.transcript.segments if seg.ref_kana]
            ctx.transcript.ref_kana = " ".join(parts).strip()


class CorpusStage:
    def run(self, ctx: PipelineContext) -> None:
        if ctx.audio is None or ctx.transcript is None:
            raise ValueError("Audio and transcript stages must run before corpus stage.")

        corpus_dir = ctx.paths.alignment_dir / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        for p in list(corpus_dir.glob("utt_*.wav")) + list(corpus_dir.glob("utt_*.lab")):
            p.unlink(missing_ok=True)

        items: list[CorpusItem] = []
        all_tokens: list[str] = []
        kana_lines: list[str] = []
        script_parts: list[str] = []
        script_len = 0

        for seg in ctx.transcript.segments:
            ref_kana = (seg.ref_kana or "").strip()
            if ctx.transcript.kind != "lrc" and not ref_kana:
                continue

            utt_id = f"utt_{len(items):04d}"
            out_wav = corpus_dir / f"{utt_id}.wav"
            out_lab = corpus_dir / f"{utt_id}.lab"

            out_lab.write_text(ref_kana + "\n", encoding="utf-8")
            kana_lines.append(ref_kana)
            all_tokens.extend(ref_kana.split())

            start_s = max(0.0, float(seg.start))
            end_s = float(seg.end)
            if end_s <= start_s:
                end_s = start_s + 0.5

            cut_wav_segment(
                ffmpeg=ctx.config.ffmpeg,
                input_wav=ctx.audio.asr_input,
                start_seconds=start_s,
                end_seconds=end_s,
                output_wav=out_wav,
            )

            seg.meta["utt_id"] = utt_id
            seg.meta["segment_i"] = len(items)

            if ctx.transcript.kind != "lrc":
                if script_parts:
                    script_len += 1
                seg.meta["char_offset"] = script_len
                script_parts.append(seg.text)
                script_len += len(seg.text)

            items.append(
                CorpusItem(
                    utt_id=utt_id,
                    wav_path=out_wav,
                    lab_path=out_lab,
                    segment=seg,
                )
            )

        if ctx.transcript.kind != "lrc":
            script_text = " ".join(script_parts).strip()
            if not script_text:
                raise ValueError("whisper.cpp JSON 'transcription' contained no usable text to align.")
            if not items:
                raise ValueError(
                    "whisper.cpp produced no usable kana segments to align. "
                    "Try providing --lyrics-lrc or a different whisper model."
                )
            ctx.transcript.script_text = script_text
        else:
            script_text = None

        if ctx.transcript.kind != "lrc" and items:
            ref_kana = " ".join(kana_lines).strip()
        else:
            ref_kana = None

        ctx.paths.transcript_dir.mkdir(parents=True, exist_ok=True)
        (ctx.paths.transcript_dir / "kana_spaced.txt").write_text(
            "\n".join(kana_lines) + "\n", encoding="utf-8"
        )
        (ctx.paths.transcript_dir / "script.txt").write_text(
            "\n".join(item.segment.text for item in items) + "\n", encoding="utf-8"
        )

        ctx.transcript.segments = [item.segment for item in items]

        ctx.corpus = CorpusResult(
            corpus_dir=corpus_dir,
            items=items,
            all_kana_tokens=all_tokens,
            script_text=script_text,
            ref_kana=ref_kana,
        )


class AlignmentStage:
    def __init__(self, aligner: AlignerProvider) -> None:
        self._aligner = aligner

    def run(self, ctx: PipelineContext) -> None:
        if ctx.corpus is None or ctx.transcript is None:
            raise ValueError("Corpus stage must run before alignment stage.")

        mfa_dict_to_use = ctx.config.mfa_dict
        if mfa_dict_to_use is None:
            tokens = sorted(set(ctx.corpus.all_kana_tokens))
            words_path = ctx.paths.alignment_dir / "words.txt"
            words_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

            g2p_model = (
                "japanese_katakana_mfa"
                if ctx.config.kana_output == "katakana"
                else "japanese_mfa"
            )
            gen_dict = ctx.paths.alignment_dir / "g2p.dict"
            self._aligner.g2p(
                word_list=words_path,
                g2p_model=g2p_model,
                output_dictionary=gen_dict,
            )
            mfa_dict_to_use = str(gen_dict)

        out_dir = ctx.paths.alignment_dir / "textgrids"
        for p in out_dir.rglob("*.TextGrid"):
            p.unlink(missing_ok=True)
        self._aligner.align_corpus(
            corpus_dir=ctx.corpus.corpus_dir,
            pronunciation_dict=mfa_dict_to_use,
            acoustic_model=ctx.config.mfa_acoustic_model,
            output_dir=out_dir,
        )

        if ctx.transcript.kind == "lrc":
            events: list[dict[str, object]] = []
            script_units_events: list[dict[str, object]] = []
            for idx, item in enumerate(ctx.corpus.items):
                tg = out_dir / f"{item.utt_id}.TextGrid"
                if not tg.exists():
                    continue

                line_events = textgrid_to_kana_events(tg, offset_seconds=item.segment.start)
                for e in line_events:
                    e["line_i"] = idx
                    e["line_event_i"] = e.get("i")

                units = item.segment.script_units or []
                mapped_events, timed_units = map_kana_events_to_script(
                    script_text=item.segment.text,
                    script_units=units,
                    kana_events=line_events,
                )
                for e in mapped_events:
                    e["line_i"] = idx
                for u in timed_units:
                    u["line_i"] = idx

                events.extend(mapped_events)
                script_units_events.extend(timed_units)

            events.sort(key=lambda e: float(e["start"]))  # type: ignore[index]
            for i, e in enumerate(events):
                e["i"] = i

        else:
            events = []
            script_units_events = []
            unit_base = 0
            for seg_i, item in enumerate(ctx.corpus.items):
                tg = out_dir / f"{item.utt_id}.TextGrid"
                if not tg.exists():
                    unit_base += len(item.segment.script_units or [])
                    continue

                seg_events0 = textgrid_to_kana_events(
                    tg, offset_seconds=float(item.segment.start)
                )
                for e in seg_events0:
                    e["segment_i"] = seg_i
                    e["segment_event_i"] = e.get("i")
                    e["utt_id"] = item.utt_id
                    e["whisper_i"] = int(item.segment.meta.get("whisper_i", seg_i))

                units = item.segment.script_units or []
                mapped_seg_events, timed_units = map_kana_events_to_script(
                    script_text=item.segment.text,
                    script_units=units,
                    kana_events=seg_events0,
                )

                char_offset = int(item.segment.meta.get("char_offset", 0))
                for e in mapped_seg_events:
                    e["segment_i"] = seg_i
                    e["utt_id"] = item.utt_id
                    e["whisper_i"] = int(item.segment.meta.get("whisper_i", seg_i))
                    if isinstance(e.get("script_unit_i"), int):
                        e["script_unit_i"] = int(e["script_unit_i"]) + unit_base
                    if isinstance(e.get("script_char_start"), int):
                        e["script_char_start"] = int(e["script_char_start"]) + char_offset
                    if isinstance(e.get("script_char_end"), int):
                        e["script_char_end"] = int(e["script_char_end"]) + char_offset

                for u in timed_units:
                    u["segment_i"] = seg_i
                    u["utt_id"] = item.utt_id
                    u["whisper_i"] = int(item.segment.meta.get("whisper_i", seg_i))
                    u["i"] = int(u["i"]) + unit_base
                    u["char_start"] = int(u["char_start"]) + char_offset
                    u["char_end"] = int(u["char_end"]) + char_offset

                events.extend(mapped_seg_events)
                script_units_events.extend(timed_units)
                unit_base += len(units)

            events.sort(key=lambda e: float(e["start"]))  # type: ignore[index]
            for i, e in enumerate(events):
                e["i"] = i

        ctx.alignment = AlignmentResult(
            events=events,
            script_units_events=script_units_events or None,
        )


class ExportStage:
    def run(self, ctx: PipelineContext) -> None:
        if ctx.transcript is None or ctx.corpus is None or ctx.alignment is None:
            raise ValueError("Alignment stage must run before export stage.")

        if ctx.transcript.kind == "lrc":
            source_meta = {
                "type": "lrc",
                "path": str(ctx.transcript.source_path),
                "num_lines": len(ctx.transcript.segments),
            }
        else:
            asr_result = ctx.transcript.asr_result
            source_meta = {
                "type": "asr",
                "path": str(asr_result.json_path) if asr_result else None,
                "provider": asr_result.provider if asr_result else None,
            }

        out: dict[str, object] = {
            "version": 2,
            "language": "ja",
            "kana_output": ctx.config.kana_output,
            "units": "kana",
            "source": source_meta,
            "events": ctx.alignment.events,
        }

        if ctx.transcript.kind == "lrc":
            out["lines"] = [
                {
                    "i": i,
                    "start": round(float(seg.start), 4),
                    "end": round(float(seg.end), 4),
                    "text": seg.text,
                    "ref_kana": seg.ref_kana or "",
                }
                for i, seg in enumerate(ctx.transcript.segments)
            ]
        else:
            out["script"] = {
                "text": ctx.transcript.script_text or "",
                "ref_kana": ctx.corpus.ref_kana or "",
            }
            out["segments"] = [
                {
                    "i": i,
                    "utt_id": item.utt_id,
                    "whisper_i": int(item.segment.meta.get("whisper_i", i)),
                    "start": round(float(item.segment.start), 4),
                    "end": round(float(item.segment.end), 4),
                    "text": item.segment.text,
                    "ref_kana": item.segment.ref_kana or "",
                }
                for i, item in enumerate(ctx.corpus.items)
            ]

        if ctx.alignment.script_units_events is not None:
            out["script_units"] = ctx.alignment.script_units_events

        (ctx.paths.output_dir / "subtitles.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
