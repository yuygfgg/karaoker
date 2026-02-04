from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from karaoker.external.ffmpeg import ensure_wav_16k_mono
from karaoker.external.mfa import run_mfa_align, run_mfa_align_corpus, run_mfa_g2p
from karaoker.external.whispercpp import run_whisper_cpp
from karaoker.lyrics import parse_lrc
from karaoker.mapping import map_kana_events_to_script, to_spaced_kana_with_units
from karaoker.textgrid_parser import textgrid_to_kana_events


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


def _project_root() -> Path:
    # repo_root/src/karaoker/pipeline.py -> parents[2] == repo_root
    return Path(__file__).resolve().parents[2]


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
    whisper_cpp: str | None,
    whisper_model: Path | None,
    mfa: str,
    mfa_dict: str | None,
    mfa_acoustic_model: str,
    kana_output: str,
    lyrics_lrc: Path | None = None,
) -> None:
    """
    Generate per-kana timing events for a song: audio -> transcript -> MFA alignment -> JSON.
    """
    input_path = input_path.expanduser().resolve()
    workdir = workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    paths = PipelinePaths.from_workdir(workdir)
    for d in [
        paths.audio_dir,
        paths.asr_dir,
        paths.transcript_dir,
        paths.alignment_dir,
        paths.output_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) Decode/convert to wav (and optionally separate vocals)
    song_wav = paths.audio_dir / "song.wav"
    ensure_wav_16k_mono(ffmpeg=ffmpeg, input_audio=input_path, output_wav=song_wav)

    vocals_wav = paths.audio_dir / "vocals.wav"
    if audio_separator:
        from karaoker.external.audio_separator import run_audio_separator

        sep_dir = paths.audio_dir / "audio_separator"
        dereverb_dir = paths.audio_dir / "dereverb"
        vocals_raw = sep_dir / "vocals_raw.wav"
        run_audio_separator(
            audio_separator=audio_separator,
            input_audio=input_path,
            output_audio=vocals_raw,
            model_filename=audio_separator_model,
            # Cache models under the repo so repeated runs reuse downloads.
            model_file_dir=_project_root() / "models" / "audio_separator",
        )

        vocals_dry_raw = vocals_raw
        if enable_dereverb:
            vocals_dry_raw = dereverb_dir / "vocals_dry_raw.wav"
            run_audio_separator(
                audio_separator=audio_separator,
                input_audio=vocals_raw,
                output_audio=vocals_dry_raw,
                model_filename=dereverb_model,
                single_stem="noreverb",
                pick_stem="noreverb",
                model_file_dir=_project_root() / "models" / "audio_separator",
            )

        vocals_dry_16k = paths.audio_dir / "vocals_dry.wav"
        ensure_wav_16k_mono(ffmpeg=ffmpeg, input_audio=vocals_dry_raw, output_wav=vocals_dry_16k)

        if enable_silero_vad:
            from karaoker.external.silero_vad import zero_non_speech_with_silero_vad

            zero_non_speech_with_silero_vad(
                input_wav=vocals_dry_16k,
                output_wav=vocals_wav,
                model_dir=_project_root() / "models" / "silero_vad",
                threshold=silero_vad_threshold,
                min_speech_duration_ms=silero_vad_min_speech_ms,
                min_silence_duration_ms=silero_vad_min_silence_ms,
                speech_pad_ms=silero_vad_speech_pad_ms,
            )
        else:
            shutil.copyfile(vocals_dry_16k, vocals_wav)

        asr_input = vocals_wav
    else:
        asr_input = song_wav

    # 2) Transcript source: LRC lyrics (preferred) OR ASR (whisper.cpp)
    lrc_lines = None
    if lyrics_lrc is not None:
        lrc_lines = parse_lrc(lyrics_lrc)

    script_text: str | None = None
    if lrc_lines:
        (paths.asr_dir / "asr.json").write_text(
            json.dumps(
                {
                    "source": "lrc",
                    "num_lines": len(lrc_lines),
                    "path": str(lyrics_lrc),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        if whisper_cpp is None or whisper_model is None:
            raise ValueError("Either provide --lyrics-lrc or provide whisper_cpp + whisper_model for ASR.")
        asr_json = paths.asr_dir / "asr.json"
        asr_result = run_whisper_cpp(
            whisper_cpp=whisper_cpp,
            model_path=whisper_model,
            input_wav=asr_input,
            output_json=asr_json,
        )
        # Expect whisper.cpp JSON output with a top-level "transcription" list.
        transcription = asr_result.get("transcription") if isinstance(asr_result, dict) else None
        if not isinstance(transcription, list):
            keys = list(asr_result.keys()) if isinstance(asr_result, dict) else type(asr_result).__name__
            raise ValueError(
                "Unexpected whisper.cpp JSON schema (expected top-level 'transcription'). "
                f"See: {asr_json} (top-level keys: {keys})"
            )
        transcript_text = " ".join(
            str(seg.get("text", "")).strip() for seg in transcription if isinstance(seg, dict)
        ).strip()
        if not transcript_text:
            raise ValueError(f"whisper.cpp JSON 'transcription' contained no text. See: {asr_json}")
        script_text = transcript_text

    # 3) Kana conversion (pykakasi) + prepare MFA corpus
    corpus_dir = paths.alignment_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    all_kana_tokens: list[str] = []
    lrc_kana_spaced: list[str] | None = None
    lrc_script_units = None
    asr_script_units = None
    if lrc_lines:
        # Split audio + transcripts by LRC line times to make alignment feasible.
        kana_lines = []
        lrc_kana_spaced = kana_lines
        lrc_script_units = []
        for idx, line in enumerate(lrc_lines):
            utt_id = f"utt_{idx:04d}"
            out_wav = corpus_dir / f"{utt_id}.wav"
            out_lab = corpus_dir / f"{utt_id}.lab"

            kana_spaced, units = to_spaced_kana_with_units(line.text, output=kana_output)
            out_lab.write_text(kana_spaced + "\n", encoding="utf-8")
            kana_lines.append(kana_spaced)
            lrc_script_units.append(units)
            all_kana_tokens.extend(kana_spaced.split())

            from karaoker.external.ffmpeg import cut_wav_segment

            cut_wav_segment(
                ffmpeg=ffmpeg,
                input_wav=asr_input,
                start_seconds=line.start,
                end_seconds=line.end,
                output_wav=out_wav,
            )
        # Also write a convenience joined transcript.
        (paths.transcript_dir / "kana_spaced.txt").write_text("\n".join(kana_lines) + "\n", encoding="utf-8")
        (paths.transcript_dir / "script.txt").write_text(
            "\n".join(x.text for x in lrc_lines) + "\n",
            encoding="utf-8",
        )
    else:
        # Single-utterance transcript (ASR path).
        kana_spaced, asr_script_units = to_spaced_kana_with_units(script_text or "", output=kana_output)
        kana_spaced_path = paths.transcript_dir / "kana_spaced.txt"
        kana_spaced_path.write_text(kana_spaced + "\n", encoding="utf-8")
        (paths.transcript_dir / "script.txt").write_text((script_text or "") + "\n", encoding="utf-8")
        all_kana_tokens = kana_spaced.split()

    # 4) MFA forced alignment (TextGrid)
    # If no dictionary is provided, generate one from the kana tokens using an MFA G2P model.
    mfa_dict_to_use: str | None = mfa_dict
    if mfa_dict_to_use is None:
        tokens = sorted(set(all_kana_tokens))
        words_path = paths.alignment_dir / "words.txt"
        words_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

        g2p_model = "japanese_katakana_mfa" if kana_output == "katakana" else "japanese_mfa"
        gen_dict = paths.alignment_dir / "g2p.dict"
        run_mfa_g2p(mfa=mfa, word_list=words_path, g2p_model=g2p_model, output_dictionary=gen_dict)
        mfa_dict_to_use = str(gen_dict)

    script_units_events: list[dict[str, object]] | None = None
    if lrc_lines:
        out_dir = paths.alignment_dir / "textgrids"
        run_mfa_align_corpus(
            mfa=mfa,
            corpus_dir=corpus_dir,
            pronunciation_dict=mfa_dict_to_use,
            acoustic_model=mfa_acoustic_model,
            output_dir=out_dir,
        )
        # Merge TextGrids into a single event list by applying per-line offsets.
        events: list[dict[str, object]] = []
        script_units_events = []
        for idx, line in enumerate(lrc_lines):
            tg = out_dir / f"utt_{idx:04d}.TextGrid"
            if not tg.exists():
                continue

            line_events = textgrid_to_kana_events(tg, offset_seconds=line.start)
            # Preserve per-line indices before global reindexing.
            for e in line_events:
                e["line_i"] = idx
                e["line_event_i"] = e.get("i")

            units = lrc_script_units[idx] if lrc_script_units is not None else []
            mapped_line_events, timed_units = map_kana_events_to_script(
                script_text=line.text,
                script_units=units,
                kana_events=line_events,
            )
            for e in mapped_line_events:
                e["line_i"] = idx
            for u in timed_units:
                u["line_i"] = idx

            events.extend(mapped_line_events)
            script_units_events.extend(timed_units)
        # Re-index events
        events.sort(key=lambda e: float(e["start"]))  # type: ignore[index]
        for i, e in enumerate(events):
            e["i"] = i
    else:
        textgrid_path = paths.alignment_dir / "aligned.TextGrid"
        run_mfa_align(
            mfa=mfa,
            input_wav=asr_input,
            transcript_spaced_kana=kana_spaced_path,
            pronunciation_dict=mfa_dict_to_use,
            acoustic_model=mfa_acoustic_model,
            output_textgrid=textgrid_path,
        )
        events0 = textgrid_to_kana_events(textgrid_path)
        mapped_events, script_units_events0 = map_kana_events_to_script(
            script_text=script_text or "",
            script_units=asr_script_units or [],
            kana_events=events0,
        )
        events = mapped_events
        script_units_events = script_units_events0

    # 5) Parse TextGrid -> JSON
    source_meta: dict[str, object]
    if lrc_lines:
        source_meta = {"type": "lrc", "path": str(lyrics_lrc), "num_lines": len(lrc_lines)}
    else:
        source_meta = {"type": "asr", "path": str(paths.asr_dir / "asr.json")}

    out: dict[str, object] = {
        "version": 2,
        "language": "ja",
        "kana_output": kana_output,
        "units": "kana",
        "source": source_meta,
        "events": events,
    }
    if lrc_lines:
        out["lines"] = [
            {
                "i": i,
                "start": round(float(x.start), 4),
                "end": round(float(x.end), 4),
                "text": x.text,
                "ref_kana": (lrc_kana_spaced[i] if lrc_kana_spaced is not None else ""),
            }
            for i, x in enumerate(lrc_lines)
        ]
    else:
        out["script"] = {"text": script_text or "", "ref_kana": kana_spaced}

    if script_units_events is not None:
        out["script_units"] = script_units_events
    (paths.output_dir / "subtitles.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
