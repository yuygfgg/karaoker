from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from karaoker.external.ffmpeg import ensure_wav_16k_mono
from karaoker.external.mfa import run_mfa_align, run_mfa_align_corpus, run_mfa_g2p
from karaoker.external.whispercpp import run_whisper_cpp
from karaoker.kana import to_spaced_kana
from karaoker.lyrics import parse_lrc
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

        vocals_raw = paths.audio_dir / "vocals_raw.wav"
        run_audio_separator(
            audio_separator=audio_separator,
            input_audio=input_path,
            output_vocals=vocals_raw,
            # Cache models under the repo so repeated runs reuse downloads.
            model_file_dir=_project_root() / "models" / "audio_separator",
        )
        ensure_wav_16k_mono(ffmpeg=ffmpeg, input_audio=vocals_raw, output_wav=vocals_wav)
        asr_input = vocals_wav
    else:
        asr_input = song_wav

    # 2) Transcript source: LRC lyrics (preferred) OR ASR (whisper.cpp)
    lrc_lines = None
    if lyrics_lrc is not None:
        lrc_lines = parse_lrc(lyrics_lrc)

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

    # 3) Kana conversion (pykakasi) + prepare MFA corpus
    corpus_dir = paths.alignment_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    all_kana_tokens: list[str] = []
    if lrc_lines:
        # Split audio + transcripts by LRC line times to make alignment feasible.
        kana_lines: list[str] = []
        for idx, line in enumerate(lrc_lines):
            utt_id = f"utt_{idx:04d}"
            out_wav = corpus_dir / f"{utt_id}.wav"
            out_lab = corpus_dir / f"{utt_id}.lab"

            kana_spaced = to_spaced_kana(line.text, output=kana_output)
            out_lab.write_text(kana_spaced + "\n", encoding="utf-8")
            kana_lines.append(kana_spaced)
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
    else:
        # Single-utterance transcript (ASR path).
        kana_spaced = to_spaced_kana(transcript_text, output=kana_output)
        kana_spaced_path = paths.transcript_dir / "kana_spaced.txt"
        kana_spaced_path.write_text(kana_spaced + "\n", encoding="utf-8")
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
        for idx, line in enumerate(lrc_lines):
            tg = out_dir / f"utt_{idx:04d}.TextGrid"
            if not tg.exists():
                continue
            events.extend(textgrid_to_kana_events(tg, offset_seconds=line.start))
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
        events = textgrid_to_kana_events(textgrid_path)

    # 5) Parse TextGrid -> JSON
    out = {
        "version": 1,
        "language": "ja",
        "units": "kana",
        "events": events,
    }
    (paths.output_dir / "subtitles.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
