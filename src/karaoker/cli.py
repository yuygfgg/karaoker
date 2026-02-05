from __future__ import annotations

import argparse
from pathlib import Path

from karaoker.external.ffmpeg import ensure_wav_16k_mono
from karaoker.external.mfa import run_mfa_align
from karaoker.external.whispercpp import run_whisper_cpp
from karaoker.kana import to_spaced_kana
from karaoker.pipeline import run_pipeline
from karaoker.textgrid_parser import textgrid_to_kana_events


def _default_whisper_model_path() -> str:
    # Default whisper.cpp model shipped alongside this repo. Override via --whisper-model.
    return str(
        (
            Path(__file__).resolve().parents[2]
            / "third_party/whisper.cpp/models/ggml-large-v2.bin"
        )
    )


def _default_whisper_cli_path() -> str:
    return str(
        (
            Path(__file__).resolve().parents[2]
            / "third_party/whisper.cpp/build/bin/whisper-cli"
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="karaoker")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the full local-first pipeline.")
    run.add_argument(
        "--input", required=True, help="Input song path (e.g. .flac/.mp3/.wav)."
    )
    run.add_argument(
        "--workdir", required=True, help="Working directory for intermediate outputs."
    )
    run.add_argument(
        "--lyrics-lrc",
        default=None,
        help="Optional .lrc path. If provided, skip ASR and use these lyrics as transcript.",
    )
    run.add_argument(
        "--asr-backend",
        choices=["whispercpp", "gemini"],
        default="whispercpp",
        help=(
            "ASR backend to use when --lyrics-lrc is not provided (default: %(default)s). "
            "Gemini requires GEMINI_API_KEY and `pip install -e \".[gemini]\"`."
        ),
    )
    run.add_argument(
        "--kana-backend",
        choices=["pykakasi", "gemini"],
        default="pykakasi",
        help=(
            "Kana conversion backend (default: %(default)s). "
            "Gemini requires GEMINI_API_KEY and `pip install -e \".[gemini]\"`."
        ),
    )
    run.add_argument(
        "--gemini-model",
        default="gemini-3-flash-preview",
        help=(
            "Gemini model name for gemini backends (default: %(default)s). "
            "Used by both ASR and kana conversion."
        ),
    )

    # External tool entrypoints
    run.add_argument(
        "--ffmpeg", default="ffmpeg", help="ffmpeg executable (default: ffmpeg)."
    )
    run.add_argument(
        "--audio-separator",
        default=None,
        help="Audio-separator python module/CLI (optional; see README).",
    )
    run.add_argument(
        "--audio-separator-model",
        default=None,
        help=(
            "audio-separator model filename for vocal separation "
            "(default: audio-separator CLI default)."
        ),
    )
    run.add_argument(
        "--dereverb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable a de-reverb pass on the separated vocals (default: enabled).",
    )
    run.add_argument(
        "--dereverb-model",
        default="dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
        help="audio-separator model filename for de-reverb (default: %(default)s).",
    )
    run.add_argument(
        "--silero-vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Silero VAD gating (non-speech samples set to 0) (default: enabled).",
    )
    run.add_argument(
        "--silero-vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD threshold (default: %(default)s).",
    )
    run.add_argument(
        "--silero-vad-min-speech-ms",
        type=int,
        default=250,
        help="Silero VAD min speech duration in ms (default: %(default)s).",
    )
    run.add_argument(
        "--silero-vad-min-silence-ms",
        type=int,
        default=100,
        help="Silero VAD min silence duration in ms (default: %(default)s).",
    )
    run.add_argument(
        "--silero-vad-speech-pad-ms",
        type=int,
        default=30,
        help="Silero VAD speech padding in ms (default: %(default)s).",
    )
    run.add_argument(
        "--whisper-cpp",
        default=_default_whisper_cli_path(),
        help="whisper.cpp executable (e.g. /path/to/whisper.cpp/main).",
    )
    run.add_argument(
        "--whisper-model",
        default=_default_whisper_model_path(),
        help="Path to a whisper.cpp ggml model file (default: ggml-large-v2.bin).",
    )
    run.add_argument("--mfa", default="mfa", help="Montreal Forced Aligner executable.")
    run.add_argument(
        "--mfa-dict",
        default=None,
        help=(
            "Path to MFA pronunciation dictionary OR an installed model name "
            "(default: japanese_mfa)."
        ),
    )
    run.add_argument(
        "--mfa-acoustic-model",
        default="japanese_mfa",
        help="MFA acoustic model path or installed model name (default: japanese_mfa).",
    )

    # Tuning
    run.add_argument(
        "--kana-output",
        choices=["katakana", "hiragana"],
        default="katakana",
        help="Kana output type (default: katakana).",
    )
    
    # TODO
    run.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate files (default keeps anyway; reserved for future cleanup).",
    )

    # Step commands (useful for debugging / iterative workflows)
    conv = sub.add_parser(
        "convert-wav", help="Convert input audio to 16kHz mono PCM wav (ffmpeg)."
    )
    conv.add_argument("--input", required=True)
    conv.add_argument("--output", required=True)
    conv.add_argument("--ffmpeg", default="ffmpeg")

    sep = sub.add_parser(
        "separate", help="Separate vocals via python-audio-separator wrapper."
    )
    sep.add_argument("--input-wav", required=True)
    sep.add_argument("--output-vocals", required=True)
    sep.add_argument(
        "--audio-separator",
        required=True,
        help="Audio-separator CLI or 'python -m ...' command string.",
    )

    asr = sub.add_parser("asr", help="Run whisper.cpp ASR and produce JSON.")
    asr.add_argument("--input-wav", required=True)
    asr.add_argument("--output-json", required=True)
    asr.add_argument("--whisper-cpp", default="whisper-cpp")
    asr.add_argument("--whisper-model", required=True)

    kana = sub.add_parser("kana", help="Convert text to spaced kana (pykakasi).")
    kana.add_argument("--text", required=True, help="Input text (Japanese).")
    kana.add_argument(
        "--output", required=True, help="Output .txt path for spaced kana."
    )
    kana.add_argument(
        "--kana-output", choices=["katakana", "hiragana"], default="katakana"
    )

    align = sub.add_parser("align", help="Run MFA align for one wav + transcript.")
    align.add_argument("--input-wav", required=True)
    align.add_argument(
        "--transcript", required=True, help="Spaced kana transcript (.txt)."
    )
    align.add_argument("--output-textgrid", required=True)
    align.add_argument("--mfa", default="mfa")
    align.add_argument("--mfa-dict", required=True)
    align.add_argument("--mfa-acoustic-model", required=True)

    export = sub.add_parser("export", help="Convert TextGrid to JSON events.")
    export.add_argument("--textgrid", required=True)
    export.add_argument("--output-json", required=True)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "run":
        lyrics_lrc = None
        if args.lyrics_lrc:
            lyrics_lrc = Path(args.lyrics_lrc)

        run_pipeline(
            input_path=Path(args.input),
            workdir=Path(args.workdir),
            ffmpeg=args.ffmpeg,
            audio_separator=args.audio_separator,
            audio_separator_model=args.audio_separator_model,
            enable_dereverb=args.dereverb,
            dereverb_model=args.dereverb_model,
            enable_silero_vad=args.silero_vad,
            silero_vad_threshold=args.silero_vad_threshold,
            silero_vad_min_speech_ms=args.silero_vad_min_speech_ms,
            silero_vad_min_silence_ms=args.silero_vad_min_silence_ms,
            silero_vad_speech_pad_ms=args.silero_vad_speech_pad_ms,
            whisper_cpp=args.whisper_cpp,
            whisper_model=Path(args.whisper_model),
            mfa=args.mfa,
            mfa_dict=args.mfa_dict,
            mfa_acoustic_model=args.mfa_acoustic_model,
            kana_output=args.kana_output,
            lyrics_lrc=lyrics_lrc,
            asr_backend=args.asr_backend,
            kana_backend=args.kana_backend,
            gemini_model=args.gemini_model,
        )
        return 0
    if args.cmd == "convert-wav":
        ensure_wav_16k_mono(
            ffmpeg=args.ffmpeg,
            input_audio=Path(args.input),
            output_wav=Path(args.output),
        )
        return 0
    if args.cmd == "separate":
        from karaoker.external.audio_separator import run_audio_separator

        run_audio_separator(
            audio_separator=args.audio_separator,
            input_audio=Path(args.input_wav),
            output_audio=Path(args.output_vocals),
        )
        return 0
    if args.cmd == "asr":
        run_whisper_cpp(
            whisper_cpp=args.whisper_cpp,
            model_path=Path(args.whisper_model),
            input_wav=Path(args.input_wav),
            output_json=Path(args.output_json),
        )
        return 0
    if args.cmd == "kana":
        out = to_spaced_kana(args.text, output=args.kana_output)
        Path(args.output).write_text(out + "\n", encoding="utf-8")
        return 0
    if args.cmd == "align":
        run_mfa_align(
            mfa=args.mfa,
            input_wav=Path(args.input_wav),
            transcript_spaced_kana=Path(args.transcript),
            pronunciation_dict=Path(args.mfa_dict),
            acoustic_model=args.mfa_acoustic_model,
            output_textgrid=Path(args.output_textgrid),
        )
        return 0
    if args.cmd == "export":
        import json

        events = textgrid_to_kana_events(Path(args.textgrid))
        Path(args.output_json).write_text(
            json.dumps(
                {"version": 1, "language": "ja", "units": "kana", "events": events},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    raise AssertionError(f"unhandled cmd: {args.cmd}")
