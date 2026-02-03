# karaoker

Local-first Japanese karaoke subtitle generator that produces per-kana (roughly per-mora) timing JSON.

It integrates with:

1. `python-audio-separator` to extract vocals from a song (de-instrumentalize).
2. `whisper.cpp` for offline ASR.
3. `pykakasi` to convert recognized text to pure kana and insert spaces between kana tokens.
4. Montreal Forced Aligner (MFA) to force-align the spaced kana transcript to audio and produce a Praat
   `.TextGrid`.
5. TextGrid parsing to generate the final JSON subtitle events.

## Status / Scope

- Language: Japanese only.
- Subtitle granularity: kana tokens (katakana by default).

## Requirements (External Tools)

- `ffmpeg` (audio decoding/conversion).
- `whisper.cpp` (ASR). You need a model file (e.g. `ggml-large-v3.bin`).
- Montreal Forced Aligner (MFA) (alignment).
- `python-audio-separator` (vocal separation).

Notes:
- MFA requires a Japanese acoustic model + a pronunciation dictionary compatible with that model.

## Install

```bash
conda create -p ./.conda/env -y python=3.11 pip
conda activate ./.conda/env
pip install -e ".[dev,textgrid]"
```

If you prefer a one-shot bootstrap (conda + MFA + whisper.cpp + audio-separator), see:

```bash
./scripts/bootstrap_local.sh
```

For day-to-day usage, use:

```bash
./scripts/karaoker_run.sh --help
```

## whisper.cpp Model Default

If you run without `--lrc` (i.e. you use ASR), the default whisper.cpp model is:

- `./third_party/whisper.cpp/models/ggml-large-v3.bin` (large-v3)

Download it with whisper.cpp's helper script (or just run `./scripts/karaoker_run.sh` without `--lrc`
and it will auto-download on demand):

```bash
./third_party/whisper.cpp/models/download-ggml-model.sh large-v3
```

## Quickstart (Pipeline)

Example (paths are illustrative):

```bash
karaoker run \
  --input "/path/to/song.flac" \
  --workdir "./runs/symbol3" \
  --whisper-cpp "./third_party/whisper.cpp/build/bin/main" \
  --whisper-model "/path/to/ggml-model.bin" \
  --mfa "./.conda/env/bin/mfa" \
  --mfa-dict "/path/to/ja.dict" \
  --mfa-acoustic-model "/path/to/ja_acoustic_model.zip"
```

Outputs:
- `workdir/audio/vocals.wav`
- `workdir/asr/asr.json`
- `workdir/transcript/kana_spaced.txt`
- `workdir/alignment/aligned.TextGrid`
- `workdir/output/subtitles.json`

## Step-By-Step Commands

For debugging, you can run individual steps:

```bash
karaoker convert-wav --input in.flac --output song.wav
karaoker separate --input-wav song.wav --output-vocals vocals.wav --audio-separator "python -m audio_separator"
karaoker asr --input-wav vocals.wav --output-json asr.json --whisper-model /path/to/model.bin
karaoker kana --text "..." --output kana_spaced.txt
karaoker align --input-wav vocals.wav --transcript kana_spaced.txt --output-textgrid aligned.TextGrid --mfa-dict ja.dict --mfa-acoustic-model ja.zip
karaoker export --textgrid aligned.TextGrid --output-json subtitles.json
```

## JSON Output Schema

`subtitles.json`:

```json
{
  "version": 1,
  "language": "ja",
  "units": "kana",
  "events": [
    { "i": 0, "start": 1.23, "end": 1.45, "text": "サ" },
    { "i": 1, "start": 1.45, "end": 1.60, "text": "ン" }
  ]
}
```

## Development

```bash
pytest -q
ruff check .
```
