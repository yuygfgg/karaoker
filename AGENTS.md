# Repository Guidelines

## Project Structure & Module Organization

- `src/karaoker/`: Python package (CLI + pipeline).
  - `cli.py`: `karaoker` entrypoint and subcommands.
  - `pipeline.py`: end-to-end orchestration.
  - `external/`: thin wrappers around `ffmpeg`, `whisper.cpp`, MFA, and audio-separator.
- `tests/`: `pytest` suite (`test_*.py`).
- `scripts/`: local helper scripts (`bootstrap_local.sh`, `karaoker_run.sh`).
- `third_party/`: vendored external tools (e.g., `whisper.cpp/`, `python-audio-separator/`).
- `runs/`, `models/`, `.mfa/`, `.conda/`: local state and sample outputs; avoid committing large artifacts.

## Build, Test, and Development Commands

- `./scripts/bootstrap_local.sh`: create `./.conda/env`, install MFA, and clone/build tools under `third_party/`.
- `pip install -e ".[dev,textgrid]"`: editable install (expects an active Python 3.11 conda env).
- `karaoker --help`: show CLI usage once installed.
- `./scripts/karaoker_run.sh --input song.flac --workdir runs/demo --lrc lyrics.lrc`: run the pipeline (use `--lrc` to skip ASR).
- `pytest -q`: run tests.
- `ruff check .`: run lint checks (fast; use before pushing).

## Coding Style & Naming Conventions

- Python: 4-space indentation, standard library `pathlib`, and type hints where practical.
- Linting: Ruff with `line-length = 100` (see `pyproject.toml`).
- Names: modules/functions in `snake_case`, classes in `PascalCase`.
- Keep external tool interactions inside `src/karaoker/external/` (don't shell out directly from unrelated modules).

## Testing Guidelines

- Framework: `pytest` (configured via `pytest.ini`).
- Conventions: add tests under `tests/` named `test_<area>.py`; keep fixtures small (no large audio/model files).

## Commit & Pull Request Guidelines

- Commit subjects in history are short, imperative phrases (e.g., "Map kana back to text"). Follow that style.
- PRs: describe behavior changes, include repro commands, and list what you ran (at least `pytest -q` and `ruff check .`).
  If output changes, attach a small JSON sample (e.g., `runs/<name>/output/subtitles.json`) and avoid committing media.
