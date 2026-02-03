from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from karaoker.utils import run_checked


def run_whisper_cpp(
    *,
    whisper_cpp: str,
    model_path: Path,
    input_wav: Path,
    output_json: Path,
) -> dict[str, Any]:
    """
    Run whisper.cpp (whisper-cli) with JSON output and return the parsed result.

    Expects whisper.cpp to write `<output-prefix>.json` when invoked with
    `--output-json --output-file <output-prefix>`.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)

    out_prefix = output_json.with_suffix("")
    cmd = [
        whisper_cpp,
        "--model",
        str(model_path),
        "--file",
        str(input_wav),
        "--output-json",
        "--output-file",
        str(out_prefix),
        "--language",
        "ja",
    ]
    run_checked(cmd)

    produced = out_prefix.with_suffix(".json")
    if produced != output_json:
        output_json.write_bytes(produced.read_bytes())
    return json.loads(output_json.read_text(encoding="utf-8"))
