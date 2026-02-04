from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def _load_json_to_ass_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "json_to_ass.py"
    spec = importlib.util.spec_from_file_location("json_to_ass", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses (with future-annotations) may consult sys.modules during class creation.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_lrc_mode_preserves_kanji_and_punct():
    mod = _load_json_to_ass_module()

    # Text includes punctuation not covered by timed units: it should still appear in the ASS output.
    data = {
        "version": 2,
        "language": "ja",
        "units": "kana",
        "source": {"type": "lrc"},
        "lines": [
            {"i": 0, "start": 0.0, "end": 2.0, "text": "君の名は。"},
        ],
        "script_units": [
            # Pretend "君" is sung from 0.50s to 0.90s.
            {
                "i": 0,
                "line_i": 0,
                "start": 0.50,
                "end": 0.90,
                "text": "君",
                "char_start": 0,
                "char_end": 1,
            },
            # Pretend "の名は" is sung from 0.90s to 1.80s.
            {
                "i": 1,
                "line_i": 0,
                "start": 0.90,
                "end": 1.80,
                "text": "の名は",
                "char_start": 1,
                "char_end": 4,
            },
        ],
        "events": [],
    }

    ass = mod.subtitles_json_to_ass(data)
    # Kanji text is present (not kana-only). Strip override tags so the script text is contiguous.
    plain = re.sub(r"\{[^}]*\}", "", ass)
    assert "君の名は" in plain
    # Punctuation is preserved even if it has no timing unit.
    assert "。" in ass
    # Leading gap from 0.0 -> 0.50 is expressed via a \k tag.
    assert r"{\k50}" in ass


def test_asr_mode_chunks_on_gaps():
    mod = _load_json_to_ass_module()

    script_text = "一行目。二行目。"
    data = {
        "version": 2,
        "language": "ja",
        "units": "kana",
        "source": {"type": "asr"},
        "script": {"text": script_text, "ref_kana": ""},
        "script_units": [
            {
                "i": 0,
                "start": 0.0,
                "end": 0.5,
                "text": "一行目",
                "char_start": 0,
                "char_end": 3,
            },
            # A long pause should split into a new dialogue.
            {
                "i": 1,
                "start": 2.0,
                "end": 2.4,
                "text": "二行目",
                "char_start": 4,
                "char_end": 7,
            },
        ],
        "events": [],
    }

    ass = mod.subtitles_json_to_ass(data, gap_threshold_s=0.8)
    # Should produce 2 Dialogue lines.
    assert ass.count("Dialogue:") == 2
    assert "一行目" in ass
    assert "二行目" in ass
