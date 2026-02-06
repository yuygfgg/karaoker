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


def test_asr_mode_uses_per_kana_events_when_unit_text_matches_reading():
    mod = _load_json_to_ass_module()

    # "せる" is pure hiragana and matches the unit reading "セル", so we can safely
    # map per-kana timing onto the displayed script text.
    data = {
        "version": 2,
        "language": "ja",
        "units": "kana",
        "source": {"type": "asr"},
        "script": {"text": "せる夢", "ref_kana": "セ ル ユ メ"},
        "script_units": [
            {
                "i": 0,
                "start": 0.0,
                "end": 0.10,
                "text": "せる",
                "char_start": 0,
                "char_end": 2,
                "reading": "セル",
                "ref_kana_start": 0,
                "ref_kana_end": 2,
            },
            # "夢" (kanji) does not match its reading 1:1, so it should remain a single unit.
            {
                "i": 1,
                "start": 0.10,
                "end": 0.30,
                "text": "夢",
                "char_start": 2,
                "char_end": 3,
                "reading": "ユメ",
                "ref_kana_start": 2,
                "ref_kana_end": 4,
            },
        ],
        "events": [
            {"i": 0, "start": 0.00, "end": 0.05, "text": "セ", "tier": "words", "ref_kana_i": 0},
            {"i": 1, "start": 0.05, "end": 0.10, "text": "ル", "tier": "words", "ref_kana_i": 1},
            {"i": 2, "start": 0.10, "end": 0.20, "text": "ユ", "tier": "words", "ref_kana_i": 2},
            {"i": 3, "start": 0.20, "end": 0.30, "text": "メ", "tier": "words", "ref_kana_i": 3},
        ],
    }

    ass = mod.subtitles_json_to_ass(data, gap_threshold_s=5.0, max_caption_s=20.0)
    assert r"{\k5}せ{\k5}る" in ass
    assert r"{\k20}夢" in ass


def test_asr_mode_uses_segment_local_ref_kana_indices():
    mod = _load_json_to_ass_module()

    # Segment-local `ref_kana_i` restarts from 0 for each segment, so the converter must
    # key events by (segment_i, ref_kana_i) instead of just ref_kana_i.
    data = {
        "version": 2,
        "language": "ja",
        "units": "kana",
        "source": {"type": "asr"},
        # Intentionally wrong global ref_kana so we only succeed if we use segments' ref_kana.
        "script": {"text": "せるせる", "ref_kana": "ガ ツ"},
        "segments": [
            {"i": 0, "start": 0.0, "end": 0.1, "text": "せる", "ref_kana": "セ ル"},
            {"i": 1, "start": 1.0, "end": 1.1, "text": "せる", "ref_kana": "セ ル"},
        ],
        "script_units": [
            {
                "i": 0,
                "segment_i": 0,
                "start": 0.0,
                "end": 0.10,
                "text": "せる",
                "char_start": 0,
                "char_end": 2,
                "reading": "セル",
                "ref_kana_start": 0,
                "ref_kana_end": 2,
            },
            {
                "i": 1,
                "segment_i": 1,
                "start": 1.0,
                "end": 1.10,
                "text": "せる",
                "char_start": 2,
                "char_end": 4,
                "reading": "セル",
                "ref_kana_start": 0,
                "ref_kana_end": 2,
            },
        ],
        "events": [
            {
                "i": 0,
                "segment_i": 0,
                "ref_kana_i": 0,
                "start": 0.00,
                "end": 0.05,
                "text": "セ",
                "tier": "words",
            },
            {
                "i": 1,
                "segment_i": 0,
                "ref_kana_i": 1,
                "start": 0.05,
                "end": 0.10,
                "text": "ル",
                "tier": "words",
            },
            {
                "i": 2,
                "segment_i": 1,
                "ref_kana_i": 0,
                "start": 1.00,
                "end": 1.05,
                "text": "セ",
                "tier": "words",
            },
            {
                "i": 3,
                "segment_i": 1,
                "ref_kana_i": 1,
                "start": 1.05,
                "end": 1.10,
                "text": "ル",
                "tier": "words",
            },
        ],
    }

    # Keep as a single caption so we can assert on the inserted gap.
    ass = mod.subtitles_json_to_ass(data, gap_threshold_s=5.0, max_caption_s=20.0)
    assert r"{\k5}せ{\k5}る" in ass
    # Gap from 0.10 -> 1.00 is 0.90s = 90cs.
    assert r"{\k90}" in ass


def test_asr_mode_uses_per_kana_events_for_mixed_kanji_kana_units():
    mod = _load_json_to_ass_module()

    # Mixed script: kanji + kana. We should still be able to use per-kana token timings
    # by distributing tokens over visible glyphs.
    data = {
        "version": 2,
        "language": "ja",
        "units": "kana",
        "source": {"type": "asr"},
        "script": {"text": "月を読む", "ref_kana": "ツ キ オ ヨ ム"},
        "script_units": [
            {
                "i": 0,
                "start": 0.0,
                "end": 0.50,
                "text": "月を読む",
                "char_start": 0,
                "char_end": 4,
                "reading": "ツ キ オ ヨ ム",
                "ref_kana_start": 0,
                "ref_kana_end": 5,
            }
        ],
        "events": [
            {"i": 0, "start": 0.00, "end": 0.05, "text": "ツ", "tier": "words", "ref_kana_i": 0},
            {"i": 1, "start": 0.05, "end": 0.10, "text": "キ", "tier": "words", "ref_kana_i": 1},
            {"i": 2, "start": 0.10, "end": 0.20, "text": "オ", "tier": "words", "ref_kana_i": 2},
            {"i": 3, "start": 0.20, "end": 0.30, "text": "ヨ", "tier": "words", "ref_kana_i": 3},
            {"i": 4, "start": 0.30, "end": 0.50, "text": "ム", "tier": "words", "ref_kana_i": 4},
        ],
    }

    ass = mod.subtitles_json_to_ass(data, gap_threshold_s=5.0, max_caption_s=20.0)
    assert r"{\k10}月{\k10}を{\k10}読{\k20}む" in ass


def test_lrc_mode_uses_per_kana_events_for_mixed_kanji_kana_units():
    mod = _load_json_to_ass_module()

    # LRC mode should also distribute per-kana timings over visible glyphs (DP segmentation),
    # rather than only highlighting at the script_unit granularity.
    data = {
        "version": 2,
        "language": "ja",
        "units": "kana",
        "source": {"type": "lrc"},
        "lines": [
            {
                "i": 0,
                "start": 0.0,
                "end": 0.5,
                "text": "月を読む",
                "ref_kana": "ツ キ オ ヨ ム",
            },
        ],
        "script_units": [
            {
                "i": 0,
                "line_i": 0,
                "start": 0.0,
                "end": 0.50,
                "text": "月を読む",
                "char_start": 0,
                "char_end": 4,
                "reading": "ツ キ オ ヨ ム",
                "ref_kana_start": 0,
                "ref_kana_end": 5,
            }
        ],
        "events": [
            {
                "i": 0,
                "line_i": 0,
                "ref_kana_i": 0,
                "start": 0.00,
                "end": 0.05,
                "text": "ツ",
                "tier": "words",
            },
            {
                "i": 1,
                "line_i": 0,
                "ref_kana_i": 1,
                "start": 0.05,
                "end": 0.10,
                "text": "キ",
                "tier": "words",
            },
            {
                "i": 2,
                "line_i": 0,
                "ref_kana_i": 2,
                "start": 0.10,
                "end": 0.20,
                "text": "オ",
                "tier": "words",
            },
            {
                "i": 3,
                "line_i": 0,
                "ref_kana_i": 3,
                "start": 0.20,
                "end": 0.30,
                "text": "ヨ",
                "tier": "words",
            },
            {
                "i": 4,
                "line_i": 0,
                "ref_kana_i": 4,
                "start": 0.30,
                "end": 0.50,
                "text": "ム",
                "tier": "words",
            },
        ],
    }

    ass = mod.subtitles_json_to_ass(data)
    assert r"{\k10}月{\k10}を{\k10}読{\k20}む" in ass
