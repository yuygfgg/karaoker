from __future__ import annotations

import pytest

from karaoker.external.gemini import (
    normalize_spaced_kana,
    parse_gemini_asr_kana_response,
)


def test_normalize_spaced_kana_merges_small_kana() -> None:
    # Model might output spaces between every character; we normalize to mora-ish tokens.
    assert normalize_spaced_kana("き ゃ く", output="hiragana") == "きゃ く"


def test_normalize_spaced_kana_strips_noise_and_converts_script() -> None:
    assert normalize_spaced_kana("カ ラ！ オ ケ??", output="katakana") == "カ ラ オ ケ"
    assert normalize_spaced_kana("カラオケ", output="hiragana") == "か ら お け"


def test_parse_gemini_asr_kana_block_format() -> None:
    text = "\n".join(
        [
            "1",
            "00:00:01.000 --> 00:00:02.500",
            "風吹けば",
            "カ ゼ フ ケ バ",
            "",
            "2",
            "2.5 --> 5.0",
            "世界へ",
            "セ カ イ エ",
            "",
        ]
    )
    segs = parse_gemini_asr_kana_response(text, kana_output="katakana")
    assert [s.i for s in segs] == [1, 2]
    assert segs[0].start == pytest.approx(1.0)
    assert segs[0].end == pytest.approx(2.5)
    assert segs[0].text == "風吹けば"
    assert segs[0].kana == "カ ゼ フ ケ バ"
    assert segs[1].start == pytest.approx(2.5)
    assert segs[1].end == pytest.approx(5.0)
    assert segs[1].kana == "セ カ イ エ"


def test_parse_gemini_asr_kana_strips_code_fences() -> None:
    fenced = "```\\n1\\n0 --> 1\\nテスト\\nテ ス ト\\n```"
    segs = parse_gemini_asr_kana_response(fenced, kana_output="katakana")
    assert len(segs) == 1
    assert segs[0].text == "テスト"
    assert segs[0].kana == "テ ス ト"
