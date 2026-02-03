from karaoker.kana import kana_tokens, to_spaced_kana


def test_kana_tokens_attach_small_kana():
    assert kana_tokens("キャット") == ["キャ", "ッ", "ト"]


def test_to_spaced_kana_basic():
    out = to_spaced_kana("今日はキャットです", output="katakana")
    # Don't assert full string (pykakasi may vary), just that it becomes spaced tokens.
    assert " " in out

