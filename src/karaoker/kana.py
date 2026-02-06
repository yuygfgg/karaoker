from __future__ import annotations

import re

from karaoker.mecab_kana import mecab_readings

_RE_JP_SPACE = re.compile(r"\s+")


def kana_tokens(kana: str) -> list[str]:
    """
    Split kana into "rough mora" tokens.

    Heuristics (good enough for karaoke-style display):
    - small kana (ャュョァィゥェォ) attach to the previous token
    - sokuon (ッ/っ) attaches to the next token is more correct, but for display we attach to previous
    - prolonged sound mark (ー) attaches to previous token
    """
    kana = _RE_JP_SPACE.sub("", kana)
    if not kana:
        return []

    small = set("ャュョァィゥェォゃゅょぁぃぅぇぉ")
    attach_prev = set("ーッっ")

    out: list[str] = []
    for ch in kana:
        if not out:
            out.append(ch)
            continue

        if ch in small or ch in attach_prev:
            out[-1] = out[-1] + ch
        else:
            out.append(ch)
    return out


def to_spaced_kana(text: str, *, output: str = "katakana") -> str:
    """
    Convert Japanese text to pure kana, then insert spaces between kana tokens.
    Non-kana characters are dropped except ASCII spaces (used as word boundaries).
    """
    # `mecab_readings` already returns cleaned kana (hiragana/katakana depending on `output`).
    tokens: list[str] = []
    for reading in mecab_readings(text, output=output):  # type: ignore[arg-type]
        tokens.extend(kana_tokens(reading))
    return " ".join(tokens)
