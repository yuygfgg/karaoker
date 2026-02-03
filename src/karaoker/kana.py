from __future__ import annotations

import re

from pykakasi import kakasi


_RE_JP_SPACE = re.compile(r"\s+")
_RE_NON_KANA = re.compile(r"[^\u3040-\u30ff\u30fc ]+")  # keep kana + prolonged mark + space


def _kakasi_converter(output: str):
    """
    Build a kakasi converter.

    output:
      - 'katakana' -> pure katakana
      - 'hiragana' -> pure hiragana
    """
    kks = kakasi()
    # Kanji -> kana
    kks.setMode("J", "K" if output == "katakana" else "H")
    # Hiragana -> kana
    kks.setMode("H", "K" if output == "katakana" else "H")
    # Katakana -> kana
    kks.setMode("K", "K" if output == "katakana" else "H")
    # Ascii -> keep
    kks.setMode("a", "a")
    kks.setMode("E", "a")
    return kks.getConverter()


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
    conv = _kakasi_converter(output)
    kana = conv.do(text)
    # Normalize spaces and remove punctuation that MFA dictionaries usually don't cover.
    kana = kana.replace("\u3000", " ")
    kana = _RE_NON_KANA.sub(" ", kana)
    kana = _RE_JP_SPACE.sub(" ", kana).strip()

    # Keep spaces as explicit boundaries, but tokenize within each chunk.
    tokens: list[str] = []
    for chunk in kana.split(" "):
        if not chunk:
            continue
        tokens.extend(kana_tokens(chunk))
    return " ".join(tokens)
