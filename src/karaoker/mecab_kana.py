from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal

KanaOutput = Literal["katakana", "hiragana"]

_RE_JP_SPACE = re.compile(r"\s+")
_RE_NON_KANA = re.compile(r"[^\u3040-\u30ff\u30fc ]+")  # keep kana + prolonged mark + space
_RE_ALL_KANA = re.compile(r"^[\u3040-\u30ff\u30fc]+$")


def _kata_to_hira(s: str) -> str:
    # Katakana [ァ..ヶ] -> Hiragana [ぁ..ゖ] via fixed offset.
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _hira_to_kata(s: str) -> str:
    # Hiragana [ぁ..ゖ] -> Katakana [ァ..ヶ] via fixed offset.
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if 0x3041 <= o <= 0x3096:
            out.append(chr(o + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _clean_kana(s: str) -> str:
    # Normalize spaces and drop punctuation/symbols that MFA dictionaries usually don't cover.
    s = s.replace("\u3000", " ")
    s = _RE_NON_KANA.sub(" ", s)
    s = _RE_JP_SPACE.sub(" ", s).strip()
    return s


def _looks_like_kana(s: str) -> bool:
    return bool(s) and bool(_RE_ALL_KANA.fullmatch(s))


@dataclass(frozen=True)
class MecabToken:
    surface: str
    char_start: int
    char_end: int
    reading: str
    pos1: str | None = None
    pos2: str | None = None


@lru_cache(maxsize=1)
def _get_mecab_tagger():
    try:
        import MeCab  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - depends on optional runtime deps
        raise ModuleNotFoundError(
            "MeCab is required for kana conversion. "
            "Install `mecab-python3` and `unidic-lite`."
        ) from e

    args = ""
    try:
        import unidic_lite  # type: ignore[import-not-found]

        dicdir = unidic_lite.DICDIR
        # Prefer the packaged mecabrc if present; otherwise isolate from system config.
        mecabrc = os.path.join(dicdir, "mecabrc")
        null_rc = "nul" if os.name == "nt" else "/dev/null"
        rc = mecabrc if os.path.exists(mecabrc) else null_rc
        args = f"-r {rc} -d {dicdir}"
    except Exception:
        # Fall back to the system default dictionary/config.
        args = ""

    tagger = MeCab.Tagger(args)
    # Workaround for a long-standing MeCab python binding quirk: ensure internal
    # state is initialized before parseToNode().
    tagger.parse("")
    return tagger


def _extract_reading_kata(surface: str, feature: str) -> tuple[str, str | None, str | None]:
    """
    Return (reading_katakana, pos1, pos2) for this token.

    We prefer UniDic "kana" (orthographic reading) to avoid long-vowel marker 'ー' in `pron`.
    For IPA-DIC we fall back to the standard reading/pron fields.
    """
    fields = feature.split(",") if feature else []
    pos1 = fields[0] if fields else None
    pos2 = fields[1] if len(fields) > 1 else None

    # Normalize surface for particle handling (hiragana).
    surf_h = _kata_to_hira(surface)

    # UniDic provides both spelling ("kana") and pronunciation ("pron"). For alignment/MFA
    # the spelling form is usually safer (dictionary coverage), but we special-case the
    # classic particles which are pronounced differently.
    if pos1 == "助詞":
        if surf_h == "は":
            return "ワ", pos1, pos2
        if surf_h == "へ":
            return "エ", pos1, pos2
        if surf_h == "を":
            return "オ", pos1, pos2

    # Candidate indices by common dictionaries:
    # - UniDic: kana at index 20, pron at index 9
    # - IPA-DIC: reading at index 7, pron at index 8
    candidates: list[str] = []
    for idx in (20, 7, 8, 9):
        if idx >= len(fields):
            continue
        v = fields[idx]
        if not v or v == "*":
            continue
        if _looks_like_kana(v):
            candidates.append(v)

    if candidates:
        return _hira_to_kata(candidates[0]), pos1, pos2

    # Fallback: if surface itself is kana, keep it.
    if _looks_like_kana(surface):
        return _hira_to_kata(surface), pos1, pos2

    return "", pos1, pos2


def mecab_tokenize(text: str, *, output: KanaOutput = "katakana") -> list[MecabToken]:
    """
    Tokenize Japanese text using MeCab and return per-token readings + character spans.
    """
    if output not in ("katakana", "hiragana"):
        raise ValueError(f"Unsupported output={output!r} (expected katakana/hiragana).")

    tagger = _get_mecab_tagger()
    node = tagger.parseToNode(text)

    out: list[MecabToken] = []
    pos = 0

    while node is not None:
        surface = getattr(node, "surface", "") or ""
        feature = getattr(node, "feature", "") or ""
        node = getattr(node, "next", None)

        if not surface:
            continue

        # Best-effort offset mapping: advance monotonically through the original text.
        start = text.find(surface, pos)
        if start == -1:
            start = pos
        end = start + len(surface)
        pos = end

        reading_kata, pos1, pos2 = _extract_reading_kata(surface, feature)
        reading = _clean_kana(reading_kata)
        if output == "hiragana":
            reading = _kata_to_hira(reading)

        out.append(
            MecabToken(
                surface=surface,
                char_start=start,
                char_end=end,
                reading=reading,
                pos1=pos1,
                pos2=pos2,
            )
        )

    return out


def mecab_readings(text: str, *, output: KanaOutput = "katakana") -> Iterable[str]:
    for tok in mecab_tokenize(text, output=output):
        if tok.reading:
            yield tok.reading

