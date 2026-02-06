from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal


SofaDictionaryKind = Literal["kana", "romaji"]


_SMALL_Y = {
    "ゃ": "ya",
    "ゅ": "yu",
    "ょ": "yo",
    "ャ": "ya",
    "ュ": "yu",
    "ョ": "yo",
}
_SMALL_VOWEL = {
    "ぁ": "a",
    "ぃ": "i",
    "ぅ": "u",
    "ぇ": "e",
    "ぉ": "o",
    "ァ": "a",
    "ィ": "i",
    "ゥ": "u",
    "ェ": "e",
    "ォ": "o",
}
_SMALL = set(_SMALL_Y) | set(_SMALL_VOWEL)
_SOKUON = {"っ", "ッ"}
_LONG = "ー"
_VOWELS = {"a", "i", "u", "e", "o"}


def detect_sofa_dictionary_kind(path: Path) -> SofaDictionaryKind:
    """
    Detect whether a SOFA dictionary is keyed by kana or by romaji.

    We treat "kana" as: the left-hand token contains any hiragana/katakana/prolonged-mark chars.
    """
    path = path.expanduser().resolve()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0]
        if _contains_kana(key):
            return "kana"
        return "romaji"
    raise ValueError(f"SOFA dictionary appears empty: {path}")


def load_sofa_dictionary(path: Path) -> dict[str, list[str]]:
    """
    Load a SOFA dictionary file.

    Format (whitespace-separated):
      <token> <ph1> <ph2> ...
    """
    path = path.expanduser().resolve()
    out: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        out[parts[0]] = parts[1:]
    if not out:
        raise ValueError(f"No entries found in SOFA dictionary: {path}")
    return out


def generate_sofa_kana_dictionary(
    *,
    kana_tokens: Iterable[str],
    romaji_dictionary: Path,
    output_dictionary: Path,
) -> None:
    """
    Generate a kana-keyed SOFA dictionary for `kana_tokens` based on a romaji-keyed dictionary.

    This lets karaoker keep `.lab` transcripts as kana while using SOFA Japanese romaji models.
    """
    romaji_to_phones = load_sofa_dictionary(romaji_dictionary)
    output_dictionary.parent.mkdir(parents=True, exist_ok=True)

    tokens = sorted({t.strip() for t in kana_tokens if t.strip()})
    lines: list[str] = []
    for tok in tokens:
        phones = kana_token_to_sofa_phones(tok, romaji_to_phones=romaji_to_phones)
        lines.append(tok + "\t" + " ".join(phones))

    output_dictionary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def kana_token_to_sofa_phones(
    kana_token: str, *, romaji_to_phones: dict[str, list[str]]
) -> list[str]:
    """
    Convert a kana token (as emitted by `karaoker.kana.kana_tokens`) into a SOFA phone sequence.
    """
    parts = _split_kana_token(kana_token)
    phones: list[str] = []
    for part in parts:
        if part in _SOKUON:
            phones.append("cl")
            continue
        if part == _LONG:
            v = _last_vowel(phones)
            if v is None:
                raise ValueError(
                    f"Cannot expand long-vowel mark with no previous vowel: {kana_token!r}"
                )
            phones.append(v)
            continue

        romaji = kana_unit_to_romaji(part)
        try:
            phones.extend(romaji_to_phones[romaji])
        except KeyError as e:
            raise KeyError(
                f"SOFA romaji dictionary is missing key {romaji!r} (from kana {part!r})."
            ) from e

    return phones


def kana_unit_to_romaji(unit: str) -> str:
    """
    Convert a kana unit (1-3 chars; may include small kana) into a romaji dictionary key.

    This is intentionally scoped to the Japanese romaji dictionary shipped for SOFA.
    """
    u = _kata_to_hira(unit)
    if u == "ん":
        return "N"

    if len(u) == 1:
        try:
            return _BASE[u]
        except KeyError as e:
            raise KeyError(f"Unsupported kana unit: {unit!r}") from e

    last = u[-1]
    if last in _SMALL_Y:
        base = u[:-1]
        base_r = kana_unit_to_romaji(base)
        if base_r == "shi":
            return {"ゃ": "sha", "ゅ": "shu", "ょ": "sho"}[last]
        if base_r == "chi":
            return {"ゃ": "cha", "ゅ": "chu", "ょ": "cho"}[last]
        if base_r == "ji":
            return {"ゃ": "ja", "ゅ": "ju", "ょ": "jo"}[last]
        if base_r.endswith("i"):
            return base_r[:-1] + _SMALL_Y[last]
        raise KeyError(f"Unsupported kana unit: {unit!r}")

    if last in _SMALL_VOWEL:
        base = u[:-1]
        base_r = kana_unit_to_romaji(base)
        v = _SMALL_VOWEL[last]

        if base_r == "shi" and v == "e":
            return "she"
        if base_r == "chi" and v == "e":
            return "che"
        if base_r == "ji" and v == "e":
            return "je"

        if base_r == "fu" and v in {"a", "i", "e", "o"}:
            return "f" + v  # fa/fi/fe/fo
        if base_r == "tsu" and v in {"a", "i", "e", "o"}:
            return "ts" + v  # tsa/tsi/tse/tso

        # Common foreign-kana patterns.
        if base_r == "te" and v == "i":
            return "ti"
        if base_r == "de" and v == "i":
            return "di"
        if base_r == "to" and v == "u":
            return "tu"
        if base_r == "do" and v == "u":
            return "du"

        if base_r.endswith("i"):
            return base_r[:-1] + "y" + v  # kyi/kye/.../dye
        raise KeyError(f"Unsupported kana unit: {unit!r}")

    raise KeyError(f"Unsupported kana unit: {unit!r}")


def _split_kana_token(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []

    out: list[str] = []
    i = 0
    while i < len(token):
        ch = token[i]
        if ch in _SOKUON or ch == _LONG:
            out.append(ch)
            i += 1
            continue

        # Base kana + (optional) one or two small kana.
        unit = ch
        i += 1
        while i < len(token) and token[i] in _SMALL:
            unit += token[i]
            i += 1
        out.append(unit)
    return out


def _last_vowel(phones: list[str]) -> str | None:
    for ph in reversed(phones):
        if ph in _VOWELS:
            return ph
    return None


def _contains_kana(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if ch == _LONG:
            return True
        # Hiragana + Katakana blocks.
        if 0x3040 <= o <= 0x30FF:
            return True
    return False


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


_BASE: dict[str, str] = {
    # vowels
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
    # small vowels (rare alone)
    "ぁ": "a",
    "ぃ": "i",
    "ぅ": "u",
    "ぇ": "e",
    "ぉ": "o",
    # k
    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",
    # s
    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",
    # t
    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",
    # n
    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",
    # h
    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",
    # m
    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",
    # y
    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",
    # small y (rare alone)
    "ゃ": "ya",
    "ゅ": "yu",
    "ょ": "yo",
    # r
    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",
    # w
    "わ": "wa",
    "ゐ": "wi",
    "ゑ": "we",
    "を": "wo",
    # g
    "が": "ga",
    "ぎ": "gi",
    "ぐ": "gu",
    "げ": "ge",
    "ご": "go",
    # z/j
    "ざ": "za",
    "じ": "ji",
    "ず": "zu",
    "ぜ": "ze",
    "ぞ": "zo",
    # d
    "だ": "da",
    "ぢ": "ji",
    "づ": "zu",
    "で": "de",
    "ど": "do",
    # b
    "ば": "ba",
    "び": "bi",
    "ぶ": "bu",
    "べ": "be",
    "ぼ": "bo",
    # p
    "ぱ": "pa",
    "ぴ": "pi",
    "ぷ": "pu",
    "ぺ": "pe",
    "ぽ": "po",
}
