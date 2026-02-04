from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from pykakasi import kakasi

from karaoker.kana import kana_tokens


_RE_JP_SPACE = re.compile(r"\s+")
_RE_NON_KANA = re.compile(
    r"[^\u3040-\u30ff\u30fc ]+"
)  # keep kana + prolonged mark + space


def _kakasi_instance(*, output: str):
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
    return kks


def _clean_kana(s: str) -> str:
    # Match `karaoker.kana.to_spaced_kana` cleaning rules.
    s = s.replace("\u3000", " ")
    s = _RE_NON_KANA.sub(" ", s)
    s = _RE_JP_SPACE.sub(" ", s).strip()
    return s


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


def normalize_kana_token(token: str) -> str:
    # Normalize for alignment only (does not change what we emit).
    return _kata_to_hira(token)


@dataclass(frozen=True)
class ScriptUnit:
    """
    One contiguous span of the original script text (typically a kakasi token).

    `ref_kana_*` ranges are indices into the flattened reference kana token list.
    """

    i: int
    text: str
    char_start: int
    char_end: int
    reading: str
    ref_kana_tokens: tuple[str, ...]
    ref_kana_start: int
    ref_kana_end: int


def script_to_kana_units(
    text: str, *, output: str = "katakana"
) -> tuple[list[str], list[ScriptUnit]]:
    """
    Convert script text to reference kana tokens, while keeping a mapping back to script spans.

    Returns:
      - ref_kana_tokens: flattened mora-ish tokens (the same kind of tokens used for MFA transcripts)
      - units: per-script-span mapping into ref_kana_tokens
    """
    kks = _kakasi_instance(output=output)
    pieces = kks.convert(text)

    units: list[ScriptUnit] = []
    ref_tokens: list[str] = []
    ref_i = 0
    pos = 0
    for piece in pieces:
        orig = str(piece.get("orig", ""))
        start = pos
        end = pos + len(orig)
        # Safety: if kakasi tokenization doesn't exactly cover the original text, resync.
        if text[start:end] != orig:
            found = text.find(orig, pos)
            if found != -1:
                start = found
                end = found + len(orig)
        pos = end

        reading_raw = piece.get("kana") if output == "katakana" else piece.get("hira")
        reading = _clean_kana(str(reading_raw or ""))

        toks: list[str] = []
        for chunk in reading.split(" "):
            if not chunk:
                continue
            toks.extend(kana_tokens(chunk))

        units.append(
            ScriptUnit(
                i=len(units),
                text=orig,
                char_start=start,
                char_end=end,
                reading=reading,
                ref_kana_tokens=tuple(toks),
                ref_kana_start=ref_i,
                ref_kana_end=ref_i + len(toks),
            )
        )
        ref_tokens.extend(toks)
        ref_i += len(toks)

    return ref_tokens, units


def to_spaced_kana_with_units(
    text: str, *, output: str = "katakana"
) -> tuple[str, list[ScriptUnit]]:
    ref_tokens, units = script_to_kana_units(text, output=output)
    return " ".join(ref_tokens), units


@dataclass(frozen=True)
class TokenAlignment:
    """
    Global sequence alignment between reference tokens and output tokens.

    Arrays are 0-based indices, or None if aligned to a gap.
    """

    ref_to_out: list[int | None]
    out_to_ref: list[int | None]
    cost: int


def needleman_wunsch(
    ref_tokens: list[str],
    out_tokens: list[str],
    *,
    gap_cost: int = 1,
    sub_cost: int = 1,
    normalize: Callable[[str], str] | None = normalize_kana_token,
) -> TokenAlignment:
    """
    Needleman–Wunsch global alignment (edit-distance style costs) with backtrace.
    """
    n = len(ref_tokens)
    m = len(out_tokens)

    if n == 0 and m == 0:
        return TokenAlignment(ref_to_out=[], out_to_ref=[], cost=0)
    if n == 0:
        return TokenAlignment(ref_to_out=[], out_to_ref=[None] * m, cost=m * gap_cost)
    if m == 0:
        return TokenAlignment(ref_to_out=[None] * n, out_to_ref=[], cost=n * gap_cost)

    norm = normalize or (lambda s: s)
    ref_n = [norm(t) for t in ref_tokens]
    out_n = [norm(t) for t in out_tokens]

    # DP over (n+1) x (m+1) but only keep costs for the previous row.
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    ptr = bytearray((n + 1) * (m + 1))  # 0 diag, 1 up (del), 2 left (ins)

    # Initialize first row/col.
    for j in range(1, m + 1):
        ptr[j] = 2
    for i in range(1, n + 1):
        ptr[i * (m + 1)] = 1

    for i in range(1, n + 1):
        curr[0] = i * gap_cost
        base = i * (m + 1)
        for j in range(1, m + 1):
            sub = 0 if ref_n[i - 1] == out_n[j - 1] else sub_cost
            cost_diag = prev[j - 1] + sub
            cost_up = prev[j] + gap_cost
            cost_left = curr[j - 1] + gap_cost

            # Tie-break: prefer diag > up > left (keeps matches when costs tie).
            best = cost_diag
            direction = 0
            if cost_up < best:
                best = cost_up
                direction = 1
            if cost_left < best:
                best = cost_left
                direction = 2

            curr[j] = best
            ptr[base + j] = direction

        prev, curr = curr, prev

    cost = prev[m]

    ref_to_out: list[int | None] = [None] * n
    out_to_ref: list[int | None] = [None] * m

    i = n
    j = m
    while i > 0 or j > 0:
        direction = ptr[i * (m + 1) + j]
        if i > 0 and j > 0 and direction == 0:
            i -= 1
            j -= 1
            ref_to_out[i] = j
            out_to_ref[j] = i
        elif i > 0 and (j == 0 or direction == 1):
            i -= 1
            ref_to_out[i] = None
        else:
            j -= 1
            out_to_ref[j] = None

    return TokenAlignment(ref_to_out=ref_to_out, out_to_ref=out_to_ref, cost=cost)


def map_kana_events_to_script(
    *,
    script_text: str,
    script_units: list[ScriptUnit],
    kana_events: list[dict[str, Any]],
    normalize: Callable[[str], str] | None = normalize_kana_token,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Align reference kana (from `script_text`) to timed kana events and project times back to script units.

    Returns:
      - enriched kana_events (adds script/ref indices)
      - timed script unit events (per script unit timing)
    """
    ref_tokens: list[str] = []
    ref_token_to_unit: list[int | None] = []
    for u in script_units:
        for tok in u.ref_kana_tokens:
            ref_tokens.append(tok)
            ref_token_to_unit.append(u.i)

    out_tokens = [str(e.get("text", "")) for e in kana_events]
    ali = needleman_wunsch(ref_tokens, out_tokens, normalize=normalize)

    # Enrich kana events: attach the best-effort script span for each kana token.
    enriched_events: list[dict[str, Any]] = []
    for out_i, e in enumerate(kana_events):
        ref_i = ali.out_to_ref[out_i] if out_i < len(ali.out_to_ref) else None
        unit_i = None
        unit = None
        if ref_i is not None and 0 <= ref_i < len(ref_token_to_unit):
            unit_i = ref_token_to_unit[ref_i]
            if unit_i is not None and 0 <= unit_i < len(script_units):
                unit = script_units[unit_i]

        e2 = dict(e)
        e2["ref_kana_i"] = ref_i
        e2["script_unit_i"] = unit_i
        e2["script_char_start"] = unit.char_start if unit is not None else None
        e2["script_char_end"] = unit.char_end if unit is not None else None
        e2["script_text"] = unit.text if unit is not None else ""
        enriched_events.append(e2)

    # Time each script unit by spanning the aligned kana tokens.
    timed_units: list[dict[str, Any]] = []
    for u in script_units:
        if u.ref_kana_start == u.ref_kana_end:
            continue
        out_is = [
            ali.ref_to_out[k]
            for k in range(u.ref_kana_start, u.ref_kana_end)
            if ali.ref_to_out[k] is not None
        ]
        if not out_is:
            continue
        start = min(float(kana_events[i]["start"]) for i in out_is)  # type: ignore[index]
        end = max(float(kana_events[i]["end"]) for i in out_is)  # type: ignore[index]
        timed_units.append(
            {
                "i": u.i,
                "start": round(start, 4),
                "end": round(end, 4),
                "text": u.text,
                "char_start": u.char_start,
                "char_end": u.char_end,
                "reading": u.reading,
                "ref_kana_start": u.ref_kana_start,
                "ref_kana_end": u.ref_kana_end,
            }
        )

    return enriched_events, timed_units
