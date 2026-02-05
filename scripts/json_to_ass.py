#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _ass_escape(text: str) -> str:
    # ASS uses {...} for override tags; escape braces and backslashes for literal display.
    # Newlines become \N.
    return (
        text.replace("\\", "\\\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", "\n")
        .replace("\n", r"\N")
    )


def _format_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    cs = int(round(seconds * 100.0))
    s, cc = divmod(cs, 100)
    m, ss = divmod(s, 60)
    h, mm = divmod(m, 60)
    return f"{h}:{mm:02d}:{ss:02d}.{cc:02d}"


def _k(cs: int) -> str:
    # \k expects centiseconds. 0 is technically allowed but can be flaky across renderers.
    return r"{\k" + str(max(1, int(cs))) + "}"


def _hira_to_kata(text: str) -> str:
    # Hiragana [ぁ..ゖ] maps 1:1 to Katakana [ァ..ヶ] by +0x60 codepoint offset.
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if 0x3041 <= o <= 0x3096:
            out.append(chr(o + 0x60))
        else:
            out.append(ch)
    return "".join(out)


@dataclass(frozen=True)
class TimedUnit:
    start: float
    end: float
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Line:
    i: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class _Atom:
    char_start: int
    char_end: int
    elastic: bool


def _iter_script_units(data: dict[str, Any]) -> list[dict[str, Any]]:
    su = data.get("script_units")
    if isinstance(su, list):
        return [x for x in su if isinstance(x, dict)]
    return []


def _iter_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    evs = data.get("events")
    if isinstance(evs, list):
        return [x for x in evs if isinstance(x, dict)]
    return []


def _events_by_segment_ref_kana_i(
    events: list[dict[str, Any]],
) -> dict[int | None, dict[int, dict[str, Any]]]:
    """
    Build a lookup map for aligned per-kana events.

    karaoker enriches events with `ref_kana_i`. Depending on the pipeline version this index can be:
        - global (monotonic across the whole track), or
        - segment-local (restarts from 0 for each segment; includes `segment_i`).

    We key by `(segment_i, ref_kana_i)` when `segment_i` is present, otherwise use `None`.
    """
    out: dict[int | None, dict[int, dict[str, Any]]] = {}
    for e in events:
        rk = e.get("ref_kana_i")
        if rk is None:
            continue
        try:
            rk_i = int(rk)
        except Exception:
            continue

        seg_key: int | None = None
        if e.get("segment_i") is not None:
            try:
                seg_key = int(e["segment_i"])
            except Exception:
                seg_key = None

        # Validate timing early so callers can assume start/end are present and numeric.
        try:
            float(e["start"])
            float(e["end"])
        except Exception:
            continue

        bucket = out.setdefault(seg_key, {})
        # Prefer first occurrence if duplicates exist.
        bucket.setdefault(rk_i, e)
    return out


def _ref_kana_tokens_by_segment(data: dict[str, Any]) -> dict[int | None, list[str]]:
    """
    Return ref_kana tokens keyed by segment_i when available, otherwise by None.
    """
    out: dict[int | None, list[str]] = {}

    script = data.get("script")
    if isinstance(script, dict) and isinstance(script.get("ref_kana"), str):
        out[None] = script["ref_kana"].split()

    segs = data.get("segments")
    if isinstance(segs, list):
        for idx, seg in enumerate(segs):
            if not isinstance(seg, dict):
                continue
            rk = seg.get("ref_kana")
            if not isinstance(rk, str):
                continue
            seg_i_raw = seg.get("i", idx)
            try:
                seg_i = int(seg_i_raw)
            except Exception:
                seg_i = idx
            out[seg_i] = rk.split()

    return out


def _is_kana_char(ch: str) -> bool:
    # Hiragana/Katakana (includes prolonged sound mark ー in the Katakana block).
    o = ord(ch)
    return (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF)


def _is_punct_or_symbol(ch: str) -> bool:
    # Skip punctuation/symbols when distributing kana tokens to visible script glyphs.
    # These usually have no corresponding MFA kana token.
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def _kana_token_spans(text: str, *, offset: int = 0) -> list[tuple[int, int]]:
    """
    Tokenize a kana-only string into "rough mora" spans (start/end indices in `text`).

    Mirrors karaoker.kana.kana_tokens heuristics:
        - small kana attach to previous
        - sokuon (ッ/っ) attaches to previous (display-friendly)
        - prolonged sound mark (ー) attaches to previous
    """
    if not text:
        return []

    small = set("ャュョァィゥェォゃゅょぁぃぅぇぉ")
    attach_prev = set("ーッっ")

    spans: list[list[int]] = []
    for i, ch in enumerate(text):
        if not spans:
            spans.append([offset + i, offset + i + 1])
            continue
        if ch in small or ch in attach_prev:
            spans[-1][1] = offset + i + 1
        else:
            spans.append([offset + i, offset + i + 1])

    return [(s, e) for s, e in spans]


def _visible_atoms(text: str) -> list[_Atom]:
    """
    Split visible script text into display atoms that can be assigned kana token timings.

    - whitespace / punctuation / symbols are untimed (excluded)
    - kana runs are tokenized with `_kana_token_spans` (1 atom per rough mora)
    - ASCII alnum runs are grouped (so "LOVE" doesn't require 4 kana tokens)
    - everything else is per-codepoint
    """
    atoms: list[_Atom] = []
    i = 0
    while i < len(text):
        ch = text[i]

        if ch.isspace() or _is_punct_or_symbol(ch):
            i += 1
            continue

        if _is_kana_char(ch):
            j = i + 1
            while j < len(text) and _is_kana_char(text[j]):
                j += 1
            for s, e in _kana_token_spans(text[i:j], offset=i):
                atoms.append(_Atom(char_start=s, char_end=e, elastic=False))
            i = j
            continue

        if ch.isascii() and ch.isalnum():
            j = i + 1
            while j < len(text) and text[j].isascii() and text[j].isalnum():
                j += 1
            atoms.append(_Atom(char_start=i, char_end=j, elastic=True))
            i = j
            continue

        atoms.append(_Atom(char_start=i, char_end=i + 1, elastic=True))
        i += 1

    return atoms


def _expand_script_unit_to_syllables(
    u: dict[str, Any],
    *,
    ref_kana_tokens: list[str],
    events_by_ref_kana_i: dict[int, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """
    If this script unit is pure kana, expand it to per-(ref_kana) syllable timing.

    We only expand when the script unit's visible text maps 1:1 to its reading, so
    we can safely assign each kana token to a substring of the displayed text.
    """
    try:
        char_start = int(u["char_start"])
        char_end = int(u["char_end"])
    except Exception:
        return None
    if char_end <= char_start:
        return None

    text = u.get("text")
    reading = u.get("reading")
    if not isinstance(text, str) or not isinstance(reading, str) or not reading:
        return None

    try:
        rk_start = int(u["ref_kana_start"])
        rk_end = int(u["ref_kana_end"])
    except Exception:
        return None
    if rk_end <= rk_start:
        return None
    if rk_start < 0 or rk_end > len(ref_kana_tokens):
        return None

    # Only expand when the unit text is kana that matches the reading exactly.
    if _hira_to_kata(text) != reading:
        return None
    if "".join(ref_kana_tokens[rk_start:rk_end]) != reading:
        return None

    # Ensure char span length matches the rendered text; otherwise mapping is unsafe.
    if (char_end - char_start) != len(text):
        return None

    out: list[dict[str, Any]] = []
    pos = 0
    for ref_i in range(rk_start, rk_end):
        tok = ref_kana_tokens[ref_i]
        if not tok:
            return None
        if reading[pos : pos + len(tok)] != tok:
            return None

        ev = events_by_ref_kana_i.get(ref_i)
        if ev is None:
            return None
        start = float(ev["start"])
        end = float(ev["end"])

        out.append(
            {
                "start": start,
                "end": end,
                "char_start": char_start + pos,
                "char_end": char_start + pos + len(tok),
            }
        )
        pos += len(tok)

    if pos != len(text):
        return None
    return out


def _expand_script_unit_to_animation_units(
    u: dict[str, Any],
    *,
    script_text: str,
    tok_events: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """
    Expand a script unit into finer ASS-timed units using per-kana token timings.

    Unlike `_expand_script_unit_to_syllables`, this works for mixed-script text (kanji+kana),
    by distributing kana tokens over "visible atoms" derived from the script text.
    """
    if not tok_events:
        return None

    try:
        char_start = int(u["char_start"])
        char_end = int(u["char_end"])
    except Exception:
        return None
    if char_end <= char_start:
        return None

    text = u.get("text")
    if not isinstance(text, str) or (char_end - char_start) != len(text):
        # Fall back to the canonical slice when we have it.
        if 0 <= char_start <= len(script_text):
            char_end = min(char_end, len(script_text))
            text = script_text[char_start:char_end]
        else:
            return None

    atoms = _visible_atoms(text)
    if not atoms:
        return None

    if len(tok_events) < len(atoms):
        return None

    # DP segmentation: assign a contiguous slice of kana tokens to each visible atom.
    # We heavily prefer matching visible kana atoms to their corresponding kana token(s),
    # and otherwise distribute tokens roughly evenly across non-kana atoms.
    try:
        token_texts = [_hira_to_kata(str(e.get("text", ""))) for e in tok_events]
        # Validate timing fields eagerly.
        for e in tok_events:
            float(e["start"])
            float(e["end"])
    except Exception:
        return None

    atom_texts = [text[a.char_start : a.char_end] for a in atoms]
    atom_is_kana = [
        bool(s) and all(_is_kana_char(ch) for ch in s) for s in atom_texts
    ]

    def _kana_variants(s: str) -> set[str]:
        s2 = _hira_to_kata(s)
        out = {s2}
        # Common particle spelling vs pronunciation.
        if s2 == "ヲ":
            out.add("オ")
        elif s2 == "ヘ":
            out.add("エ")
        elif s2 == "ハ":
            out.add("ワ")
        return out

    kana_atom_variants = [
        _kana_variants(s) if is_kana else set() for s, is_kana in zip(atom_texts, atom_is_kana)
    ]

    K = len(atoms)
    N = len(tok_events)
    INF = 10**12
    dp: list[list[int]] = [[INF] * (N + 1) for _ in range(K + 1)]
    back: list[list[int | None]] = [[None] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 0

    for i in range(1, K + 1):
        # Atoms and tokens are 1-indexed in DP, but arrays are 0-indexed.
        is_kana = atom_is_kana[i - 1]
        variants = kana_atom_variants[i - 1]
        # j = total tokens consumed so far.
        j_min = i  # at least 1 token per atom
        j_max = N - (K - i)  # leave at least 1 token for each remaining atom
        for j in range(j_min, j_max + 1):
            # Try all previous split points j0 (< j).
            best_cost = INF
            best_j0: int | None = None
            for j0 in range(i - 1, j):
                prev = dp[i - 1][j0]
                if prev >= INF:
                    continue

                slice_text = "".join(token_texts[j0:j])
                c = j - j0
                if is_kana:
                    # Prefer matching kana atoms to their token(s). Allow 1+ tokens when the
                    # concatenation matches (e.g. キ + ャ == キャ).
                    mismatch = 0 if slice_text in variants else 10_000
                    # Small penalty for consuming multiple tokens, but allow it when needed.
                    cost = prev + mismatch + (c - 1) * 10
                else:
                    # Even-ish distribution across non-kana atoms.
                    cost = prev + (c - 1) * (c - 1)

                if cost < best_cost:
                    best_cost = cost
                    best_j0 = j0

            dp[i][j] = best_cost
            back[i][j] = best_j0

    if dp[K][N] >= INF or back[K][N] is None:
        return None

    # Reconstruct token counts per atom.
    alloc: list[int] = [0] * K
    j = N
    for i in range(K, 0, -1):
        j0 = back[i][j]
        if j0 is None:
            return None
        alloc[i - 1] = j - j0
        j = j0
    if j != 0 or any(n <= 0 for n in alloc):
        return None

    out: list[dict[str, Any]] = []
    idx = 0
    for atom, n_tok in zip(atoms, alloc):
        ev0 = tok_events[idx]
        ev1 = tok_events[idx + n_tok - 1]
        out.append(
            {
                "start": float(ev0["start"]),
                "end": float(ev1["end"]),
                "char_start": char_start + atom.char_start,
                "char_end": char_start + atom.char_end,
            }
        )
        idx += n_tok

    if idx != len(tok_events):
        return None
    return out


def _iter_lines(data: dict[str, Any]) -> list[Line]:
    lines = data.get("lines")
    if not isinstance(lines, list):
        return []
    out: list[Line] = []
    for x in lines:
        if not isinstance(x, dict):
            continue
        try:
            out.append(
                Line(
                    i=int(x.get("i", len(out))),
                    start=float(x["start"]),
                    end=float(x["end"]),
                    text=str(x.get("text", "")),
                )
            )
        except Exception:
            continue
    out.sort(key=lambda line: line.start)
    return out


def _timed_units_for_text(
    *,
    full_text: str,
    raw_units: Iterable[dict[str, Any]],
) -> list[TimedUnit]:
    units: list[TimedUnit] = []
    for u in raw_units:
        try:
            cs = int(u["char_start"])
            ce = int(u["char_end"])
            if cs < 0 or ce < 0:
                continue
            if ce <= cs:
                continue
            if cs > len(full_text):
                continue
            ce = min(ce, len(full_text))
            units.append(
                TimedUnit(
                    start=float(u["start"]),
                    end=float(u["end"]),
                    char_start=cs,
                    char_end=ce,
                )
            )
        except Exception:
            continue

    # Prefer text order; for correctly aligned outputs this is also chronological.
    units.sort(key=lambda x: (x.char_start, x.start))
    return units


def _render_karaoke_text(
    *,
    full_text: str,
    units: list[TimedUnit],
    line_start: float,
) -> str:
    """
    Render ASS text with \\k tags for timed units, preserving the original script text.

    Untimed spans (punctuation/whitespace not covered by timed units) are emitted as plain text.
    """
    if not units:
        return _ass_escape(full_text)

    parts: list[str] = []

    # Leading time gap (no visible text) so the first highlight starts at units[0].start.
    cursor_t = float(line_start)
    if units[0].start > cursor_t:
        parts.append(_k(int(round((units[0].start - cursor_t) * 100.0))))
        cursor_t = units[0].start

    pos = 0
    prev_end_t = units[0].start
    for u in units:
        # If alignment produced a pause between timed units, account for it.
        if u.start > prev_end_t:
            parts.append(_k(int(round((u.start - prev_end_t) * 100.0))))

        cs = max(pos, min(len(full_text), int(u.char_start)))
        ce = max(cs, min(len(full_text), int(u.char_end)))
        if cs > pos:
            parts.append(_ass_escape(full_text[pos:cs]))

        dur_cs = int(round(max(0.0, u.end - u.start) * 100.0))
        parts.append(_k(dur_cs) + _ass_escape(full_text[cs:ce]))
        pos = ce
        prev_end_t = max(prev_end_t, u.end)

    if pos < len(full_text):
        parts.append(_ass_escape(full_text[pos:]))
    return "".join(parts)


def _chunk_script_units(
    raw_units: list[dict[str, Any]],
    *,
    gap_threshold_s: float,
    max_caption_s: float,
) -> list[list[dict[str, Any]]]:
    # Units should already have start/end/char_start/char_end.
    units = [
        u
        for u in raw_units
        if "start" in u and "end" in u and "char_start" in u and "char_end" in u
    ]
    units.sort(key=lambda u: float(u.get("start", 0.0)))
    if not units:
        return []

    out: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    chunk_start = float(units[0]["start"])
    prev_end = float(units[0]["end"])

    for u in units:
        s = float(u["start"])
        e = float(u["end"])
        gap = s - prev_end
        span = e - chunk_start
        if cur and (gap > gap_threshold_s or span > max_caption_s):
            out.append(cur)
            cur = []
            chunk_start = s
        cur.append(u)
        prev_end = max(prev_end, e)

    if cur:
        out.append(cur)
    return out


def subtitles_json_to_ass(
    data: dict[str, Any],
    *,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    fontname: str = "Arial",
    fontsize: int = 60,
    margin_v: int = 40,
    lead_in_s: float = 0.0,
    tail_out_s: float = 0.0,
    gap_threshold_s: float = 0.8,
    max_caption_s: float = 8.0,
) -> str:
    """
    Convert karaoker `subtitles.json` to ASS karaoke subtitles.
    """
    raw_units = _iter_script_units(data)
    raw_events = _iter_events(data)
    lines = _iter_lines(data)

    # Pick script text source.
    script_text: str | None = None
    script = data.get("script")
    if isinstance(script, dict) and isinstance(script.get("text"), str):
        script_text = script["text"]

    # ASS header
    header = "\n".join(
        [
            "[Script Info]",
            "; Script generated by karaoker (scripts/json_to_ass.py)",
            "ScriptType: v4.00+",
            f"PlayResX: {int(play_res_x)}",
            f"PlayResY: {int(play_res_y)}",
            "ScaledBorderAndShadow: yes",
            "WrapStyle: 0",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            # Primary = base color, Secondary = karaoke fill color.
            (
                "Style: Default,"
                f"{fontname},{int(fontsize)},"
                "&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,"
                "0,0,0,0,100,100,0,0,1,3,0,2,60,60,"
                f"{int(margin_v)},1"
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    events: list[str] = []

    if lines:
        # LRC mode: one dialogue per line, timed by LRC start/end, karaoke by per-line script_units.
        for ln in lines:
            ln_units = [u for u in raw_units if int(u.get("line_i", -1)) == ln.i]
            full_text = ln.text
            timed = _timed_units_for_text(full_text=full_text, raw_units=ln_units)

            start = max(0.0, float(ln.start) - float(lead_in_s))
            end = max(start + 0.01, float(ln.end) + float(tail_out_s))
            text = _render_karaoke_text(
                full_text=full_text, units=timed, line_start=start
            )

            events.append(
                "Dialogue: 0,"
                f"{_format_ass_time(start)},{_format_ass_time(end)},"
                "Default,,0,0,0,,"
                f"{text}"
            )
    else:
        # ASR mode: chunk script_units by pause/duration.
        if script_text is None:
            script_text = ""

        # Optional: use per-kana timing (`events`) to create finer karaoke units within
        # each script span (including mixed kanji+kana text).
        ref_kana_tokens_by_seg = _ref_kana_tokens_by_segment(data)
        events_by_seg = _events_by_segment_ref_kana_i(raw_events)

        chunks = _chunk_script_units(
            raw_units,
            gap_threshold_s=float(gap_threshold_s),
            max_caption_s=float(max_caption_s),
        )
        if not chunks:
            # Fallback: no script units -> cannot preserve kanji; emit kana if available.
            evs = data.get("events")
            if isinstance(evs, list) and evs:
                start = float(evs[0].get("start", 0.0))
                end = float(evs[-1].get("end", start + 1.0))
                parts: list[str] = []
                for e in evs:
                    try:
                        dur_cs = int(
                            round((float(e["end"]) - float(e["start"])) * 100.0)
                        )
                        parts.append(_k(dur_cs) + _ass_escape(str(e.get("text", ""))))
                    except Exception:
                        continue
                text = "".join(parts) if parts else _ass_escape(script_text)
                events.append(
                    "Dialogue: 0,"
                    f"{_format_ass_time(start)},{_format_ass_time(end)},"
                    "Default,,0,0,0,,"
                    f"{text}"
                )
        else:
            for chunk in chunks:
                # Determine the substring of the script this chunk covers.
                cs = min(int(u["char_start"]) for u in chunk)
                ce = max(int(u["char_end"]) for u in chunk)
                cs = max(0, min(cs, len(script_text)))
                ce = max(cs, min(ce, len(script_text)))
                full_text = script_text[cs:ce]

                # Expand script units to finer "animation units" when per-kana timings
                # are available, then shift unit char offsets into the chunk-local text.
                expanded: list[dict[str, Any]] = []
                for u in chunk:
                    seg_key: int | None = None
                    if u.get("segment_i") is not None:
                        try:
                            seg_key = int(u["segment_i"])
                        except Exception:
                            seg_key = None

                    ref_kana_tokens = ref_kana_tokens_by_seg.get(seg_key) or ref_kana_tokens_by_seg.get(None, [])
                    by_ref = events_by_seg.get(seg_key) or events_by_seg.get(None, {})

                    pieces: list[dict[str, Any]] | None = None
                    if by_ref:
                        # Exact mapping for pure kana units (when safe).
                        if ref_kana_tokens:
                            pieces = _expand_script_unit_to_syllables(
                                u,
                                ref_kana_tokens=ref_kana_tokens,
                                events_by_ref_kana_i=by_ref,
                            )
                        # Heuristic mapping for mixed-script units.
                        if pieces is None and u.get("ref_kana_start") is not None and u.get("ref_kana_end") is not None:
                            try:
                                rk_start = int(u["ref_kana_start"])
                                rk_end = int(u["ref_kana_end"])
                            except Exception:
                                rk_start = rk_end = 0

                            if rk_end > rk_start:
                                tok_events: list[dict[str, Any]] = []
                                for ref_i in range(rk_start, rk_end):
                                    ev = by_ref.get(ref_i)
                                    if ev is None:
                                        tok_events = []
                                        break
                                    tok_events.append(ev)
                                if tok_events:
                                    pieces = _expand_script_unit_to_animation_units(
                                        u,
                                        script_text=script_text,
                                        tok_events=tok_events,
                                    )
                    if pieces is not None:
                        expanded.extend(pieces)
                    else:
                        expanded.append(u)

                shifted: list[dict[str, Any]] = []
                for u in expanded:
                    u2 = dict(u)
                    u2["char_start"] = int(u["char_start"]) - cs
                    u2["char_end"] = int(u["char_end"]) - cs
                    shifted.append(u2)
                timed = _timed_units_for_text(full_text=full_text, raw_units=shifted)

                start_u = float(chunk[0]["start"])
                end_u = max(float(u["end"]) for u in chunk)
                start = max(0.0, start_u - float(lead_in_s))
                end = max(start + 0.01, end_u + float(tail_out_s))
                text = _render_karaoke_text(
                    full_text=full_text, units=timed, line_start=start
                )

                events.append(
                    "Dialogue: 0,"
                    f"{_format_ass_time(start)},{_format_ass_time(end)},"
                    "Default,,0,0,0,,"
                    f"{text}"
                )

    return header + "\n" + "\n".join(events) + ("\n" if events else "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert karaoker subtitles.json to ASS karaoke subtitles."
    )
    ap.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to subtitles.json (karaoker output).",
    )
    ap.add_argument("--output", required=True, type=Path, help="Output .ass path.")

    ap.add_argument("--play-res-x", type=int, default=1920)
    ap.add_argument("--play-res-y", type=int, default=1080)
    ap.add_argument("--fontname", default="Arial")
    ap.add_argument("--fontsize", type=int, default=60)
    ap.add_argument("--margin-v", type=int, default=40)

    ap.add_argument(
        "--lead-in",
        type=float,
        default=0.0,
        help="Seconds to show caption before first highlight.",
    )
    ap.add_argument(
        "--tail-out",
        type=float,
        default=0.0,
        help="Seconds to keep caption after last highlight.",
    )
    ap.add_argument(
        "--gap-threshold",
        type=float,
        default=0.8,
        help="ASR mode: split captions when the pause between units exceeds this (seconds).",
    )
    ap.add_argument(
        "--max-caption",
        type=float,
        default=8.0,
        help="ASR mode: maximum caption length in seconds before forcing a split.",
    )

    args = ap.parse_args(argv)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Expected a JSON object at the top level.")

    ass = subtitles_json_to_ass(
        data,
        play_res_x=args.play_res_x,
        play_res_y=args.play_res_y,
        fontname=args.fontname,
        fontsize=args.fontsize,
        margin_v=args.margin_v,
        lead_in_s=args.lead_in,
        tail_out_s=args.tail_out,
        gap_threshold_s=args.gap_threshold,
        max_caption_s=args.max_caption,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(ass, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
