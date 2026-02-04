#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _iter_script_units(data: dict[str, Any]) -> list[dict[str, Any]]:
    su = data.get("script_units")
    if isinstance(su, list):
        return [x for x in su if isinstance(x, dict)]
    return []


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

                # Shift unit char offsets into the chunk-local text.
                shifted: list[dict[str, Any]] = []
                for u in chunk:
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
