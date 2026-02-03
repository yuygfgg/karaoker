from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    text: str


def _load_textgrid_via_praat_textgrids(path: Path) -> dict[str, list[Interval]] | None:
    """
    Parse a TextGrid via praat-textgrids (import: `textgrids`).

    Returns `None` if the dependency isn't available.
    """
    try:
        from textgrids import TextGrid  # type: ignore
    except Exception:
        return None

    tg = TextGrid(str(path))
    out: dict[str, list[Interval]] = {}
    for tier_name, tier in tg.items():
        intervals: list[Interval] = []
        for itv in tier:
            # praat-textgrids uses 'xmin/xmax/text' naming
            txt = getattr(itv, "text", "")
            if txt is None:
                txt = ""
            intervals.append(
                Interval(
                    start=float(getattr(itv, "xmin")),
                    end=float(getattr(itv, "xmax")),
                    text=str(txt),
                )
            )
        out[str(tier_name)] = intervals
    return out


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def _load_textgrid_fallback(path: Path) -> dict[str, list[Interval]]:
    """
    Minimal fallback parser for the common "long text" Praat TextGrid format.

    Not a full parser; it targets typical MFA outputs.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    tiers: dict[str, list[Interval]] = {}
    tier_name: str | None = None
    in_intervals = False
    cur_start: float | None = None
    cur_end: float | None = None
    cur_text: str | None = None

    def flush():
        nonlocal cur_start, cur_end, cur_text
        if tier_name is None:
            return
        if cur_start is None or cur_end is None or cur_text is None:
            return
        tiers.setdefault(tier_name, []).append(Interval(cur_start, cur_end, cur_text))
        cur_start = cur_end = None
        cur_text = None

    for raw in lines:
        line = raw.strip()
        if line.startswith("name ="):
            tier_name = _strip_quotes(line.split("=", 1)[1].strip())
            tiers.setdefault(tier_name, [])
            in_intervals = False
            continue

        if line.startswith("intervals [") or line.startswith("intervals["):
            # Starting a new interval, flush the previous one.
            flush()
            in_intervals = True
            continue

        if not in_intervals or tier_name is None:
            continue

        if line.startswith("xmin ="):
            cur_start = float(line.split("=", 1)[1].strip())
        elif line.startswith("xmax ="):
            cur_end = float(line.split("=", 1)[1].strip())
        elif line.startswith("text ="):
            cur_text = _strip_quotes(line.split("=", 1)[1].strip())

    flush()
    return tiers


def load_textgrid(path: Path) -> dict[str, list[Interval]]:
    path = path.expanduser().resolve()
    parsed = _load_textgrid_via_praat_textgrids(path)
    if parsed is not None:
        return parsed
    return _load_textgrid_fallback(path)


def _pick_tier(tiers: dict[str, list[Interval]]) -> tuple[str, list[Interval]]:
    """
    Pick a "words/kana" tier.

    Prefers common tier names; otherwise chooses the tier with the most non-empty intervals.
    """
    preferred = ["kana", "words", "word", "syllables", "mora"]
    for name in preferred:
        for k in tiers.keys():
            if k.lower() == name:
                return k, tiers[k]
    # Heuristic: if any tier contains non-empty short texts, prefer that.
    best_name: str | None = None
    best_score = -1
    for k, intervals in tiers.items():
        score = sum(1 for itv in intervals if itv.text.strip() and itv.text.strip() != "<eps>")
        if score > best_score:
            best_name = k
            best_score = score
    if best_name is not None:
        return best_name, tiers[best_name]
    raise ValueError("No tiers found in TextGrid.")


def textgrid_to_kana_events(textgrid_path: Path, *, offset_seconds: float = 0.0) -> list[dict[str, Any]]:
    """
    Convert a TextGrid tier into timed kana events.
    """
    tiers = load_textgrid(textgrid_path)
    tier_name, intervals = _pick_tier(tiers)

    events: list[dict[str, Any]] = []
    i = 0
    for itv in intervals:
        text = itv.text.strip()
        if not text or text == "<eps>":
            continue
        start = float(itv.start) + float(offset_seconds)
        end = float(itv.end) + float(offset_seconds)
        events.append(
            {
                "i": i,
                "start": round(start, 4),
                "end": round(end, 4),
                "text": text,
                "tier": tier_name,
            }
        )
        i += 1
    return events
