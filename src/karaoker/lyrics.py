from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_RE_LRC_TAG = re.compile(r"\[([0-9]{1,2}):([0-9]{2})(?:\.([0-9]{1,3}))?\]")
_RE_LRC_META = re.compile(r"^\[(ar|ti|al|by|offset|re|ve):.*\]$", re.IGNORECASE)


@dataclass(frozen=True)
class LrcLine:
    start: float
    end: float
    text: str


def parse_lrc(path: Path) -> list[LrcLine]:
    """
    Parse LRC into timed lines. End times are inferred from the next line's start.
    Metadata lines are ignored.
    """
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: list[tuple[float, str]] = []
    for raw in raw_lines:
        line = raw.strip()
        if not line or _RE_LRC_META.match(line):
            continue

        matches = list(_RE_LRC_TAG.finditer(line))
        if not matches:
            continue

        text = _RE_LRC_TAG.sub("", line).strip()
        if not text:
            continue

        for m in matches:
            mm = int(m.group(1))
            ss = int(m.group(2))
            frac = m.group(3) or "0"
            ms = int(frac.ljust(3, "0")[:3])
            start = mm * 60 + ss + ms / 1000.0
            entries.append((start, text))

    entries.sort(key=lambda x: x[0])

    out: list[LrcLine] = []
    for i, (start, text) in enumerate(entries):
        # Default end: next line's start, otherwise unknown (we'll set later).
        end = entries[i + 1][0] if i + 1 < len(entries) else start + 5.0
        if end < start:
            end = start + 0.5
        out.append(LrcLine(start=start, end=end, text=text))
    return out


def lrc_to_text(path: Path) -> str:
    """
    Convert a .lrc file to plain transcript by joining parsed lines.
    """
    return " ".join(x.text for x in parse_lrc(path)).strip()
