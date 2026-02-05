from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from karaoker.kana import kana_tokens

KanaOutput = Literal["hiragana", "katakana"]

_RE_JP_SPACE = re.compile(r"\s+")
_RE_NON_KANA_OR_SPACE = re.compile(r"[^\u3040-\u30ff\u30fc ]+")
_RE_TS_HHMMSS = re.compile(
    r"^(?P<h>\d+):(?P<m>[0-9]{2}):(?P<s>[0-9]{2})(?:[,.](?P<ms>[0-9]{1,3}))?$"
)
_RE_TS_MMSS = re.compile(r"^(?P<m>\d+):(?P<s>[0-9]{2})(?:[,.](?P<ms>[0-9]{1,3}))?$")


def _clean_kana_keep_spaces(s: str) -> str:
    # Match cleaning rules in karaoker.kana/to_spaced_kana and karaoker.mapping.
    s = s.replace("\u3000", " ")
    s = _RE_NON_KANA_OR_SPACE.sub(" ", s)
    s = _RE_JP_SPACE.sub(" ", s).strip()
    return s


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


def normalize_spaced_kana(kana: str, *, output: KanaOutput) -> str:
    """
    Normalize a kana reading into karaoker's "spaced mora-ish tokens" format.

    - Drops non-kana characters (except spaces and prolonged mark 'ー').
    - Collapses whitespace.
    - Converts to either pure hiragana or pure katakana.
    - Re-tokenizes using karaoker.kana.kana_tokens (small kana / ー / ッ attach to previous).
    """
    cleaned = _clean_kana_keep_spaces(kana)
    if not cleaned:
        return ""

    # We don't preserve word boundaries; MFA runs per-token, not per-word.
    joined = cleaned.replace(" ", "")
    joined = _kata_to_hira(joined) if output == "hiragana" else _hira_to_kata(joined)
    toks = kana_tokens(joined)
    return " ".join(toks)


def _parse_timestamp_seconds(v: object) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass

    m = _RE_TS_HHMMSS.match(s)
    if m:
        h = int(m.group("h"))
        mm = int(m.group("m"))
        ss = int(m.group("s"))
        ms_raw = m.group("ms") or "0"
        ms = int(ms_raw.ljust(3, "0")[:3])
        return h * 3600 + mm * 60 + ss + ms / 1000.0

    m = _RE_TS_MMSS.match(s)
    if m:
        mm = int(m.group("m"))
        ss = int(m.group("s"))
        ms_raw = m.group("ms") or "0"
        ms = int(ms_raw.ljust(3, "0")[:3])
        return mm * 60 + ss + ms / 1000.0

    return None


def _extract_json(text: str) -> Any:
    """
    Best-effort extraction of JSON from a model response.

    We ask Gemini to return JSON-only, but in practice it may wrap it in code fences.
    """
    raw = text.strip()
    if not raw:
        raise ValueError("Empty Gemini response.")

    # First try: parse as-is.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences if present.
    if raw.startswith("```"):
        # ```json\n...\n```
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw2 = raw[first_nl + 1 :]
            end = raw2.rfind("```")
            if end != -1:
                raw2 = raw2[:end].strip()
                try:
                    return json.loads(raw2)
                except json.JSONDecodeError:
                    raw = raw2

    # Last resort: find the outermost JSON object/array.
    starts = [i for i in [raw.find("["), raw.find("{")] if i != -1]
    if not starts:
        raise ValueError("Gemini response did not contain JSON.")
    start = min(starts)
    if raw[start] == "[":
        end = raw.rfind("]")
    else:
        end = raw.rfind("}")
    if end == -1 or end <= start:
        raise ValueError("Gemini response had truncated JSON.")
    snippet = raw[start : end + 1].strip()
    return json.loads(snippet)


@dataclass(frozen=True)
class GeminiLyricSegment:
    i: int
    start: float
    end: float
    text: str
    kana: str


def parse_gemini_lyrics_segments(
    obj: object, *, kana_output: KanaOutput
) -> list[GeminiLyricSegment]:
    """
    Parse and normalize the JSON payload we ask Gemini to return.

    Accepts either:
        - a list of {i,start,end,text,kana}
        - a dict with a top-level "segments" list
    """
    raw_segs: object
    if isinstance(obj, dict):
        raw_segs = obj.get("segments")
    else:
        raw_segs = obj

    if not isinstance(raw_segs, list):
        raise ValueError(
            "Gemini JSON must be a list, or an object with a 'segments' list."
        )

    out: list[GeminiLyricSegment] = []
    for idx, item in enumerate(raw_segs):
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        if not text:
            continue

        start_s = _parse_timestamp_seconds(item.get("start"))
        end_s = _parse_timestamp_seconds(item.get("end"))
        if start_s is None or end_s is None:
            raise ValueError(
                "Gemini segment missing/invalid start/end timestamps. "
                f"Segment index={idx}, start={item.get('start')!r}, end={item.get('end')!r}"
            )

        start_s = max(0.0, float(start_s))
        end_s = float(end_s)
        if end_s <= start_s:
            end_s = start_s + 0.5

        kana_raw = str(item.get("kana", "")).strip()
        kana_norm = normalize_spaced_kana(kana_raw, output=kana_output)
        if not kana_norm:
            # Keep the segment but it likely won't align; upstream may choose to drop.
            kana_norm = ""

        i_val = item.get("i")
        i_norm = int(i_val) if isinstance(i_val, int) else idx + 1

        out.append(
            GeminiLyricSegment(
                i=i_norm,
                start=start_s,
                end=end_s,
                text=text,
                kana=kana_norm,
            )
        )

    return out


def build_gemini_asr_kana_prompt(*, kana_output: KanaOutput) -> str:
    kana_name = "HIRAGANA" if kana_output == "hiragana" else "KATAKANA"
    example_kana = "か ぜ ふ け ば" if kana_output == "hiragana" else "カ ゼ フ ケ バ"
    return "\n".join(
        [
            "You are a professional Japanese lyric transcriber and phonetic annotator.",
            "",
            "Task:",
            "- Listen to the attached audio file (a Japanese song).",
            "- Transcribe the sung lyrics and provide line-level timestamps.",
            "- Transcribe the lyrics carefully and as accurately as possible. Make sure they are high-quality."
            "- For each line, also output the *sung* pronunciation in "
            f"{kana_name} (what is actually sung, not dictionary reading).",
            "",
            "Output format (STRICT):",
            "- Output plain text ONLY. No markdown, no code fences, no explanations.",
            "- For each segment, output exactly 4 lines, then a blank line:",
            "  1) <index starting from 1>",
            "  2) <start_time> --> <end_time>",
            "  3) <lyrics text>",
            "  4) <kana reading>",
            "",
            "Timestamp format:",
            "- Prefer `HH:MM:SS.mmm` (e.g. `00:01:23.456`).",
            "- `start_time` and `end_time` must be increasing.",
            "- `end_time` must be strictly greater than `start_time`.",
            "",
            "Kana tokenization rules (IMPORTANT):",
            "- The kana output must contain only kana characters and spaces.",
            "- The prolonged sound mark 'ー' is allowed.",
            "- Separate tokens with SINGLE spaces.",
            "- Do NOT output standalone small kana (ャュョァィゥェォゃゅょぁぃぅぇぉ), standalone sokuon (ッ/っ),",
            "  or standalone prolonged sound mark (ー).",
            "  They must be attached to the preceding token.",
            "- Do NOT output romaji.",
            "",
            "Segmentation and timing:",
            "- Prefer natural lyric line boundaries (phrases).",
            "- Ignore long instrumental-only or silent sections (do not invent lyrics).",
            "",
            "Example:",
            "1",
            "00:00:00.000 --> 00:00:02.300",
            "風吹けば",
            example_kana,
            "",
            "",
            "Now transcribe the attached audio.",
        ]
    )


def _strip_markdown_fences(s: str) -> str:
    raw = s.strip()
    if not raw.startswith("```"):
        return raw
    first_nl = raw.find("\n")
    if first_nl == -1:
        return raw
    body = raw[first_nl + 1 :]
    end = body.rfind("```")
    if end == -1:
        return raw
    return body[:end].strip()


def _parse_gemini_block_format(
    text: str, *, kana_output: KanaOutput
) -> list[GeminiLyricSegment]:
    """
    Parse the requested plain-text block format:

        <i>
        <start> --> <end>
        <text>
        <kana>
        (blank)
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    out: list[GeminiLyricSegment] = []

    i = 0
    while i < len(lines):
        # Skip blank lines.
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        idx_line = lines[i].strip()
        i += 1
        try:
            seg_i = int(idx_line)
        except ValueError:
            raise ValueError(
                f"Expected segment index integer, got: {idx_line!r}"
            ) from None

        # Timestamp line
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            raise ValueError(f"Missing timestamp line for segment {seg_i}.")
        ts_line = lines[i].strip()
        i += 1
        if "-->" not in ts_line:
            raise ValueError(
                f"Expected '<start> --> <end>' for segment {seg_i}, got: {ts_line!r}"
            )
        left, right = (p.strip() for p in ts_line.split("-->", 1))
        start_s = _parse_timestamp_seconds(left)
        end_s = _parse_timestamp_seconds(right)
        if start_s is None or end_s is None:
            raise ValueError(
                f"Invalid timestamps for segment {seg_i}: start={left!r}, end={right!r}"
            )

        # Text line
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            raise ValueError(f"Missing text line for segment {seg_i}.")
        seg_text = lines[i].strip()
        i += 1

        # Kana line
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            raise ValueError(f"Missing kana line for segment {seg_i}.")
        kana_raw = lines[i].strip()
        i += 1

        kana_norm = normalize_spaced_kana(kana_raw, output=kana_output)
        start_f = max(0.0, float(start_s))
        end_f = float(end_s)
        if end_f <= start_f:
            end_f = start_f + 0.5

        if seg_text:
            out.append(
                GeminiLyricSegment(
                    i=seg_i,
                    start=start_f,
                    end=end_f,
                    text=seg_text,
                    kana=kana_norm,
                )
            )

    return out


def parse_gemini_asr_kana_response(
    text: str, *, kana_output: KanaOutput
) -> list[GeminiLyricSegment]:
    """
    Parse a Gemini response into lyric segments.

    We primarily expect the plain-text block format, but accept JSON as a fallback
    (useful when models ignore formatting instructions).
    """
    raw = _strip_markdown_fences(text)

    try:
        obj = _extract_json(raw)
    except Exception:
        obj = None

    if obj is not None:
        try:
            segs = parse_gemini_lyrics_segments(obj, kana_output=kana_output)
            if segs:
                return segs
        except Exception:
            # Fall back to block parsing below.
            pass

    return _parse_gemini_block_format(raw, kana_output=kana_output)


def build_gemini_kana_prompt(*, kana_output: KanaOutput) -> str:
    kana_name = "HIRAGANA" if kana_output == "hiragana" else "KATAKANA"
    example_kana = "か ら お け" if kana_output == "hiragana" else "カ ラ オ ケ"
    return "\n".join(
        [
            "You are a Japanese reading (kana) converter.",
            "",
            "Task:",
            "- Convert the provided Japanese text into its reading in " f"{kana_name}.",
            "",
            "Output format (STRICT):",
            "- Return ONLY valid JSON. No markdown, no code fences, no explanations.",
            "- The JSON value must be an object with this key:",
            '  - "kana": string (kana tokens separated by single spaces)',
            "",
            "Kana tokenization rules (IMPORTANT):",
            "- Output kana only (plus 'ー' and spaces). Do not output punctuation.",
            "- Separate tokens with SINGLE spaces.",
            "- Do NOT output standalone small kana, standalone sokuon (ッ/っ),",
            "  or standalone prolonged mark (ー).",
            "  they must be attached to the preceding token.",
            "- Do NOT output romaji.",
            "",
            "Example:",
            f'{{ "kana": "{example_kana}" }}',
        ]
    )


def _require_gemini_key() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "Gemini backend requires the environment variable GEMINI_API_KEY to be set."
        )


def _require_google_genai():
    try:
        from google import genai  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "Gemini backend requires the `google-genai` package. "
            "Install it via: pip install -e '.[gemini]'"
        ) from e
    return genai


def run_gemini_asr_with_kana(
    *,
    input_audio: Path,
    output_json: Path,
    kana_output: KanaOutput,
    model: str = "gemini-3-flash-preview",
) -> dict[str, Any]:
    """
    Call Gemini to perform ASR + kana reading in one pass.

    Writes a structured JSON payload to `output_json` and returns the parsed dict.
    """
    _require_gemini_key()
    genai = _require_google_genai()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_json.with_suffix(output_json.suffix + ".raw.txt")

    prompt = build_gemini_asr_kana_prompt(kana_output=kana_output)
    client = genai.Client()
    uploaded = client.files.upload(file=str(input_audio))
    response = client.models.generate_content(model=model, contents=[prompt, uploaded])
    text = cast(str, getattr(response, "text", "") or "")
    raw_path.write_text(text, encoding="utf-8")

    segments = parse_gemini_asr_kana_response(text, kana_output=kana_output)

    payload: dict[str, Any] = {
        "provider": "gemini",
        "model": model,
        "kana_output": kana_output,
        "segments": [
            {
                "i": s.i,
                "start": round(float(s.start), 4),
                "end": round(float(s.end), 4),
                "text": s.text,
                "kana": s.kana,
            }
            for s in segments
        ],
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_gemini_kana_convert(
    *,
    text: str,
    kana_output: KanaOutput,
    model: str = "gemini-3-flash-preview",
) -> str:
    """
    Call Gemini to convert Japanese text to spaced kana tokens.
    """
    _require_gemini_key()
    genai = _require_google_genai()

    prompt = build_gemini_kana_prompt(kana_output=kana_output)
    client = genai.Client()
    response = client.models.generate_content(model=model, contents=[prompt, text])
    resp_text = cast(str, getattr(response, "text", "") or "")
    obj = _extract_json(resp_text)

    kana_raw: str
    if isinstance(obj, dict) and isinstance(obj.get("kana"), str):
        kana_raw = obj["kana"]
    elif isinstance(obj, str):
        kana_raw = obj
    else:
        raise ValueError('Gemini kana response must be JSON: {"kana": "..."}.')

    return normalize_spaced_kana(kana_raw, output=kana_output)
