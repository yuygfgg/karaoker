from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from karaoker.kana import kana_tokens

logger = logging.getLogger(__name__)

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
    if "\n" not in raw and "\\n" in raw:
        raw = raw.replace("\\r\\n", "\n").replace("\\n", "\n")
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


def build_gemini_kana_prompt(
    *, kana_output: KanaOutput, use_audio: bool = False
) -> str:
    kana_name = "HIRAGANA" if kana_output == "hiragana" else "KATAKANA"
    example_kana = "か ら お け" if kana_output == "hiragana" else "カ ラ オ ケ"
    task_lines: list[str]
    if use_audio:
        task_lines = [
            "- You will be given BOTH the lyrics text and an audio file (dry vocals / isolated singing).",
            "- The provided text appears somewhere in the audio. Find where it is sung.",
            "- Use the audio to disambiguate pronunciation when the text alone is ambiguous.",
            f"- Output the *sung* pronunciation reading in {kana_name}.",
            "- Do NOT transcribe new lyrics; only provide the reading of the provided text.",
            "- If the audio is missing/unhelpful for some parts, fall back to the best reading from text.",
        ]
    else:
        task_lines = [
            "- Convert the provided Japanese text into its reading in " f"{kana_name}.",
        ]
    return "\n".join(
        [
            "You are a Japanese reading (kana) converter.",
            "",
            "Task:",
            *task_lines,
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


def build_gemini_kana_batch_prompt(
    *, kana_output: KanaOutput, use_audio: bool = False
) -> str:
    kana_name = "HIRAGANA" if kana_output == "hiragana" else "KATAKANA"
    example_1_text = "風吹けば"
    example_1_kana = "か ぜ ふ け ば" if kana_output == "hiragana" else "カ ゼ フ ケ バ"
    example_2_text = "世界へ"
    example_2_kana = "せ か い え" if kana_output == "hiragana" else "セ カ イ エ"

    task_lines: list[str]
    if use_audio:
        task_lines = [
            "- You will be given BOTH an audio file (dry vocals / isolated singing) and a JSON payload of lyric segments.",
            "- For each segment, output the *sung* pronunciation reading in "
            f"{kana_name}, using the audio to disambiguate when the text alone is ambiguous.",
            "- Do NOT transcribe new lyrics; only provide readings for the provided segment texts.",
        ]
    else:
        task_lines = [
            "- You will be given a JSON payload of Japanese text segments.",
            "- For each segment, convert the text into its reading in " f"{kana_name}.",
        ]

    return "\n".join(
        [
            "You are a Japanese reading (kana) converter.",
            "",
            "Task:",
            *task_lines,
            "",
            "Input format:",
            '- You will receive JSON: {"segments": [{"i": int, "text": string}, ...]}',
            "- `i` starts at 1 and uniquely identifies a segment.",
            "",
            "Output format (STRICT):",
            "- Return ONLY valid JSON. No markdown, no code fences, no explanations.",
            '- Output JSON must be: {"segments": [{"i": int, "kana": string}, ...]}',
            "- Every input segment must appear exactly once in the output (same `i`).",
            "",
            "Kana tokenization rules (IMPORTANT):",
            "- Output kana only (plus 'ー' and spaces). Do not output punctuation.",
            "- Separate tokens with SINGLE spaces.",
            "- Do NOT output standalone small kana, standalone sokuon (ッ/っ),",
            "  or standalone prolonged mark (ー).",
            "  they must be attached to the preceding token.",
            "- Do NOT output romaji.",
            "",
            "Example input:",
            json.dumps(
                {
                    "segments": [
                        {"i": 1, "text": example_1_text},
                        {"i": 2, "text": example_2_text},
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Example output:",
            json.dumps(
                {
                    "segments": [
                        {"i": 1, "kana": example_1_kana},
                        {"i": 2, "kana": example_2_kana},
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
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


@lru_cache(maxsize=1)
def _get_genai_client():
    genai = _require_google_genai()
    return genai.Client()


@lru_cache(maxsize=4)
def _upload_audio_cached(audio_path: str, *, mtime_ns: int, size: int) -> Any:
    client = _get_genai_client()
    return client.files.upload(file=audio_path)


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
    client = _get_genai_client()
    logger.info("Gemini ASR: model=%s audio=%s", model, str(input_audio))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_json.with_suffix(output_json.suffix + ".raw.txt")

    prompt = build_gemini_asr_kana_prompt(kana_output=kana_output)
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
    input_audio: Path | str | None = None,
    model: str = "gemini-3-flash-preview",
) -> str:
    """
    Call Gemini to convert Japanese text to spaced kana tokens.

    If `input_audio` is provided, it will be uploaded and sent alongside `text`
    so Gemini can use the sung pronunciation to disambiguate readings.
    """
    _require_gemini_key()
    client = _get_genai_client()
    logger.debug(
        "Gemini kana: model=%s audio=%s",
        model,
        str(input_audio) if input_audio is not None else "none",
    )

    contents: list[Any]
    if input_audio is None:
        prompt = build_gemini_kana_prompt(kana_output=kana_output, use_audio=False)
        contents = [prompt, text]
    else:
        prompt = build_gemini_kana_prompt(kana_output=kana_output, use_audio=True)
        audio_path = Path(input_audio).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Gemini kana input_audio not found: {audio_path}")
        st = audio_path.stat()
        uploaded = _upload_audio_cached(
            str(audio_path),
            mtime_ns=int(st.st_mtime_ns),
            size=int(st.st_size),
        )
        contents = [
            prompt,
            "LYRICS_TEXT:\n" + text,
            uploaded,
        ]

    response = client.models.generate_content(model=model, contents=contents)
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


def _coerce_int(v: object) -> int | None:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
    return None


def _parse_gemini_kana_batch_response(obj: object) -> dict[int, str]:
    raw_segs: object
    if isinstance(obj, dict):
        raw_segs = obj.get("segments", obj)
    else:
        raw_segs = obj

    # Lenient fallback: {"segments": {"1": "...", "2": "..."}}.
    if isinstance(raw_segs, dict):
        out: dict[int, str] = {}
        for k, v in raw_segs.items():
            i = _coerce_int(k)
            if i is None:
                continue
            if isinstance(v, str):
                out[i] = v
            elif isinstance(v, dict) and isinstance(v.get("kana"), str):
                out[i] = cast(str, v["kana"])
        return out

    if not isinstance(raw_segs, list):
        raise ValueError(
            "Gemini JSON must be a list, or an object with a 'segments' list."
        )

    out: dict[int, str] = {}
    for idx, item in enumerate(raw_segs):
        if isinstance(item, str):
            out[idx + 1] = item
            continue
        if not isinstance(item, dict):
            continue

        i = _coerce_int(item.get("i"))
        if i is None:
            i = idx + 1

        kana = item.get("kana")
        if isinstance(kana, str):
            out[i] = kana

    return out


def run_gemini_kana_convert_batch(
    *,
    texts: list[str],
    kana_output: KanaOutput,
    input_audio: Path | str | None = None,
    model: str = "gemini-3-flash-preview",
) -> list[str]:
    """
    Call Gemini to convert many Japanese text segments to spaced kana tokens in one request.

    If `input_audio` is provided, it will be uploaded and sent alongside the JSON payload
    so Gemini can use sung pronunciation to disambiguate readings.
    """
    if not texts:
        return []

    _require_gemini_key()
    client = _get_genai_client()
    logger.info(
        "Gemini kana batch: model=%s segments=%d audio=%s",
        model,
        len(texts),
        "yes" if input_audio is not None else "no",
    )

    prompt = build_gemini_kana_batch_prompt(
        kana_output=kana_output, use_audio=input_audio is not None
    )
    payload_text = json.dumps(
        {"segments": [{"i": i + 1, "text": text} for i, text in enumerate(texts)]},
        ensure_ascii=False,
        indent=2,
    )

    contents: list[Any] = [prompt, payload_text]
    if input_audio is not None:
        audio_path = Path(input_audio).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Gemini kana input_audio not found: {audio_path}")
        st = audio_path.stat()
        uploaded = _upload_audio_cached(
            str(audio_path),
            mtime_ns=int(st.st_mtime_ns),
            size=int(st.st_size),
        )
        contents = [prompt, uploaded, payload_text]

    response = client.models.generate_content(model=model, contents=contents)
    resp_text = cast(str, getattr(response, "text", "") or "")
    obj = _extract_json(resp_text)
    mapping = _parse_gemini_kana_batch_response(obj)

    missing = [i for i in range(1, len(texts) + 1) if i not in mapping]
    if missing:
        raise ValueError(
            "Gemini kana batch response missing segment indices: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    return [
        normalize_spaced_kana(mapping[i], output=kana_output)
        for i in range(1, len(texts) + 1)
    ]
