from __future__ import annotations

from pathlib import Path

from karaoker.external.gemini import (
    run_gemini_kana_convert,
    run_gemini_kana_convert_batch,
)
from karaoker.kana_convert.base import KanaConverter
from karaoker.mapping import ScriptUnit


class GeminiKanaConverter(KanaConverter):
    """
    Gemini-powered kana converter.

    This is mainly useful when you already have the lyrics text (e.g. LRC) but want a
    model-based reading instead of the local MeCab backend.

    If you also have a dry-vocals audio file, pass `input_audio`
    so Gemini can use the sung pronunciation to disambiguate readings.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
        input_audio: Path | str | None = None,
    ) -> None:
        self._model = model
        self._input_audio = (
            Path(input_audio).expanduser().resolve()
            if input_audio is not None
            else None
        )

    def to_kana(self, text: str, *, output: str) -> tuple[str, list[ScriptUnit]]:
        if output not in ("hiragana", "katakana"):
            raise ValueError(
                f"Unsupported output={output!r} (expected hiragana/katakana)."
            )

        kana = run_gemini_kana_convert(
            text=text,
            kana_output=output,  # type: ignore[arg-type]
            input_audio=self._input_audio,
            model=self._model,
        )
        tokens = kana.split() if kana else []

        # Asingle unit covering the whole string.
        units: list[ScriptUnit] = []
        if tokens:
            units = [
                ScriptUnit(
                    i=0,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    reading=" ".join(tokens),
                    ref_kana_tokens=tuple(tokens),
                    ref_kana_start=0,
                    ref_kana_end=len(tokens),
                )
            ]

        return " ".join(tokens), units

    def to_kana_batch(
        self, texts: list[str], *, output: str
    ) -> list[tuple[str, list[ScriptUnit]]]:
        if output not in ("hiragana", "katakana"):
            raise ValueError(
                f"Unsupported output={output!r} (expected hiragana/katakana)."
            )

        kanas = run_gemini_kana_convert_batch(
            texts=texts,
            kana_output=output,  # type: ignore[arg-type]
            input_audio=self._input_audio,
            model=self._model,
        )

        out: list[tuple[str, list[ScriptUnit]]] = []
        for text, kana in zip(texts, kanas, strict=True):
            tokens = kana.split() if kana else []

            # A single unit covering the whole string.
            units: list[ScriptUnit] = []
            if tokens:
                units = [
                    ScriptUnit(
                        i=0,
                        text=text,
                        char_start=0,
                        char_end=len(text),
                        reading=" ".join(tokens),
                        ref_kana_tokens=tuple(tokens),
                        ref_kana_start=0,
                        ref_kana_end=len(tokens),
                    )
                ]

            out.append((" ".join(tokens), units))

        return out
