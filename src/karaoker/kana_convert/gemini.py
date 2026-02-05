from __future__ import annotations

from karaoker.external.gemini import run_gemini_kana_convert
from karaoker.kana_convert.base import KanaConverter
from karaoker.mapping import ScriptUnit


class GeminiKanaConverter(KanaConverter):
    """
    Gemini-powered kana converter.

    This is mainly useful when you already have the lyrics text (e.g. LRC) but want a
    model-based reading instead of pykakasi.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
    ) -> None:
        self._model = model

    def to_kana(self, text: str, *, output: str) -> tuple[str, list[ScriptUnit]]:
        if output not in ("hiragana", "katakana"):
            raise ValueError(
                f"Unsupported output={output!r} (expected hiragana/katakana)."
            )

        kana = run_gemini_kana_convert(
            text=text,
            kana_output=output,  # type: ignore[arg-type]
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
