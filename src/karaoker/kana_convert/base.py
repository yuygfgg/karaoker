from __future__ import annotations

from abc import ABC, abstractmethod

from karaoker.mapping import ScriptUnit


class KanaConverter(ABC):
    @abstractmethod
    def to_kana(self, text: str, *, output: str) -> tuple[str, list[ScriptUnit]]:
        """
        Convert Japanese script text to spaced kana tokens + a mapping back to script spans.

        `output` is typically either "katakana" or "hiragana".

        Returns:
            - ref_kana: space-separated kana tokens (used as the MFA transcript)
            - script_units: spans of the original script mapped to token ranges
        """
        raise NotImplementedError


def build_kana_converter(name: str, /, **kwargs) -> KanaConverter:
    """
    Convenience factory for kana converters.

    Imports backends lazily to avoid import cycles.
    """
    key = name.strip().lower()
    if key in {"mecab", "unidic", "default"}:
        from karaoker.kana_convert.mecab import MecabKanaConverter

        return MecabKanaConverter()
    if key in {"gemini"}:
        from karaoker.kana_convert.gemini import GeminiKanaConverter

        return GeminiKanaConverter(**kwargs)

    raise ValueError(f"Unknown kana converter: {name!r}")
