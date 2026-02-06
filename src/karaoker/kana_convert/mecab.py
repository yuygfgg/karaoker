from __future__ import annotations

from karaoker.kana_convert.base import KanaConverter
from karaoker.mapping import ScriptUnit, to_spaced_kana_with_units


class MecabKanaConverter(KanaConverter):
    def to_kana(self, text: str, *, output: str) -> tuple[str, list[ScriptUnit]]:
        return to_spaced_kana_with_units(text, output=output)

