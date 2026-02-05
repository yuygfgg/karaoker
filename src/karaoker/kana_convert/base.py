from __future__ import annotations

from abc import ABC, abstractmethod

from karaoker.mapping import ScriptUnit


class KanaConverter(ABC):
    @abstractmethod
    def to_kana(self, text: str, *, output: str) -> tuple[str, list[ScriptUnit]]:
        raise NotImplementedError
