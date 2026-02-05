from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AlignerProvider(ABC):
    @abstractmethod
    def g2p(
        self, *, word_list: Path, g2p_model: str, output_dictionary: Path
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def align_corpus(
        self,
        *,
        corpus_dir: Path,
        pronunciation_dict: str,
        acoustic_model: str,
        output_dir: Path,
    ) -> None:
        raise NotImplementedError
