from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AlignerProvider(ABC):
    """
    Backend interface for forced aligners.

    Notes:
    - Some backends may not support `g2p` (auto dictionary generation) and can raise
      `NotImplementedError`.
    - `pronunciation_dict`/`acoustic_model` are backend-defined strings (e.g. MFA model names vs
      filesystem paths for other tools).
    """

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
