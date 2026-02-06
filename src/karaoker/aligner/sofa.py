from __future__ import annotations

from pathlib import Path

from karaoker.aligner.base import AlignerProvider
from karaoker.external.sofa import run_sofa_align_corpus


class SofaAlignerProvider(AlignerProvider):
    def __init__(self, *, sofa_python: str, sofa_root: Path) -> None:
        self._sofa_python = sofa_python
        self._sofa_root = sofa_root

    def g2p(
        self, *, word_list: Path, g2p_model: str, output_dictionary: Path
    ) -> None:
        raise NotImplementedError(
            "SOFA aligner does not currently support karaoker's MFA-style G2P flow. "
            "Please provide a SOFA-compatible dictionary via --sofa-dict."
        )

    def align_corpus(
        self,
        *,
        corpus_dir: Path,
        pronunciation_dict: str,
        acoustic_model: str,
        output_dir: Path,
    ) -> None:
        # For SOFA, `pronunciation_dict` is the dictionary path, and `acoustic_model` is
        # the checkpoint path. Keep the base interface to avoid ripple effects.
        run_sofa_align_corpus(
            sofa_python=self._sofa_python,
            sofa_root=self._sofa_root,
            corpus_dir=corpus_dir,
            dictionary=Path(pronunciation_dict),
            ckpt=Path(acoustic_model),
            output_dir=output_dir,
            in_format="lab",
            out_format="TextGrid",
        )
