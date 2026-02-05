from __future__ import annotations

from pathlib import Path

from karaoker.aligner.base import AlignerProvider
from karaoker.external.mfa import run_mfa_align_corpus, run_mfa_g2p


class MfaAlignerProvider(AlignerProvider):
    def __init__(self, *, mfa: str) -> None:
        self._mfa = mfa

    def g2p(
        self, *, word_list: Path, g2p_model: str, output_dictionary: Path
    ) -> None:
        run_mfa_g2p(
            mfa=self._mfa,
            word_list=word_list,
            g2p_model=g2p_model,
            output_dictionary=output_dictionary,
        )

    def align_corpus(
        self,
        *,
        corpus_dir: Path,
        pronunciation_dict: str,
        acoustic_model: str,
        output_dir: Path,
    ) -> None:
        run_mfa_align_corpus(
            mfa=self._mfa,
            corpus_dir=corpus_dir,
            pronunciation_dict=pronunciation_dict,
            acoustic_model=acoustic_model,
            output_dir=output_dir,
        )
