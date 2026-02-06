from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from karaoker.utils import run_checked


_DEFAULT_ALIGN_BEAM = 500
_DEFAULT_ALIGN_RETRY_BEAM = 2500
_DEFAULT_ADAPTED_MODEL_NAME = "karaoker_adapted_acoustic_model.zip"


def run_mfa_g2p(
    *,
    mfa: str,
    word_list: Path,
    g2p_model: str,
    output_dictionary: Path,
) -> None:
    """Generate a pronunciation dictionary from `word_list` using an MFA G2P model."""
    output_dictionary.parent.mkdir(parents=True, exist_ok=True)
    exe = shutil.which(mfa) or mfa
    exe_bin = str(Path(exe).resolve().parent)

    default_root = Path(__file__).resolve().parents[3] / ".mfa"
    mfa_root = Path(os.environ.get("MFA_ROOT_DIR", str(default_root)))
    mfa_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "g2p",
        str(word_list),
        str(g2p_model),
        str(output_dictionary),
        "--overwrite",
    ]
    run_checked(
        cmd,
        env={
            "MFA_ROOT_DIR": str(mfa_root),
            "PATH": exe_bin + ":" + os.environ.get("PATH", ""),
        },
    )


def run_mfa_align(
    *,
    mfa: str,
    input_wav: Path,
    transcript_spaced_kana: Path,
    pronunciation_dict: str | Path | None,
    acoustic_model: str,
    output_textgrid: Path,
) -> None:
    """
    Align one wav + transcript with MFA and write a TextGrid.

    `transcript_spaced_kana` should already be tokenized (spaces between kana). Provide a
    dictionary + acoustic model that match.
    """
    output_textgrid.parent.mkdir(parents=True, exist_ok=True)
    exe = shutil.which(mfa) or mfa
    exe_bin = str(Path(exe).resolve().parent)

    with tempfile.TemporaryDirectory(prefix="karaoker_mfa_") as td:
        td_path = Path(td)
        # MFA defaults to `~/Documents/MFA`. Keep everything under the repo unless overridden.
        default_root = Path(__file__).resolve().parents[3] / ".mfa"
        mfa_root = Path(os.environ.get("MFA_ROOT_DIR", str(default_root)))
        mfa_root.mkdir(parents=True, exist_ok=True)

        corpus = td_path / "corpus"
        out_dir = td_path / "out"
        corpus.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        utt = corpus / "utt"
        wav_path = utt.with_suffix(".wav")
        lab_path = utt.with_suffix(".lab")
        wav_path.write_bytes(input_wav.read_bytes())
        lab_path.write_text(
            transcript_spaced_kana.read_text(encoding="utf-8"), encoding="utf-8"
        )

        dict_arg = (
            str(pronunciation_dict) if pronunciation_dict is not None else "japanese_mfa"
        )

        # Adapt the acoustic model to the input audio before aligning.
        adapted_model = td_path / _DEFAULT_ADAPTED_MODEL_NAME
        cmd_adapt = [
            exe,
            "adapt",
            str(corpus),
            dict_arg,
            str(acoustic_model),
            str(adapted_model),
            "--clean",
            "--single_speaker",
        ]
        run_checked(
            cmd_adapt,
            env={
                "MFA_ROOT_DIR": str(mfa_root),
                "PATH": exe_bin + ":" + os.environ.get("PATH", ""),
            },
        )
        if not adapted_model.exists():
            raise FileNotFoundError(
                f"MFA adapt produced no acoustic model at: {adapted_model}"
            )

        cmd = [
            exe,
            "align",
            str(corpus),
            dict_arg,
            str(adapted_model),
            str(out_dir),
            "--clean",
            "--beam",
            str(_DEFAULT_ALIGN_BEAM),
            "--retry_beam",
            str(_DEFAULT_ALIGN_RETRY_BEAM),
        ]
        run_checked(
            cmd,
            env={
                "MFA_ROOT_DIR": str(mfa_root),
                "PATH": exe_bin + ":" + os.environ.get("PATH", ""),
            },
        )

        # MFA usually writes `<utt_id>.TextGrid` somewhere under the output dir.
        produced = next(out_dir.glob("**/utt.TextGrid"), None)
        if produced is None:
            # Try any TextGrid.
            produced = next(out_dir.glob("**/*.TextGrid"), None)
        if produced is None:
            raise FileNotFoundError(
                "MFA produced no TextGrid; check MFA logs and model/dict."
            )
        output_textgrid.write_bytes(produced.read_bytes())


def run_mfa_align_corpus(
    *,
    mfa: str,
    corpus_dir: Path,
    pronunciation_dict: str | Path,
    acoustic_model: str,
    output_dir: Path,
) -> None:
    """Align a corpus directory containing `<utt_id>.wav` + `<utt_id>.lab` pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exe = shutil.which(mfa) or mfa
    exe_bin = str(Path(exe).resolve().parent)

    default_root = Path(__file__).resolve().parents[3] / ".mfa"
    mfa_root = Path(os.environ.get("MFA_ROOT_DIR", str(default_root)))
    mfa_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="karaoker_mfa_") as td:
        adapted_model = Path(td) / _DEFAULT_ADAPTED_MODEL_NAME

        # Adapt the acoustic model to this corpus before aligning.
        cmd_adapt = [
            exe,
            "adapt",
            str(corpus_dir),
            str(pronunciation_dict),
            str(acoustic_model),
            str(adapted_model),
            "--clean",
            "--single_speaker",
        ]
        run_checked(
            cmd_adapt,
            env={
                "MFA_ROOT_DIR": str(mfa_root),
                "PATH": exe_bin + ":" + os.environ.get("PATH", ""),
            },
        )
        if not adapted_model.exists():
            raise FileNotFoundError(
                f"MFA adapt produced no acoustic model at: {adapted_model}"
            )

        cmd = [
            exe,
            "align",
            str(corpus_dir),
            str(pronunciation_dict),
            str(adapted_model),
            str(output_dir),
            "--clean",
            "--single_speaker",
            "--beam",
            str(_DEFAULT_ALIGN_BEAM),
            "--retry_beam",
            str(_DEFAULT_ALIGN_RETRY_BEAM),
        ]
        run_checked(
            cmd,
            env={
                "MFA_ROOT_DIR": str(mfa_root),
                "PATH": exe_bin + ":" + os.environ.get("PATH", ""),
            },
        )
