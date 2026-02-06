from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from karaoker.utils import run_checked


def run_sofa_align_corpus(
    *,
    sofa_python: str,
    sofa_root: Path,
    corpus_dir: Path,
    dictionary: Path,
    ckpt: Path,
    output_dir: Path,
    in_format: str = "lab",
    out_format: str = "TextGrid",
) -> None:
    """
    Run SOFA (Singing-Oriented Forced Aligner) on a corpus and collect TextGrid outputs.

    SOFA's reference CLI (per upstream README) is:
      python infer.py --ckpt <ckpt> --folder <segments_dir> --dictionary <dict> \\
        --out_formats TextGrid

    Notes:
    - SOFA expects the `--folder` to contain per-singer subfolders. We build a temporary
      `segments/karaoker/` directory and copy `<utt_id>.wav` + `<utt_id>.lab` pairs into it.
    - SOFA writes outputs into a format-named folder near the input wavs; we then copy any
      produced `*.TextGrid` files into `output_dir` (flattened).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sofa_root = sofa_root.expanduser().resolve()
    if not sofa_root.exists():
        raise FileNotFoundError(f"SOFA root dir not found: {sofa_root}")

    infer_py = sofa_root / "infer.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"SOFA infer.py not found at: {infer_py}")

    dictionary = dictionary.expanduser().resolve()
    if not dictionary.exists():
        raise FileNotFoundError(f"SOFA dictionary not found: {dictionary}")

    ckpt = ckpt.expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"SOFA checkpoint not found: {ckpt}")

    corpus_dir = corpus_dir.expanduser().resolve()
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus dir not found: {corpus_dir}")

    wavs = sorted(corpus_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in corpus: {corpus_dir}")

    with tempfile.TemporaryDirectory(prefix="karaoker_sofa_") as td:
        td_path = Path(td)
        segments_root = td_path / "segments"
        singer_dir = segments_root / "karaoker"
        singer_dir.mkdir(parents=True, exist_ok=True)

        # Copy wav/lab pairs into a structure SOFA expects.
        for wav in wavs:
            lab = wav.with_suffix(".lab")
            if not lab.exists():
                raise FileNotFoundError(f"Missing transcript for {wav.name}: {lab}")
            shutil.copyfile(wav, singer_dir / wav.name)
            shutil.copyfile(lab, singer_dir / lab.name)

        cmd = [
            sofa_python,
            "infer.py",
            "--ckpt",
            str(ckpt),
            "--folder",
            str(segments_root),
            "--dictionary",
            str(dictionary),
            "--in_format",
            str(in_format),
            "--out_formats",
            str(out_format),
        ]
        run_checked(cmd, cwd=str(sofa_root))

        produced: list[Path] = []
        for p in singer_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".textgrid":
                produced.append(p)
        if not produced:
            # Be a bit more permissive: search anywhere under segments_root.
            for p in segments_root.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".textgrid":
                    produced.append(p)

        if not produced:
            raise FileNotFoundError(
                "SOFA produced no TextGrid files. "
                "Check SOFA logs, dictionary/ckpt, and transcript format."
            )

        for tg in produced:
            shutil.copyfile(tg, output_dir / tg.name)
