from __future__ import annotations

from pathlib import Path

import pytest

from karaoker.external.sofa import run_sofa_align_corpus


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")


def test_run_sofa_align_corpus_copies_textgrids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sofa_root = tmp_path / "SOFA"
    sofa_root.mkdir(parents=True, exist_ok=True)
    _touch(sofa_root / "infer.py")

    dictionary = tmp_path / "dict.txt"
    _touch(dictionary)

    ckpt = tmp_path / "model.ckpt"
    _touch(ckpt)

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "utt_0000.wav").write_bytes(b"RIFFxxxxWAVE")
    (corpus_dir / "utt_0000.lab").write_text("a b c\n", encoding="utf-8")

    output_dir = tmp_path / "out"

    calls: list[tuple[list[str], str | None]] = []

    def fake_run_checked(
        cmd: list[str], *, cwd: str | None = None, env: object | None = None
    ) -> None:
        calls.append((cmd, cwd))
        folder = Path(cmd[cmd.index("--folder") + 1])
        singer_dir = folder / "karaoker"

        # The wrapper should have copied wav/lab pairs before invoking SOFA.
        assert (singer_dir / "utt_0000.wav").exists()
        assert (singer_dir / "utt_0000.lab").exists()

        tg_dir = singer_dir / "TextGrid"
        tg_dir.mkdir(parents=True, exist_ok=True)
        (tg_dir / "utt_0000.TextGrid").write_text("dummy\n", encoding="utf-8")

    monkeypatch.setattr("karaoker.external.sofa.run_checked", fake_run_checked)

    run_sofa_align_corpus(
        sofa_python="python3",
        sofa_root=sofa_root,
        corpus_dir=corpus_dir,
        dictionary=dictionary,
        ckpt=ckpt,
        output_dir=output_dir,
    )

    assert (output_dir / "utt_0000.TextGrid").read_text(encoding="utf-8") == "dummy\n"
    assert calls
    cmd, cwd = calls[0]
    assert cmd[:2] == ["python3", "infer.py"]
    assert cwd == str(sofa_root)


def test_run_sofa_align_corpus_raises_when_no_textgrids_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sofa_root = tmp_path / "SOFA"
    sofa_root.mkdir(parents=True, exist_ok=True)
    _touch(sofa_root / "infer.py")

    dictionary = tmp_path / "dict.txt"
    _touch(dictionary)

    ckpt = tmp_path / "model.ckpt"
    _touch(ckpt)

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "utt_0000.wav").write_bytes(b"RIFFxxxxWAVE")
    (corpus_dir / "utt_0000.lab").write_text("a b c\n", encoding="utf-8")

    def fake_run_checked(
        cmd: list[str], *, cwd: str | None = None, env: object | None = None
    ) -> None:
        return

    monkeypatch.setattr("karaoker.external.sofa.run_checked", fake_run_checked)

    with pytest.raises(FileNotFoundError, match="produced no TextGrid"):
        run_sofa_align_corpus(
            sofa_python="python3",
            sofa_root=sofa_root,
            corpus_dir=corpus_dir,
            dictionary=dictionary,
            ckpt=ckpt,
            output_dir=tmp_path / "out",
        )
