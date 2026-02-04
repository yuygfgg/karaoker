from __future__ import annotations

from pathlib import Path

from karaoker.external.audio_separator import _pick_output


def test_pick_output_prefers_stem(tmp_path: Path) -> None:
    out = tmp_path / "vocals.wav"
    out.write_bytes(b"old")

    (tmp_path / "song_(instrumental).wav").write_bytes(b"inst")
    (tmp_path / "song_(vocals).wav").write_bytes(b"voc")

    _pick_output(out, stem="vocals")
    assert out.read_bytes() == b"voc"

