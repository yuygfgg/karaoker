from __future__ import annotations

from pathlib import Path

from karaoker.sofa_dict import (
    detect_sofa_dictionary_kind,
    generate_sofa_kana_dictionary,
    load_sofa_dictionary,
)


def test_detect_sofa_dictionary_kind(tmp_path: Path) -> None:
    romaji = tmp_path / "romaji.dict"
    romaji.write_text("AP\tAP\nka\tk a\n", encoding="utf-8")
    assert detect_sofa_dictionary_kind(romaji) == "romaji"

    kana = tmp_path / "kana.dict"
    kana.write_text("ア\ta\n", encoding="utf-8")
    assert detect_sofa_dictionary_kind(kana) == "kana"


def test_generate_sofa_kana_dictionary_from_romaji(tmp_path: Path) -> None:
    romaji_dict = tmp_path / "romaji.dict"
    romaji_dict.write_text(
        "\n".join(
            [
                "kya\tky a",
                "to\tt o",
                "o\to",
                "cl\tcl",
                "N\tN",
                "ti\tt i",
                "she\tsh e",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "kana.dict"
    generate_sofa_kana_dictionary(
        kana_tokens=["キャッ", "ト", "オー", "ン", "ティ", "シェ"],
        romaji_dictionary=romaji_dict,
        output_dictionary=out,
    )

    got = load_sofa_dictionary(out)
    assert got["キャッ"] == ["ky", "a", "cl"]
    assert got["ト"] == ["t", "o"]
    assert got["オー"] == ["o", "o"]
    assert got["ン"] == ["N"]
    assert got["ティ"] == ["t", "i"]
    assert got["シェ"] == ["sh", "e"]
