from pathlib import Path

from karaoker.textgrid_parser import load_textgrid, textgrid_to_kana_events


def test_textgrid_fallback_parser(tmp_path: Path):
    tg = tmp_path / "a.TextGrid"
    tg.write_text(
        "\n".join(
            [
                'File type = "ooTextFile"',
                'Object class = "TextGrid"',
                "",
                "xmin = 0",
                "xmax = 1.0",
                "tiers? <exists>",
                "size = 1",
                "item []:",
                "    item [1]:",
                '        class = "IntervalTier"',
                '        name = "kana"',
                "        xmin = 0",
                "        xmax = 1.0",
                "        intervals: size = 2",
                "        intervals [1]:",
                "            xmin = 0",
                "            xmax = 0.5",
                '            text = "サ"',
                "        intervals [2]:",
                "            xmin = 0.5",
                "            xmax = 1.0",
                '            text = "ン"',
            ]
        ),
        encoding="utf-8",
    )
    tiers = load_textgrid(tg)
    assert "kana" in tiers
    events = textgrid_to_kana_events(tg)
    assert [e["text"] for e in events] == ["サ", "ン"]


def test_textgrid_parser_ignores_sp_and_ap_markers(tmp_path: Path) -> None:
    tg = tmp_path / "a.TextGrid"
    tg.write_text(
        "\n".join(
            [
                'File type = "ooTextFile"',
                'Object class = "TextGrid"',
                "",
                "xmin = 0",
                "xmax = 1.0",
                "tiers? <exists>",
                "size = 1",
                "item []:",
                "    item [1]:",
                '        class = "IntervalTier"',
                '        name = "words"',
                "        xmin = 0",
                "        xmax = 1.0",
                "        intervals: size = 4",
                "        intervals [1]:",
                "            xmin = 0",
                "            xmax = 0.25",
                '            text = "SP"',
                "        intervals [2]:",
                "            xmin = 0.25",
                "            xmax = 0.5",
                '            text = "サ"',
                "        intervals [3]:",
                "            xmin = 0.5",
                "            xmax = 0.75",
                '            text = "AP"',
                "        intervals [4]:",
                "            xmin = 0.75",
                "            xmax = 1.0",
                '            text = "ン"',
            ]
        ),
        encoding="utf-8",
    )
    events = textgrid_to_kana_events(tg)
    assert [e["text"] for e in events] == ["サ", "ン"]
