from karaoker.mapping import (
    needleman_wunsch,
    map_kana_events_to_script,
    to_spaced_kana_with_units,
)


def test_needleman_wunsch_deletion():
    ali = needleman_wunsch(["ア", "イ", "ウ"], ["ア", "ウ"])
    assert ali.ref_to_out == [0, None, 1]
    assert ali.out_to_ref == [0, 2]


def test_map_kana_events_to_script_units_basic():
    script = "今日はキャットです"
    spaced, units = to_spaced_kana_with_units(script, output="katakana")
    ref_tokens = spaced.split()
    assert ref_tokens  # sanity

    # Fake "aligned" kana events with monotonically increasing times.
    events = []
    t = 0.0
    for i, tok in enumerate(ref_tokens):
        events.append(
            {
                "i": i,
                "start": round(t, 4),
                "end": round(t + 0.1, 4),
                "text": tok,
                "tier": "kana",
            }
        )
        t += 0.1

    enriched, timed_units = map_kana_events_to_script(
        script_text=script,
        script_units=units,
        kana_events=events,
    )

    # Each output event should map back to a script unit with a valid substring.
    for e in enriched:
        assert "script_unit_i" in e
        assert "script_text" in e
        assert script[e["script_char_start"] : e["script_char_end"]] == e["script_text"]

    # Timed units should cover at least the kana-carrying units.
    assert timed_units
    assert all(u["end"] >= u["start"] for u in timed_units)
