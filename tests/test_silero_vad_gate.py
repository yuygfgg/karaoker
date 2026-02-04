from __future__ import annotations

import array

from karaoker.external.silero_vad import zero_outside_segments_s16


def test_zero_outside_segments_s16_basic() -> None:
    pcm = array.array("h", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    out = zero_outside_segments_s16(pcm, segments=[(2, 5), (7, 9)])
    assert out.tolist() == [0, 0, 3, 4, 5, 0, 0, 8, 9, 0]


def test_zero_outside_segments_s16_clamps() -> None:
    pcm = array.array("h", [1, 2, 3, 4])
    out = zero_outside_segments_s16(pcm, segments=[(-10, 2), (3, 100), (2, 2)])
    assert out.tolist() == [1, 2, 0, 4]

