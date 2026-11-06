"""Tangential streaking of the data spots."""

from __future__ import annotations

import dataclasses
import math
import re

import pytest

from diffraction import STREAK_CAP, DiffractionStyle, _encode, build

URL = "https://www.diamond.ac.uk/Instruments/Mx/MX-Data-Analysis/Interpreting-Experiment-Feedback.html"
STYLE = DiffractionStyle()

SPOT_COLOUR = re.escape(STYLE.spot_colour)
ELLIPSE = re.compile(
    r'<ellipse cx="([\d.]+)" cy="([\d.]+)" rx="([\d.]+)" ry="([\d.]+)" '
    rf'transform="rotate\(([-\d.]+) [\d.]+ [\d.]+\)" fill="{SPOT_COLOUR}"/>'
)
CIRCLE = re.compile(
    rf'<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)" fill="{SPOT_COLOUR}"/>'
)


def _spots(svg: str) -> dict[tuple[str, str], tuple[float, float, float | None]]:
    """Map each data spot's centre to (half-length, half-width, angle); a
    circle has equal halves and no angle."""
    spots: dict[tuple[str, str], tuple[float, float, float | None]] = {}
    for cx, cy, r in CIRCLE.findall(svg):
        spots[(cx, cy)] = (float(r), float(r), None)
    for cx, cy, rx, ry, angle in ELLIPSE.findall(svg):
        spots[(cx, cy)] = (float(rx), float(ry), float(angle))
    return spots


# The across-radius width of every spot must equal its round diameter, so
# the module's own sampling point is covered exactly as before streaking.
def test_streaks_keep_each_spot_width() -> None:
    round_spots = _spots(
        build(URL, style=dataclasses.replace(STYLE, streaks=False)).svg
    )
    streaked = _spots(build(URL).svg)
    assert round_spots.keys() == streaked.keys(), "streaks moved or dropped spots"
    for centre, (half_length, half_width, _) in streaked.items():
        assert half_width == round_spots[centre][1], f"spot at {centre} narrowed"
        assert half_length >= half_width, f"spot at {centre} shorter than wide"


# No streak may reach a neighbouring module's sampling point, one module
# away, unless the round spot was already that large.
def test_streaks_stop_at_the_cap() -> None:
    for centre, (half_length, half_width, _) in _spots(build(URL).svg).items():
        limit = max(half_width, STREAK_CAP / 2) + 1e-3
        assert half_length <= limit, f"spot at {centre} streaked past the cap"


# Streaks run across the radius, as mosaic smearing does: the long axis is
# perpendicular to the line from the code centre, within the 0.1°
# rounding of the markup.
def test_streaks_run_across_the_radius() -> None:
    grid = _encode(URL)
    ellipses = [
        (float(cx), float(cy), angle)
        for (cx, cy), (_, _, angle) in _spots(build(URL).svg).items()
        if angle is not None
    ]
    assert ellipses, "no spot was streaked"
    centre = grid.centre + 0.5
    for cx, cy, angle in ellipses:
        radial = math.degrees(math.atan2(cy - centre, cx - centre))
        offset = (angle - radial) % 180.0
        assert offset == pytest.approx(90.0, abs=0.06), (
            f"spot at {(cx, cy)} streaks {offset:.2f}° off the radius"
        )


# Streak length grows with radius, so outer spots are more elongated than
# inner ones, as on a rotation image.
def test_outer_spots_are_more_elongated() -> None:
    grid = _encode(URL)
    centre = grid.centre + 0.5
    inner, outer = [], []
    for (cx, cy), (half_length, half_width, _) in _spots(build(URL).svg).items():
        r = math.hypot(float(cx) - centre, float(cy) - centre) / grid.corner_distance
        aspect = half_length / half_width
        (inner if r < 0.3 else outer if r > 0.7 else []).append(aspect)
    assert sum(outer) / len(outer) > sum(inner) / len(inner), (
        "outer spots are not more elongated than inner ones"
    )
