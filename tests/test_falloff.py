"""Spot diameter falloff with resolution."""

from __future__ import annotations

import dataclasses
import math
from itertools import pairwise

import pytest

from diffraction import DiffractionStyle, _falloff_diameter, wilson_intensity

STYLE = DiffractionStyle()

# Normalised radii from the centre to a corner module.
RADII = [i / 20 for i in range(21)]


# exp(-B/(4d²)) at B = 20 Å², d = 2 Å is exp(-1.25), worked by hand; the
# intensity at infinite d is exactly 1.
def test_wilson_intensity_matches_hand_values() -> None:
    assert wilson_intensity(2.0, 20.0) == pytest.approx(math.exp(-1.25), rel=1e-12), (
        "Wilson exponent wrong at d = 2 Å"
    )
    assert wilson_intensity(math.inf, 20.0) == 1.0, "intensity at d = ∞ is not 1"


# B shapes the curve but must never move its ends: the centre keeps the
# centre diameter and the corner module the edge diameter, so the
# smallest spots, which set the margin, do not shrink with B.
@pytest.mark.parametrize("b_factor", [5.0, 20.0, 60.0])
def test_wilson_falloff_keeps_its_end_points(b_factor: float) -> None:
    style = dataclasses.replace(STYLE, wilson_b=b_factor)
    assert _falloff_diameter(0.0, style) == pytest.approx(
        style.spot_diameter_centre, abs=1e-12
    ), "centre diameter moved"
    assert _falloff_diameter(1.0, style) == pytest.approx(
        style.spot_diameter_edge, abs=1e-12
    ), "corner diameter moved"


# Spots must shrink steadily outwards, as intensity does on a real image.
@pytest.mark.parametrize("falloff", ["wilson", "radial"])
def test_falloff_decreases_monotonically(falloff: str) -> None:
    style = dataclasses.replace(STYLE, falloff=falloff)
    diameters = [_falloff_diameter(r, style) for r in RADII]
    assert all(a > b for a, b in pairwise(diameters)), (
        f"{falloff} falloff is not strictly decreasing"
    )


# A larger B decays faster, so every interior spot is smaller.
def test_larger_b_factor_shrinks_interior_spots() -> None:
    low = dataclasses.replace(STYLE, wilson_b=10.0)
    high = dataclasses.replace(STYLE, wilson_b=40.0)
    assert all(
        _falloff_diameter(r, high) < _falloff_diameter(r, low) for r in RADII[1:-1]
    ), "larger B did not shrink the interior spots"


# With no decay there is nothing to rescale over; every spot takes the
# centre size rather than dividing by zero.
def test_zero_b_factor_gives_uniform_spots() -> None:
    style = dataclasses.replace(STYLE, wilson_b=0.0)
    assert {_falloff_diameter(r, style) for r in RADII} == {
        style.spot_diameter_centre
    }, "B = 0 did not give uniform spots"
