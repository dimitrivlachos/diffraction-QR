"""Ice ring geometry and the ice ring layer."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

from diffraction import (
    ICE_SPOT_CAP,
    DiffractionStyle,
    _encode,
    _ice_ring_radii,
    build,
    ice_d_spacings,
    resolution,
    ring_radius,
)
from margin import render

# Ice Ih d-spacings in Å down to 1.5 Å, from DIALS dev20260922:
#   dials.python: crystal.symmetry(unit_cell=(4.498, 4.498, 7.338, 90, 90,
#   120), space_group=194).build_miller_set(anomalous_flag=False,
#   d_min=1.5).d_spacings(), distinct values rounded to 4 dp.
DIALS_ICE_D_SPACINGS = [
    3.8954, 3.669, 3.4406, 2.6708, 2.249, 2.0715, 1.9477,
    1.9174, 1.8825, 1.8345, 1.7203, 1.6597, 1.5237,
]  # fmt: skip

URL = "https://www.diamond.ac.uk"
STYLE = DiffractionStyle()


# The rings must be the ones DIALS masks, absences included, or the
# pattern is not ice.
def test_ice_d_spacings_match_dials() -> None:
    ours = [round(d, 4) for d in ice_d_spacings(1.5)]
    assert ours == DIALS_ICE_D_SPACINGS, f"ice rings differ from DIALS: {ours}"


# ring_radius inverts resolution, so a ring drawn at its radius sits at
# the resolution it stands for.
@pytest.mark.parametrize("d", DIALS_ICE_D_SPACINGS)
def test_ring_radius_inverts_resolution(d: float) -> None:
    r = ring_radius(d, STYLE)
    assert r is not None, f"{d} Å has no radius"
    assert resolution(r, STYLE) == pytest.approx(d, rel=1e-12), "round trip drifted"


# Below λ/2 Bragg's law has no solution, and from 2θ = 90° a flat
# detector facing the beam records nothing; neither may yield a ring.
@pytest.mark.parametrize("d", [0.4, STYLE.wavelength_a / 2.0, 0.69])
def test_unrecordable_resolutions_have_no_ring(d: float) -> None:
    assert ring_radius(d, STYLE) is None, f"{d} Å given a ring radius"


# A longer wavelength must push a ring outwards; this is the knob the
# wavelength constant exists for.
def test_longer_wavelength_moves_rings_outwards() -> None:
    longer = dataclasses.replace(STYLE, wavelength_a=1.2)
    assert ring_radius(3.8954, longer) > ring_radius(3.8954, STYLE), "ring moved in"


# The rings stop at the resolution of the symbol's corners: the last one
# drawn lies inside the corner, the first one left out lies beyond it.
def test_rings_stop_at_the_corner_resolution() -> None:
    grid = _encode(URL)
    radii = _ice_ring_radii(grid, STYLE)
    r_edge = (grid.corner_distance + np.sqrt(0.5)) / grid.corner_distance
    assert max(radii) <= r_edge, "a ring lies beyond the symbol corner"
    next_d = DIALS_ICE_D_SPACINGS[len(radii)]
    assert ring_radius(next_d, STYLE) > r_edge, "a reachable ring was dropped"


# A circular-finder code with one alignment pattern, and a rounded-finder
# code with six.
CLIP_URLS = [
    URL,
    "https://www.diamond.ac.uk/Instruments/Mx/MX-Data-Analysis/"
    "Interpreting-Experiment-Feedback.html",
]

# Render scale for the clipping tests, in pixels per module.
CLIP_SCALE = 10.0


def _flooded_render(url: str) -> np.ndarray:
    """Render ``url`` with black rings that cover the symbol and quiet zone,
    so any pixel the clip fails to protect turns dark.

    Each stroke stays just narrower than its ring's diameter, since cairosvg
    draws nothing for a wider one.
    """
    grid = _encode(url)
    width = 1.9 * min(_ice_ring_radii(grid, STYLE)) * grid.corner_distance
    flood = dataclasses.replace(STYLE, ice_ring_colour="#000", ice_ring_width=width)
    return render(build(url, style=flood).svg, CLIP_SCALE)


# The flood must actually darken the symbol, or the two clip tests below
# would pass with no clip at all.
@pytest.mark.parametrize("url", CLIP_URLS)
def test_flooding_rings_darken_the_symbol(url: str) -> None:
    image = _flooded_render(url)
    inner = int((STYLE.quiet_zone + 8) * CLIP_SCALE)
    assert image[inner:-inner, inner:-inner].mean() < 0.2, "flood did not render"


# The quiet zone must stay pure white, or the decoder loses the code's
# boundary; the clip keeps the rings inside the symbol.
@pytest.mark.parametrize("url", CLIP_URLS)
def test_quiet_zone_stays_white_under_flooding_rings(url: str) -> None:
    image = _flooded_render(url)
    border = int(STYLE.quiet_zone * CLIP_SCALE) - 1
    edges = [image[:border], image[-border:], image[:, :border], image[:, -border:]]
    assert all(edge.min() == 1.0 for edge in edges), "ice ring tinted the quiet zone"


# Light gaps in the finder and alignment patterns must stay pure white, or
# their run ratios break; the clip cuts the pattern squares out of the
# rings. Each gap is sampled midway through it, on all four sides.
@pytest.mark.parametrize("url", CLIP_URLS)
def test_pattern_light_gaps_stay_white_under_flooding_rings(url: str) -> None:
    grid = _encode(url)
    image = _flooded_render(url)
    offset = STYLE.quiet_zone + 0.5
    patterns = [(c, 2.0) for c in grid.finders] + [(c, 1.0) for c in grid.alignments]
    for (x, y), gap in patterns:
        for dx, dy in ((gap, 0), (-gap, 0), (0, gap), (0, -gap)):
            px = int((offset + x + dx) * CLIP_SCALE)
            py = int((offset + y + dy) * CLIP_SCALE)
            value = image[py, px]
            assert value == 1.0, f"gap of pattern at {(x, y)} is {value:.3f}"


# Ice rings only ever grow spots, never past the cap unless the spot was
# already larger, and some spots do sit on a ring.
def test_ice_rings_only_grow_spots_up_to_the_cap() -> None:
    def spot_radii(svg: str) -> list[float]:
        return [float(r) for r in re.findall(r'r="([\d.]+)" fill="#202945"/>', svg)]

    # Streaks off on both sides, so every spot is a circle with one radius.
    round_spots = dataclasses.replace(STYLE, streak=0.0)
    plain = spot_radii(
        build(URL, style=dataclasses.replace(round_spots, ice_rings=False)).svg
    )
    iced = spot_radii(build(URL, style=round_spots).svg)
    assert len(plain) == len(iced), "ice rings changed the number of spots"
    pairs = list(zip(plain, iced, strict=True))
    assert all(after >= before for before, after in pairs), "a spot shrank"
    assert all(
        after <= max(before, ICE_SPOT_CAP / 2) + 1e-3 for before, after in pairs
    ), "a spot grew past the cap"
    assert any(after > before for before, after in pairs), "no spot sits on a ring"
