"""Dead strips between the virtual detector's modules."""

from __future__ import annotations

import dataclasses

import pytest

from diffraction import (
    PANEL_COLUMNS,
    PANEL_ROWS,
    DiffractionStyle,
    _encode,
    _gap_positions,
    build,
)
from margin import render

STYLE = DiffractionStyle()

# A circular-finder code and a rounded-finder code with six alignments.
URLS = [
    "https://www.diamond.ac.uk",
    "https://www.diamond.ac.uk/Instruments/Mx/MX-Data-Analysis/"
    "Interpreting-Experiment-Feedback.html",
]

# Render scale in pixels per module; an integer so module centres and
# boundaries land on known pixels.
SCALE = 10

# Black strips, so any pixel a strip reaches is unmistakable.
BLACK_GAPS = dataclasses.replace(STYLE, panel_gap_colour="#000")


# The joins of an Eiger2 XE 16M, 4 columns by 8 rows of modules, land on
# whole module boundaries strictly inside the symbol.
@pytest.mark.parametrize("panels", [PANEL_COLUMNS, PANEL_ROWS])
@pytest.mark.parametrize("size", [21, 33, 53])
def test_gaps_sit_on_module_boundaries_inside_the_symbol(
    size: int, panels: int
) -> None:
    positions = _gap_positions(size, panels)
    assert len(positions) == panels - 1, "wrong number of joins"
    assert all(0 < p < size for p in positions), f"join outside symbol: {positions}"
    assert positions == sorted(set(positions)), f"joins collide: {positions}"


# A strip must never reach a module's sampling point: every module centre
# reads the same with and without black strips.
@pytest.mark.parametrize("url", URLS)
def test_gaps_leave_every_sampling_point_unchanged(url: str) -> None:
    grid = _encode(url)
    with_gaps = render(build(url, style=BLACK_GAPS).svg, SCALE)
    without = render(
        build(url, style=dataclasses.replace(STYLE, panel_gaps=False)).svg, SCALE
    )
    offset = STYLE.quiet_zone
    for y in range(grid.size):
        for x in range(grid.size):
            py = (offset + y) * SCALE + SCALE // 2
            px = (offset + x) * SCALE + SCALE // 2
            assert with_gaps[py, px] == without[py, px], (
                f"strip reached the sampling point of module {(x, y)}"
            )


# The strips must actually be drawn, on the joins, or the test above
# passes vacuously.
@pytest.mark.parametrize("url", URLS)
def test_gaps_are_drawn_on_the_joins(url: str) -> None:
    grid = _encode(url)
    image = render(build(url, style=BLACK_GAPS).svg, SCALE)
    offset = STYLE.quiet_zone
    column = (offset + _gap_positions(grid.size, PANEL_COLUMNS)[0]) * SCALE
    # Sample down the first vertical join, away from the finders' squares.
    rows = range((offset + 9) * SCALE, (offset + grid.size - 9) * SCALE)
    # A strip a few pixels wide straddling a pixel edge renders anti-aliased.
    assert min(image[r, column] for r in rows) < 0.5, "no strip on the first join"


# Strips stop at the finder and alignment squares and at the quiet zone,
# so neither the patterns' light gaps nor the border are cut.
@pytest.mark.parametrize("url", URLS)
def test_gaps_stay_out_of_the_patterns_and_quiet_zone(url: str) -> None:
    grid = _encode(url)
    image = render(build(url, style=BLACK_GAPS).svg, SCALE)
    reference = render(
        build(url, style=dataclasses.replace(BLACK_GAPS, panel_gaps=False)).svg, SCALE
    )
    offset = STYLE.quiet_zone
    for centres, half in ((grid.finders, 3), (grid.alignments, 2)):
        for x, y in centres:
            top = (offset + y - half) * SCALE
            left = (offset + x - half) * SCALE
            side = (2 * half + 1) * SCALE
            region = (slice(top, top + side), slice(left, left + side))
            assert (image[region] == reference[region]).all(), (
                f"strip crossed the pattern at {(x, y)}"
            )
    border = offset * SCALE - 1
    assert image[:border].min() == 1.0, "strip reached the quiet zone"
    assert image[:, :border].min() == 1.0, "strip reached the quiet zone"
