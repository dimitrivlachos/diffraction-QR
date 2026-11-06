"""The centre logo: its clear disc, its budget and its refusals."""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from diffraction import (
    DEFAULT_LOGO,
    LOGO_FRACTION,
    LOGO_MAX_CLEARED,
    MAX_SPOT_RADIUS,
    SPECKLE_SCATTER,
    DiffractionStyle,
    _encode,
    _logo_clear_radius,
    _logo_fit,
    build,
    main,
)
from margin import MARGIN_FLOOR, blur_margin

LOGO = DiffractionStyle(logo=str(DEFAULT_LOGO))

SVG = "{http://www.w3.org/2000/svg}"

# The postcard's codes and the long version 12 URL, versions 3 to 12.
URLS = [
    "https://xia2.github.io",
    "https://dials.github.io/documentation/tutorials/index.html",
    "https://www.diamond.ac.uk/Instruments/Mx/MX-Data-Analysis/"
    "Interpreting-Experiment-Feedback.html",
    "https://dials.github.io/documentation/tutorials/"
    "processing_in_detail_betalactamase.html"
    "?from=qr&campaign=diamond-open-day-2026#integration-and-scaling",
]


def _largest_allowed(url: str) -> float:
    """Return the largest logo size, to two decimals, the code accepts."""
    grid = _encode(url)
    return max(
        f / 100
        for f in range(1, 100)
        if _logo_fit(grid, f / 100)[0] <= LOGO_MAX_CLEARED
        and _logo_fit(grid, f / 100)[1]
    )


# The default asset ships with the code, so --logo works with no path.
def test_default_logo_ships_with_the_generator() -> None:
    assert DEFAULT_LOGO.is_file(), f"{DEFAULT_LOGO} missing"


# The logo is nested as vector: a well-formed <svg> carrying the asset's
# viewBox and its path, not a raster image.
def test_logo_is_nested_as_vector() -> None:
    root = ET.fromstring(build(URLS[1], style=LOGO).svg)
    nested = [e for e in root.iter(f"{SVG}svg") if e is not root]
    assert len(nested) == 1, f"expected one nested logo, found {len(nested)}"
    assert nested[0].get("viewBox") == "0 0 400 400", "asset viewBox not carried"
    assert nested[0].find(f".//{SVG}path") is not None, "logo path missing"
    assert not list(root.iter(f"{SVG}image")), "logo embedded as an image"


# A logo only removes spots and speckle near it and adds itself: every
# other element is unchanged, and no spot centre is left within reach of
# the clear disc.
def test_logo_clears_only_the_spots_it_must() -> None:
    url = URLS[1]
    plain = build(url).svg.splitlines()
    with_logo = build(url, style=LOGO).svg.splitlines()
    added = [line for line in with_logo if line not in set(plain)]
    assert len(added) == 2 and added[0].startswith("<circle"), f"added: {added}"
    assert "<svg" in added[1], "logo element not added"

    grid = _encode(url)
    reach = _logo_clear_radius(grid, LOGO) + MAX_SPOT_RADIUS
    centre = grid.size / 2.0
    removed = [line for line in plain if line not in set(with_logo)]
    for line in removed:
        cx = float(line.split('cx="')[1].split('"')[0])
        cy = float(line.split('cy="')[1].split('"')[0])
        # Speckle sits up to SPECKLE_SCATTER off its module centre, and
        # centres are written to two decimals.
        assert math.hypot(cx - centre, cy - centre) < reach + SPECKLE_SCATTER + 0.01, (
            f"spot at {(cx, cy)} removed outside the clear disc"
        )
    assert removed, "no spot was cleared"


# The centre alignment pattern on version 7 and up is drawn over the logo,
# so its ring sits in the starburst.
def test_centre_alignment_is_drawn_over_the_logo() -> None:
    url = URLS[2]
    grid = _encode(url)
    assert (grid.centre, grid.centre) in grid.alignments, "no centre alignment"
    lines = build(url, style=LOGO).svg.splitlines()
    logo_at = next(i for i, line in enumerate(lines) if i and line.startswith("<svg "))
    ring = f'cx="{grid.centre + 0.5}" cy="{grid.centre + 0.5}" r="2"'
    ring_at = next(i for i, line in enumerate(lines) if ring in line)
    assert ring_at > logo_at, "centre alignment drawn under the logo"


# At the default size every code keeps the camera headroom the floor
# demands.
@pytest.mark.parametrize("url", URLS)
def test_default_logo_keeps_the_margin(url: str) -> None:
    sigma = min(blur_margin(build(url, seed=s, style=LOGO).svg, url) for s in (0, 1, 2))
    assert sigma >= MARGIN_FLOOR, f"default logo drops σ to {sigma:.3f}"


# The budget is the guard: the largest logo each code accepts must still
# decode with headroom, since past the cliff the margin gives no warning.
@pytest.mark.parametrize("url", URLS)
def test_largest_allowed_logo_keeps_the_margin(url: str) -> None:
    style = dataclasses.replace(LOGO, logo_fraction=_largest_allowed(url))
    sigma = min(
        blur_margin(build(url, seed=s, style=style).svg, url) for s in (0, 1, 2)
    )
    assert sigma >= MARGIN_FLOOR, (
        f"logo {style.logo_fraction} accepted but σ = {sigma:.3f}"
    )


# The default size fits every code from version 2 up, so --logo alone
# never needs a size.
@pytest.mark.parametrize("url", ["https://x.io/abc", *URLS])
def test_default_size_fits_every_code(url: str) -> None:
    assert _largest_allowed(url) >= LOGO_FRACTION, f"default too large for {url}"


# A logo over budget is refused, naming the largest size that fits, and
# that size is then accepted.
def test_oversized_logo_is_refused_with_the_size_that_fits() -> None:
    url = URLS[0]
    with pytest.raises(ValueError, match=r"takes at most 0\.\d+") as refusal:
        build(url, style=dataclasses.replace(LOGO, logo_fraction=0.45))
    largest = float(str(refusal.value).rsplit(" ", 1)[1])
    build(url, style=dataclasses.replace(LOGO, logo_fraction=largest))


# Alignment patterns sit about 26 modules apart however large the code,
# so on a big code the clear disc reaches one long before the share limit;
# that logo is refused even though its cleared share is in budget.
def test_logo_reaching_an_alignment_pattern_is_refused() -> None:
    url = "https://www.diamond.ac.uk/" + "x" * 300
    grid = _encode(url)
    assert grid.version >= 19, f"expected a large code, got version {grid.version}"
    fraction = next(f / 100 for f in range(1, 100) if not _logo_fit(grid, f / 100)[1])
    assert _logo_fit(grid, fraction)[0] <= LOGO_MAX_CLEARED, "share hit first"
    with pytest.raises(ValueError, match="reaches a finder, timing or alignment"):
        build(url, style=dataclasses.replace(LOGO, logo_fraction=fraction))


# --logo with no path uses the shipped asset, and --logo-size reaches the
# style.
def test_cli_logo_flags_reach_the_style(tmp_path: Path) -> None:
    output = tmp_path / "code.svg"
    main([URLS[1], "--logo", "--logo-size", "0.25", "-o", str(output)])
    expected = build(URLS[1], style=dataclasses.replace(LOGO, logo_fraction=0.25)).svg
    assert output.read_text() == expected, "CLI logo render differs from build()"
