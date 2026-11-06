"""Decode round-trips and blur margins for the styled QR code."""

from __future__ import annotations

import hashlib

import pytest

from diffraction import ROUND_FINDER_MAX_VERSION, DiffractionStyle, build
from margin import MARGIN_FLOOR, RENDER_SCALES, blur_margin, decodes, plain_svg, render

# Real URLs mapped to the QR version each encodes at, level H. Versions
# 3 and 6 use circular finders; 7, 8 and 12 use rounded squares, with 7
# the first version to carry version information.
URL_VERSIONS = {
    "https://dials.github.io": 3,
    "https://www.diamond.ac.uk/Instruments/MX.html#x": 6,
    "https://www.diamond.ac.uk/Instruments/Mx/I03/I03-Overview.html": 7,
    "https://github.com/DiamondLightSource/fast-feedback-service/pull/110/files#diff": 8,
    "https://dials.github.io/documentation/tutorials/"
    "processing_in_detail_betalactamase.html"
    "?from=qr&campaign=diamond-open-day-2026#integration-and-scaling": 12,
}
URLS = list(URL_VERSIONS)

# The pre-Wilson look: radial falloff, round spots and no panel gaps.
RADIAL = DiffractionStyle(falloff="radial", streaks=False, panel_gaps=False)

# SHA-256 of build("https://dials.github.io", seed=0).svg with the Wilson
# falloff at B = 20 Å², tangential streaks and panel gaps, recorded when
# the panel gaps were added.
SEED_0_SHA256 = "8cf8d9e4e64ce5648248a79e6527f8eb28aebb2b1be36d6835af8493136e74a7"

# The same render with falloff="radial", recorded at commit 42969c6 when
# the ice rings were added; the spot jitter and speckle still match the
# original script at commit e44782f.
SEED_0_RADIAL_SHA256 = (
    "73c6c3d622a3b195a8785e1f3b1c737025c72321e4c86ae481c2504c246c5844"
)


# The default render must not change by accident; a deliberate change
# to the output updates SEED_0_SHA256 alongside it.
def test_default_render_is_unchanged() -> None:
    svg = build("https://dials.github.io", seed=0).svg
    digest = hashlib.sha256(svg.encode()).hexdigest()
    assert digest == SEED_0_SHA256, "seed-0 SVG differs from the recorded render"


# The radial falloff stays selectable for comparison, and selecting it
# must reproduce the render from before the Wilson falloff exactly.
def test_radial_falloff_reproduces_the_earlier_render() -> None:
    svg = build("https://dials.github.io", seed=0, style=RADIAL).svg
    digest = hashlib.sha256(svg.encode()).hexdigest()
    assert digest == SEED_0_RADIAL_SHA256, "radial render drifted from 42969c6"


# Each URL must keep landing on its intended version, or the tests
# below silently stop covering one side of the finder cut-over.
@pytest.mark.parametrize("url", URLS)
def test_url_encodes_at_its_intended_version(url: str) -> None:
    version = build(url).version
    assert version == URL_VERSIONS[url], f"{url} encodes at version {version}"


# The finder shape switches exactly at the cut-over: circles at or below
# it, rounded squares above, which ZXing needs from version 7.
@pytest.mark.parametrize("url", URLS)
def test_finder_shape_follows_the_version_cut_over(url: str) -> None:
    svg = build(url).svg
    if URL_VERSIONS[url] <= ROUND_FINDER_MAX_VERSION:
        assert 'width="6" height="6"' not in svg, "small code drew square finders"
    else:
        assert 'r="3"' not in svg, "large code drew circular finders"


# Every seed must decode at every render scale, since jitter and speckle
# vary by seed and ZXing's behaviour varies with scale.
@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("scale", RENDER_SCALES)
def test_styled_code_decodes_to_its_url(url: str, seed: int, scale: float) -> None:
    image = render(build(url, seed=seed).svg, scale)
    assert decodes(image, url), f"seed {seed} at {scale} px/module did not decode"


# The plain control must clear the floor comfortably, or the harness
# itself rather than the styling is the limit being measured.
@pytest.mark.parametrize("url", URLS)
def test_plain_control_has_a_wide_margin(url: str) -> None:
    sigma = blur_margin(plain_svg(url), url)
    assert sigma >= 0.475, f"plain control survives only σ = {sigma:.3f}"


# The styled code must keep enough headroom for a real camera; this is
# the figure each later feature is measured against.
@pytest.mark.parametrize("url", URLS)
def test_styled_margin_stays_above_floor(url: str) -> None:
    sigma = min(blur_margin(build(url, seed=seed).svg, url) for seed in (0, 1, 2))
    assert sigma >= MARGIN_FLOOR, f"worst-case margin σ = {sigma:.3f} below floor"
