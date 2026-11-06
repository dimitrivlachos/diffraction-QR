"""Decode round-trips and blur margins for the styled QR code."""

from __future__ import annotations

import hashlib

import pytest

from diffraction import build
from margin import MARGIN_FLOOR, RENDER_SCALES, blur_margin, decodes, plain_svg, render

# URLs chosen to land on versions 3 and 6, either side of the
# alignment-pattern grid growing, and below version 7 where version
# information blocks appear.
SHORT_URLS = [
    "https://dials.github.io",
    "https://www.diamond.ac.uk/Instruments/MX.html#x",
]

# Encodes at version 8, where the ring finders fail to decode.
LONG_URL = (
    "https://github.com/DiamondLightSource/fast-feedback-service/pull/110/files#diff"
)

# SHA-256 of build("https://dials.github.io", seed=0).svg, produced by the
# original script at commit e44782f and checked byte-identical to it.
SEED_0_SHA256 = "da9e2000640312a40711e28dc57dd2e984e2bd635adeeb1d079f2209fa1ad962"


# The default render must not change by accident; a deliberate change
# to the output updates SEED_0_SHA256 alongside it.
def test_default_render_is_unchanged() -> None:
    svg = build("https://dials.github.io", seed=0).svg
    digest = hashlib.sha256(svg.encode()).hexdigest()
    assert digest == SEED_0_SHA256, "seed-0 SVG differs from the recorded render"


# Every seed must decode at every render scale, since jitter and speckle
# vary by seed and ZXing's behaviour varies with scale.
@pytest.mark.parametrize("url", SHORT_URLS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("scale", RENDER_SCALES)
def test_styled_code_decodes_to_its_url(url: str, seed: int, scale: float) -> None:
    image = render(build(url, seed=seed).svg, scale)
    assert decodes(image, url), f"seed {seed} at {scale} px/module did not decode"


# The plain control must clear the floor comfortably, or the harness
# itself rather than the styling is the limit being measured.
@pytest.mark.parametrize("url", SHORT_URLS)
def test_plain_control_has_a_wide_margin(url: str) -> None:
    sigma = blur_margin(plain_svg(url), url)
    assert sigma >= 0.475, f"plain control survives only σ = {sigma:.3f}"


# The styled code must keep enough headroom for a real camera; this is
# the figure each later feature is measured against.
@pytest.mark.parametrize("url", SHORT_URLS)
def test_styled_margin_stays_above_floor(url: str) -> None:
    sigma = min(blur_margin(build(url, seed=seed).svg, url) for seed in (0, 1, 2))
    assert sigma >= MARGIN_FLOOR, f"worst-case margin σ = {sigma:.3f} below floor"


# Ring finders lose to ZXing's corner-based finder geometry at version 8;
# strict so the marker has to come off once the finders are fixed.
@pytest.mark.xfail(strict=True, reason="ring finders fail to decode at version >= 7")
def test_long_url_decodes_at_version_8() -> None:
    styled = build(LONG_URL, seed=0)
    assert styled.version == 8, f"expected version 8, got {styled.version}"
    sigma = blur_margin(styled.svg, LONG_URL)
    assert sigma >= MARGIN_FLOOR, f"version 8 margin σ = {sigma:.3f} below floor"
