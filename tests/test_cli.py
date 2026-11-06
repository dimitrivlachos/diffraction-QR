"""Printed sizing and the command-line checks."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from diffraction import build, main
from margin import RENDER_SCALES, blur_margin, decodes, render

URL = "https://xia2.github.io"


# A printed size changes only the SVG's physical dimensions; the
# drawing in module units, and so everything a decoder sees, is the same.
def test_size_mm_changes_only_the_physical_dimensions() -> None:
    unsized = build(URL).svg
    sized = build(URL, size_mm=23.5).svg
    assert 'width="23.5mm" height="23.5mm"' in sized, "printed size not set"
    strip = re.compile(r'width="[^"]*mm" height="[^"]*mm"')
    assert strip.sub("", sized) == strip.sub("", unsized), "drawing changed with size"


# The harness renders from the viewBox, so a sized SVG decodes and
# measures the same as the unsized one rather than at a scaled resolution.
def test_sized_svg_measures_the_same_margin() -> None:
    sized = build(URL, size_mm=10.0).svg
    assert all(decodes(render(sized, s), URL) for s in RENDER_SCALES), "no decode"
    assert blur_margin(sized, URL) == blur_margin(build(URL).svg, URL), (
        "margin depends on the printed size"
    )


# A slot too small for the URL must be flagged before it is printed,
# naming the size that would fit.
def test_cli_warns_when_modules_print_too_small(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "small.svg"
    with caplog.at_level(logging.WARNING):
        main([URL, "--size-mm", "10", "-o", str(output)])
    assert output.exists(), "SVG not written"
    assert "below the 0.40 mm print minimum" in caplog.text, "no print-size warning"


# A comfortable slot and a healthy code pass both checks silently.
def test_cli_check_passes_quietly_for_a_healthy_code(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "xia2.svg"
    with caplog.at_level(logging.INFO):
        main([URL, "--size-mm", "23.5", "--check", "-o", str(output)])
    assert "Blur margin" in caplog.text, "margin not reported"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        f"unexpected warning: {caplog.text}"
    )
