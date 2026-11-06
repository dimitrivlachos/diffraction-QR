"""Each optional feature switches off cleanly, alone or in any combination."""

from __future__ import annotations

import dataclasses
import itertools
import re
from pathlib import Path

import pytest

from diffraction import DiffractionStyle, _falloff_diameter, build, main
from margin import MARGIN_FLOOR, RENDER_SCALES, blur_margin, decodes, render

STYLE = DiffractionStyle()

# A circular-finder code and a rounded-finder code with six alignments.
URLS = [
    "https://dials.github.io/documentation/tutorials/index.html",
    "https://www.diamond.ac.uk/Instruments/Mx/MX-Data-Analysis/"
    "Interpreting-Experiment-Feedback.html",
]

# The on/off feature switches of DiffractionStyle.
SWITCHES = ("speckle", "ice_rings", "streaks", "panel_gaps")

# Every combination of the switches and the three falloffs.
COMBINATIONS = [
    dataclasses.replace(STYLE, falloff=falloff, **dict(zip(SWITCHES, on, strict=True)))
    for *on, falloff in itertools.product(
        *[(True, False)] * len(SWITCHES), ("wilson", "radial", "none")
    )
]

# The margin is costly to measure, so it covers every switch combination
# under the default falloff and each other falloff with every switch on.
MARGIN_COMBINATIONS = [
    style
    for style in COMBINATIONS
    if style.falloff == "wilson" or all(getattr(style, n) for n in SWITCHES)
]


def _label(style: DiffractionStyle) -> str:
    """Name a combination for the test id."""
    on = [n for n in SWITCHES if getattr(style, n)]
    return "+".join(on + [style.falloff])


# Markup each feature leaves in the SVG, so its absence can be checked.
FEATURE_MARKUP = {
    "speckle": f'fill="{STYLE.speckle_colour}"',
    "ice_rings": '<clipPath id="ice-clip"',
    "streaks": "<ellipse",
    "panel_gaps": '<clipPath id="gap-clip"',
}


# Turning one switch off must remove that feature's markup and leave the
# others in place, so the switches are independent.
@pytest.mark.parametrize("feature", FEATURE_MARKUP)
def test_each_switch_removes_only_its_feature(feature: str) -> None:
    svg = build(URLS[1], style=dataclasses.replace(STYLE, **{feature: False})).svg
    assert FEATURE_MARKUP[feature] not in svg, f"{feature} still drawn when off"
    for other, markup in FEATURE_MARKUP.items():
        if other != feature:
            assert markup in svg, f"turning off {feature} also removed {other}"


# With no falloff every spot has one size, before jitter.
def test_no_falloff_gives_one_spot_size() -> None:
    style = dataclasses.replace(STYLE, falloff="none")
    sizes = {_falloff_diameter(i / 10, style) for i in range(11)}
    assert sizes == {(STYLE.spot_diameter_centre + STYLE.spot_diameter_edge) / 2}, (
        f"spot sizes vary without falloff: {sizes}"
    )


# With everything off, and no jitter, every data spot is a circle of the
# same radius: the plainest code the generator can make.
def test_all_features_off_gives_plain_uniform_dots() -> None:
    style = dataclasses.replace(
        STYLE,
        speckle=False,
        ice_rings=False,
        streaks=False,
        panel_gaps=False,
        falloff="none",
        spot_jitter=0.0,
    )
    svg = build(URLS[1], style=style).svg
    colour = re.escape(STYLE.spot_colour)
    # Data spots are written with two-decimal centres; the alignment dots,
    # also circles in the spot colour, with one.
    data_spot = rf'<circle cx="\d+\.\d\d" cy="\d+\.\d\d" r="([\d.]+)" fill="{colour}"/>'
    radii = set(re.findall(data_spot, svg))
    assert "<ellipse" not in svg and "<clipPath" not in svg, "a feature survived"
    assert radii == {"0.375"}, f"data spots vary in size: {radii}"


# Every combination of features must decode at every render scale, so
# no pairing of features silently breaks a code.
@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("style", COMBINATIONS, ids=_label)
def test_every_combination_decodes(url: str, style: DiffractionStyle) -> None:
    svg = build(url, style=style).svg
    for scale in RENDER_SCALES:
        assert decodes(render(svg, scale), url), f"no decode at {scale} px/module"


# Every combination must also keep the camera headroom the default has.
@pytest.mark.parametrize("style", MARGIN_COMBINATIONS, ids=_label)
def test_every_combination_keeps_the_margin(style: DiffractionStyle) -> None:
    url = URLS[1]
    sigma = blur_margin(build(url, style=style).svg, url)
    assert sigma >= MARGIN_FLOOR, f"margin σ = {sigma:.3f} below floor"


# The command-line switches must reach the style: each --no- flag removes
# its feature from the written SVG.
@pytest.mark.parametrize("feature", FEATURE_MARKUP)
def test_cli_switch_turns_its_feature_off(tmp_path: Path, feature: str) -> None:
    output = tmp_path / "code.svg"
    flag = f"--no-{feature.replace('_', '-')}"
    main([URLS[1], flag, "-o", str(output)])
    assert FEATURE_MARKUP[feature] not in output.read_text(), f"{flag} ignored"


# --falloff reaches the style too: the CLI's flat render matches build()
# with no falloff, and differs from the default.
def test_cli_falloff_choice_reaches_the_style(tmp_path: Path) -> None:
    default, flat = tmp_path / "default.svg", tmp_path / "flat.svg"
    main([URLS[1], "-o", str(default)])
    main([URLS[1], "--falloff", "none", "-o", str(flat)])
    assert (
        flat.read_text()
        == build(URLS[1], style=dataclasses.replace(STYLE, falloff="none")).svg
    ), "--falloff none did not reach the style"
    assert default.read_text() != flat.read_text(), "falloff choice had no effect"
