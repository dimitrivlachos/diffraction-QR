"""Render a QR code styled as a diffraction pattern, as an SVG.

Data modules become round spots whose diameter falls off with distance
from the code centre, the way spot size falls off towards the edge of a
detector; the finder and alignment patterns become concentric rings. The
result still scans because of how decoders such as ZXing and the iOS
camera read a code (ISO/IEC 18004).

A decoder locates the three finder patterns by scanning rows, columns and
diagonals for a dark:light:dark:light:dark run ratio of 1:1:3:1:1, with
each run allowed about ±50%. Concentric circles preserve that ratio along
any line through their centre, so ring finders are found as readily as
square ones. Finding them is not the whole job, though: ZXing also
refines module size and geometry from the finders' square outlines, and
from version 7 onwards, where the code carries version information, a
circular finder makes that refinement fail outright. Codes up to version
6 therefore keep circular finders and larger ones get rounded squares,
rounded only as far as a margin sweep showed ZXing still tolerates.

Once located, the decoder fits a perspective transform to
the finder centres and any alignment patterns, then samples the centre
of each module. A dark module therefore only has to be dark at its
centre: a round spot of diameter at least ~0.55 of the module pitch is
safe and everything outside it is free for styling. Error correction
level H tolerates 30% codeword loss, which absorbs the occasional
sampling error from small spots or print blur.

The faint speckle drawn in light modules must stay well clear of the
decoder's binarisation threshold, so it is kept both tiny and pale. No
decode check is performed here; ``margin.py`` measures how much blur a
render survives, and a styled code should still be verified with a real
scanner before it is printed.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import qrcode
from qrcode.constants import ERROR_CORRECT_H
from qrcode.util import pattern_position

logger = logging.getLogger(__name__)

# Smallest data spot diameter, in module pitches, that is still dark at
# the sampling point after modest misregistration and print blur.
SAFE_SPOT_DIAMETER = 0.55

# Half-widths, in modules, of the 7x7 finder and 5x5 alignment patterns.
FINDER_HALF_WIDTH = 3
ALIGNMENT_HALF_WIDTH = 2

# Largest QR version drawn with circular finders. From version 7 the
# code carries version information and ZXing 2.3 failed to decode
# circular finders at every version swept (7 to 22).
ROUND_FINDER_MAX_VERSION = 6

# Corner radius of the larger codes' finders, as a fraction of each
# square's half-width: 0 is a square, 1 a circle. Under ZXing 2.3, 0.55
# collapsed the blur margin at versions 13 and 22, while 0.4 kept
# σ >= 0.375 module pitches from version 7 to 22.
LARGE_FINDER_ROUNDING = 0.4

# Exponent of the radial falloff in spot diameter; above 1 keeps the
# central spots large for longer before they shrink towards the corners.
FALLOFF_EXPONENT = 1.5

# Maximum offset, in modules, of a background speckle from its module
# centre.
SPECKLE_SCATTER = 0.2

# URL encoded when none is given on the command line.
DEFAULT_URL = "https://dials.github.io"


@dataclass(frozen=True)
class DiffractionStyle:
    """Appearance of the rendered code.

    Diameters are in units of the module pitch; probabilities are per
    light module.
    """

    spot_colour: str = "#202945"
    quiet_zone: int = 4
    spot_diameter_centre: float = 0.92
    spot_diameter_edge: float = 0.58
    spot_jitter: float = 0.12
    speckle: bool = True
    speckle_colour: str = "#b7b7b7"
    speckle_diameter: float = 0.10
    speckle_probability: float = 0.35


class StyledQr(NamedTuple):
    """An SVG document with the QR version and side length it encodes."""

    svg: str
    size: int
    version: int


def _pattern_centres(
    size: int, version: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the (x, y) module centres of the finder and alignment patterns."""
    finders = [
        (FINDER_HALF_WIDTH, FINDER_HALF_WIDTH),
        (size - 1 - FINDER_HALF_WIDTH, FINDER_HALF_WIDTH),
        (FINDER_HALF_WIDTH, size - 1 - FINDER_HALF_WIDTH),
    ]

    # The alignment grid is the cross product of one coordinate list; the
    # three positions that fall on a finder are not drawn.
    positions = pattern_position(version)
    near, far = 8, size - 9
    alignments = [
        (i, j)
        for i in positions
        for j in positions
        if not (
            (i < near and j < near) or (i < near and j > far) or (i > far and j < near)
        )
    ]
    return finders, alignments


def _covered_modules(
    centres: list[tuple[int, int]], half_width: int
) -> set[tuple[int, int]]:
    """Return every module inside the square patterns centred on ``centres``."""
    offsets = range(-half_width, half_width + 1)
    return {
        (cx + dx, cy + dy) for cx, cy in centres for dx in offsets for dy in offsets
    }


def _finder_svg(cx: float, cy: float, colour: str, version: int) -> list[str]:
    """Return the SVG elements of one finder centred on (``cx``, ``cy``).

    A unit-width stroke on the half-width-3 outline spans the dark ring
    2.5-3.5, leaving light 1.5-2.5 around a dark core of half-width 1.5.
    """
    if version <= ROUND_FINDER_MAX_VERSION:
        return [
            f'<circle cx="{cx}" cy="{cy}" r="3" fill="none" '
            f'stroke="{colour}" stroke-width="1"/>',
            f'<circle cx="{cx}" cy="{cy}" r="1.5" fill="{colour}"/>',
        ]

    # Rounded squares; the stroke's corner radius is set on its centre
    # line, so the core's radius scales with its own half-width.
    return [
        f'<rect x="{cx - 3}" y="{cy - 3}" width="6" height="6" '
        f'rx="{3 * LARGE_FINDER_ROUNDING:g}" fill="none" '
        f'stroke="{colour}" stroke-width="1"/>',
        f'<rect x="{cx - 1.5}" y="{cy - 1.5}" width="3" height="3" '
        f'rx="{1.5 * LARGE_FINDER_ROUNDING:g}" fill="{colour}"/>',
    ]


def build(url: str, seed: int = 0, style: DiffractionStyle | None = None) -> StyledQr:
    """Encode ``url`` at error correction level H and render it as an SVG.

    The SVG viewBox is in module units and is sized at one millimetre per
    module. ``seed`` fixes the spot jitter and speckle so a given URL and
    seed always render identically.
    """
    style = style or DiffractionStyle()
    rng = random.Random(seed)

    # Encode at the smallest version that fits, with no built-in border;
    # the quiet zone is added by the SVG transform instead.
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, border=0)
    qr.add_data(url)
    qr.make(fit=True)
    modules = qr.modules
    size = len(modules)

    # Finder and alignment modules are drawn as rings afterwards, so the
    # per-module pass skips them.
    finders, alignments = _pattern_centres(size, qr.version)
    rings = _covered_modules(finders, FINDER_HALF_WIDTH) | _covered_modules(
        alignments, ALIGNMENT_HALF_WIDTH
    )

    width = size + 2 * style.quiet_zone
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {width}" '
        f'width="{width}mm" height="{width}mm">',
        f'<rect width="{width}" height="{width}" fill="#fff"/>',
        f'<g transform="translate({style.quiet_zone},{style.quiet_zone})">',
    ]

    # The code centre plays the beam centre; radii are normalised so the
    # corner modules sit at 1.
    centre = (size - 1) / 2.0
    corner_distance = math.hypot(centre, centre)

    for y in range(size):
        for x in range(size):
            if (x, y) in rings:
                continue
            cx, cy = x + 0.5, y + 0.5

            # Dark modules: large spots at the centre shrinking towards the
            # edge, jittered, and never below the safe sampling diameter.
            if modules[y][x]:
                r = math.hypot(x - centre, y - centre) / corner_distance
                d = (
                    style.spot_diameter_centre
                    + (style.spot_diameter_edge - style.spot_diameter_centre)
                    * r**FALLOFF_EXPONENT
                )
                d *= 1.0 + rng.uniform(-style.spot_jitter, style.spot_jitter)
                d = max(d, SAFE_SPOT_DIAMETER)
                out.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{d / 2:.3f}" '
                    f'fill="{style.spot_colour}"/>'
                )

            # Light modules: occasional faint speckle, scattered off centre.
            elif style.speckle and rng.random() < style.speckle_probability:
                sx = cx + rng.uniform(-SPECKLE_SCATTER, SPECKLE_SCATTER)
                sy = cy + rng.uniform(-SPECKLE_SCATTER, SPECKLE_SCATTER)
                out.append(
                    f'<circle cx="{sx:.2f}" cy="{sy:.2f}" '
                    f'r="{style.speckle_diameter / 2:.3f}" '
                    f'fill="{style.speckle_colour}"/>'
                )

    # Finders: circles for small codes, rounded squares for large ones.
    for fx, fy in finders:
        out.extend(_finder_svg(fx + 0.5, fy + 0.5, style.spot_colour, qr.version))

    # Alignment patterns: dark ring 1.5-2.5, light 0.5-1.5, dark dot r = 0.5.
    for ax, ay in alignments:
        cx, cy = ax + 0.5, ay + 0.5
        out.append(
            f'<circle cx="{cx}" cy="{cy}" r="2" fill="none" '
            f'stroke="{style.spot_colour}" stroke-width="1"/>'
        )
        out.append(f'<circle cx="{cx}" cy="{cy}" r="0.5" fill="{style.spot_colour}"/>')

    out.append("</g></svg>")
    return StyledQr(svg="\n".join(out), size=size, version=qr.version)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="data to encode")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for jitter and speckle"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("diffqr.svg"), help="SVG to write"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    styled = build(args.url, seed=args.seed)
    args.output.write_text(styled.svg)
    logger.info(
        "Wrote %s: version %d, %dx%d modules",
        args.output,
        styled.version,
        styled.size,
        styled.size,
    )


if __name__ == "__main__":
    main()
