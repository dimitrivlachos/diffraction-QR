"""Measure how much blur a styled QR code survives before it stops decoding.

Every styling feature spends some of the error-correction budget, and a
code that still decodes from a perfect render may have nothing left for
a phone camera. This module turns that leftover budget into one
number: render the SVG, convolve it with a Gaussian of width σ (in
module pitches) and step σ upwards until ZXing no longer returns the
encoded payload. Comparing that σ before and after a feature gives that
feature's cost, and the same figure for a plain square-module code
gives the ceiling.

A single render scale is not trustworthy. At exactly 8 px per module
the pixel grid aligns with the module grid, and ZXing 2.3 fails to
read even a plain code that it reads at 6 or 10 px; a camera never
produces such an image. The margin is therefore the worst case over
several render scales chosen to be non-integer and mutually
incommensurate. For the same reason the scan starts from a small σ
floor rather than from a perfectly sharp render, which ZXing
occasionally rejects and no lens delivers.

Decoding is not monotonic in σ: a marginal code can fail at one blur
and succeed at a wider one. A bisection would land on either side of
such a gap by chance, so the scan walks up from the floor and stops at
the first failure; the margin is the largest σ below which every step
decoded. Blur stands in for only one kind of damage; perspective, glare
and uneven lighting are not modelled.
"""

from __future__ import annotations

import argparse
import io
import logging
import re

import cairosvg
import numpy as np
import qrcode
import zxingcpp
from PIL import Image
from qrcode.constants import ERROR_CORRECT_H
from scipy.ndimage import gaussian_filter

from diffraction import DEFAULT_URL, build

logger = logging.getLogger(__name__)

# Render scales in pixels per module: non-integer so no scale aligns the
# pixel grid with the module grid, and fine enough that σ of a few
# hundredths of a module is still resolved by the blur kernel.
RENDER_SCALES = (6.3, 7.7, 9.1, 11.3)

# Blur every render already carries, in module pitches; a stand-in for
# the camera's own point-spread function.
SIGMA_FLOOR = 0.05

# Step of the σ scan, in module pitches; margins closer than this are
# indistinguishable.
SIGMA_STEP = 0.025

# Largest blur scanned, in module pitches; beyond one module no code
# decodes.
SIGMA_CEILING = 1.0

# Smallest acceptable worst-case margin, in module pitches. Plain
# square-module codes reach about 0.525 and the default style 0.425;
# below 0.4 the styling has eaten most of the headroom a camera needs.
MARGIN_FLOOR = 0.4

# Side of the SVG viewBox, which is in module units.
VIEWBOX_SIDE = re.compile(r'viewBox="0 0 ([\d.]+) ')


def render(svg: str, px_per_module: float) -> np.ndarray:
    """Rasterise ``svg`` to a greyscale float image in [0, 1].

    The output size comes from the viewBox, which is in module units, so
    the printed size set on the SVG does not affect the render scale.
    """
    side = float(VIEWBOX_SIDE.search(svg).group(1))
    png = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=round(side * px_per_module)
    )
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def decodes(image: np.ndarray, expected: str) -> bool:
    """Return whether ZXing reads exactly ``expected`` from ``image``."""
    pixels = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    results = zxingcpp.read_barcodes(pixels, formats=zxingcpp.BarcodeFormat.QRCode)
    return any(result.text == expected for result in results)


def blur_margin(
    svg: str, expected: str, scales: tuple[float, ...] = RENDER_SCALES
) -> float:
    """Return the largest Gaussian σ, in module pitches, that still decodes.

    The result is the worst case over ``scales``. Returns 0.0 when the
    render fails even at SIGMA_FLOOR, and SIGMA_CEILING when every step
    up to it decodes.
    """
    return min(_blur_margin_at(svg, expected, scale) for scale in scales)


def _blur_margin_at(svg: str, expected: str, px_per_module: float) -> float:
    """Scan σ upwards from SIGMA_FLOOR at one render scale."""
    image = render(svg, px_per_module)
    steps = int(round((SIGMA_CEILING - SIGMA_FLOOR) / SIGMA_STEP))
    survived = 0.0
    for i in range(steps + 1):
        sigma = SIGMA_FLOOR + i * SIGMA_STEP
        if not decodes(gaussian_filter(image, sigma * px_per_module), expected):
            break
        survived = sigma
    return survived


def plain_svg(url: str, quiet_zone: int = 4) -> str:
    """Render ``url`` as an unstyled square-module code, the control case.

    Uses the same encoding settings as the styled code (level H, smallest
    version) so the two margins differ only by the styling.
    """
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, border=0)
    qr.add_data(url)
    qr.make(fit=True)
    width = len(qr.modules) + 2 * quiet_zone
    squares = [
        f'<rect x="{x}" y="{y}" width="1" height="1"/>'
        for y, row in enumerate(qr.modules)
        for x, dark in enumerate(row)
        if dark
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {width}" '
        f'width="{width}mm" height="{width}mm">'
        f'<rect width="{width}" height="{width}" fill="#fff"/>'
        f'<g transform="translate({quiet_zone},{quiet_zone})">'
        + "".join(squares)
        + "</g></svg>"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="data to encode")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds to measure"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # The plain code is the ceiling every styled render is compared to.
    logger.info("plain squares: σ = %.3f", blur_margin(plain_svg(args.url), args.url))

    # Report each seed and the worst case, since jitter and speckle move
    # the margin from seed to seed.
    margins = []
    for seed in args.seeds:
        styled = build(args.url, seed=seed)
        sigma = blur_margin(styled.svg, args.url)
        margins.append(sigma)
        logger.info("seed %d (version %d): σ = %.3f", seed, styled.version, sigma)
    logger.info("worst case: σ = %.3f module pitches", min(margins))


if __name__ == "__main__":
    main()
