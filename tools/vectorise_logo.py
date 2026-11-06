"""Trace the Diamond logo from a raster image into a single-colour SVG.

The source is a flat two-colour image: a yellow disc with a white
starburst cut out of it. The yellow is the only colour with almost no
blue, so thresholding the blue channel separates it cleanly from both
the white starburst and the white background. The source edge is hard,
a staircase of whole pixels, and upsampling alone keeps the staircase;
blurring by about a source pixel first turns each step into a ramp whose
midpoint contour is the true edge. potrace then traces the yellow
region, starburst holes included, as one path with smooth curves.

The trace is accepted only if it rasterises back onto the source mask:
their intersection over union must reach IOU_FLOOR at the source's own
resolution. This runs once, by hand; the generator embeds the committed
SVG and never needs potrace. It requires the potrace command (1.15 was
used), which conda-forge does not package.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Blue-channel level below which a pixel counts as logo yellow; the
# yellow sits near 30 and the white near 255.
BLUE_THRESHOLD = 128

# Upsampling before thresholding, so the traced edge can fall between
# source pixels.
UPSAMPLE = 8

# Gaussian blur before thresholding, in source pixels. At 0.8 the
# staircase still showed in the trace; 1.5 smooths it at an IoU of 0.993.
BLUR_SOURCE_PX = 1.5

# Speckles smaller than this many pixels (at the upsampled size) are
# dropped by potrace as JPEG noise.
TURD_SIZE = 20

# Smallest intersection over union between trace and source mask.
IOU_FLOOR = 0.98


def yellow_mask(image: Image.Image, upsample: int = 1, blur: float = 0.0) -> np.ndarray:
    """Return a boolean mask of the logo's yellow, optionally upsampled and
    smoothed by a Gaussian of ``blur`` source pixels before thresholding."""
    blue = image.convert("RGB").getchannel("B")
    if upsample > 1:
        blue = blue.resize(
            (blue.width * upsample, blue.height * upsample), Image.Resampling.BICUBIC
        )
    if blur > 0.0:
        blue = blue.filter(ImageFilter.GaussianBlur(blur * upsample))
    return np.asarray(blue) < BLUE_THRESHOLD


def logo_colour(image: Image.Image) -> str:
    """Return the median colour of the yellow pixels as a hex string."""
    rgb = np.asarray(image.convert("RGB"))
    r, g, b = np.median(rgb[yellow_mask(image)], axis=0).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def trace(mask: np.ndarray) -> tuple[str, str]:
    """Trace ``mask`` with potrace and return its group transform and path.

    potrace traces black pixels, so the mask is written as a PBM with the
    yellow black. Its SVG output flips y with a transform on a group,
    which is returned alongside the path so the geometry stays exact.
    """
    with tempfile.TemporaryDirectory() as work:
        pbm, svg = Path(work) / "logo.pbm", Path(work) / "logo.svg"
        Image.fromarray(~mask).convert("1").save(pbm)
        subprocess.run(
            [
                "potrace",
                "--svg",
                "--flat",
                f"--turdsize={TURD_SIZE}",
                "--resolution=72",
                f"--output={svg}",
                str(pbm),
            ],
            check=True,
        )
        traced = svg.read_text()
    transform = re.search(r'<g transform="([^"]+)"', traced).group(1)
    path = re.search(r'<path d="([^"]+)"', traced, re.DOTALL).group(1)
    return transform, " ".join(path.split())


def logo_svg(transform: str, path: str, colour: str, size: int, upsample: int) -> str:
    """Return the logo as a standalone SVG in source pixel units.

    potrace's coordinates are in upsampled pixels, so the group is scaled
    back down to the source's pixel grid.
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">\n'
        f'<g transform="scale({1 / upsample:g}) {transform}" fill="{colour}">\n'
        f'<path d="{path}"/>\n'
        "</g>\n"
        "</svg>\n"
    )


def iou(svg: str, mask: np.ndarray) -> float:
    """Return the intersection over union of the rendered ``svg`` and
    ``mask``, at the mask's resolution."""
    png = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=mask.shape[1], output_height=mask.shape[0]
    )
    with Image.open(io.BytesIO(png)) as rendered:
        alpha = np.asarray(rendered.convert("RGBA").getchannel("A")) >= 128
    return float((alpha & mask).sum() / (alpha | mask).sum())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="raster logo, square")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("assets/diamond-logo.svg"),
        help="SVG to write",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Trace at the upsampled size, colour from the source, then check the
    # trace against the source mask at the source's own resolution.
    with Image.open(args.source) as image:
        if image.width != image.height:
            parser.error(f"{args.source} is {image.width}x{image.height}, not square")
        transform, path = trace(yellow_mask(image, UPSAMPLE, BLUR_SOURCE_PX))
        svg = logo_svg(transform, path, logo_colour(image), image.width, UPSAMPLE)
        score = iou(svg, yellow_mask(image))

    logger.info("Trace overlaps the source mask with IoU %.4f", score)
    if score < IOU_FLOOR:
        raise SystemExit(f"IoU {score:.4f} is below {IOU_FLOOR}; trace rejected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
