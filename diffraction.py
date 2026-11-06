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

Ice rings are placed by a virtual experiment rather than by eye. The
symbol's corner modules sit at the corners of a detector of known
half-diagonal at a known distance, so each module has a scattering
angle 2θ and, through Bragg's law at wavelength λ, a resolution d. The
rings are the allowed reflections of hexagonal ice Ih, generated from
its unit cell and space group exactly as DIALS does when masking them,
down to the resolution at the symbol's corners. They are drawn pale and
beneath the spots, clipped away from the quiet zone and from the finder
and alignment squares, whose light gaps must stay white. Dark spots that
sit on a ring grow a little, as reflections overlapping a ring do on a
real image; light modules are never darkened beyond the ring's pale
tint, so the decoder still reads them as light.

The faint speckle drawn in light modules must stay well clear of the
decoder's binarisation threshold, so it is kept both tiny and pale. No
decode check is performed here; ``margin.py`` measures how much blur a
render survives, and a styled code should still be verified with a real
scanner before it is printed.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

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

# Longest a streaked spot may grow, in module pitches. A half-length
# under one module never reaches a neighbouring module's sampling point,
# so a streak cannot darken a light module where the decoder looks.
STREAK_CAP = 1.0

# Overall B factor of the Wilson falloff, in Å², typical of a well-ordered
# cryo-cooled protein crystal. A larger B shrinks the mid-radius spots:
# at 30 Å² a version 12 code fell below the margin floor, at 20 Å² it held.
WILSON_B = 20.0

# Maximum offset, in modules, of a background speckle from its module
# centre.
SPECKLE_SCATTER = 0.2

# Wavelength of the virtual beam, in Å; 0.9763 Å is 12.7 keV, a common MX
# energy at Diamond. A longer wavelength pushes every ring outwards.
WAVELENGTH_A = 0.9763

# Sample-to-detector distance of the virtual experiment, in mm; chosen so
# the 1.95 Å ice ring lands near the corners at WAVELENGTH_A.
DETECTOR_DISTANCE_MM = 380.0

# Half-diagonal of the virtual detector, in mm, mapped onto the symbol's
# corner modules: an Eiger2 XE 16M, 4148 x 4362 pixels of 75 µm.
DETECTOR_HALF_DIAGONAL_MM = 225.7

# Hexagonal ice Ih cell edges in Å, space group P6₃/mmc (194); the values
# DIALS masks ice rings with (dials.util.masking, ice_rings.unit_cell).
ICE_CELL_A = 4.498
ICE_CELL_C = 7.338

# Fractional growth of a dark spot's diameter at the centre of an ice
# ring, tapering to none at the ring's edge.
ICE_SPOT_BOOST = 0.25

# Diameter, in module pitches, beyond which the ice boost stops growing a
# spot, so spots on a ring do not merge with their neighbours.
ICE_SPOT_CAP = 1.0

# URL encoded when none is given on the command line.
DEFAULT_URL = "https://dials.github.io"

# Smallest printed module, in mm, that a phone camera reads reliably at
# arm's length; below it the URL is too long for the space it is given.
MIN_PRINT_MODULE_MM = 0.4


@dataclass(frozen=True)
class DiffractionStyle:
    """Appearance of the rendered code.

    Diameters and widths are in units of the module pitch; probabilities
    are per light module. The wavelength is in Å and detector lengths are
    in mm.
    """

    # Optional features, each switched independently; with all of them off
    # the code is plain uniform dots with ring finders.
    speckle: bool = True
    ice_rings: bool = True
    streaks: bool = True
    falloff: Literal["wilson", "radial", "none"] = "wilson"

    spot_colour: str = "#202945"
    quiet_zone: int = 4
    spot_diameter_centre: float = 0.92
    spot_diameter_edge: float = 0.58
    spot_jitter: float = 0.12
    speckle_colour: str = "#b7b7b7"
    speckle_diameter: float = 0.10
    speckle_probability: float = 0.35
    wilson_b: float = WILSON_B
    streak_strength: float = 0.4
    ice_ring_colour: str = "#e8ebf1"
    ice_ring_width: float = 0.8
    wavelength_a: float = WAVELENGTH_A
    detector_distance_mm: float = DETECTOR_DISTANCE_MM
    detector_half_diagonal_mm: float = DETECTOR_HALF_DIAGONAL_MM


class StyledQr(NamedTuple):
    """An SVG document with the QR version and dimensions it encodes.

    ``size`` is the symbol's side in modules; ``width`` adds the quiet
    zone on both sides and is the SVG viewBox side.
    """

    svg: str
    size: int
    version: int
    width: int


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


def _alignment_svg(cx: float, cy: float, colour: str) -> list[str]:
    """Return the SVG elements of one alignment pattern centred on (cx, cy).

    Dark ring 1.5-2.5, light 0.5-1.5, dark dot of radius 0.5.
    """
    return [
        f'<circle cx="{cx}" cy="{cy}" r="2" fill="none" '
        f'stroke="{colour}" stroke-width="1"/>',
        f'<circle cx="{cx}" cy="{cy}" r="0.5" fill="{colour}"/>',
    ]


@dataclass(frozen=True)
class CodeGrid:
    """The encoded symbol and the geometry every layer draws against.

    Coordinates are in modules with the origin at the symbol's top-left
    corner; the quiet zone lies outside. ``reserved`` holds the modules
    the pattern layer draws as rings, which the spot layer leaves alone.
    """

    modules: list[list[bool]]
    size: int
    version: int
    finders: list[tuple[int, int]]
    alignments: list[tuple[int, int]]
    reserved: frozenset[tuple[int, int]]

    @property
    def centre(self) -> float:
        """Module index of the symbol centre, which plays the beam centre."""
        return (self.size - 1) / 2.0

    @property
    def corner_distance(self) -> float:
        """Distance, in modules, from the centre to a corner module's centre."""
        return math.hypot(self.centre, self.centre)

    def radius(self, x: float, y: float) -> float:
        """Return the distance of module (x, y) from the centre, normalised
        so the corner modules sit at 1."""
        return math.hypot(x - self.centre, y - self.centre) / self.corner_distance


def resolution(r: float, style: DiffractionStyle) -> float:
    """Return the resolution d, in Å, at normalised radius ``r``.

    The radius maps linearly onto the detector, r = 1 at its corner, and
    Bragg's law gives d = λ / (2 sin θ) with 2θ = atan(r_mm / D).
    """
    if r == 0:
        return math.inf
    two_theta = math.atan(
        r * style.detector_half_diagonal_mm / style.detector_distance_mm
    )
    return style.wavelength_a / (2.0 * math.sin(two_theta / 2.0))


def ring_radius(d: float, style: DiffractionStyle) -> float | None:
    """Return the normalised radius of resolution ``d`` in Å, the inverse of
    ``resolution``.

    Returns None when 2θ would reach 90°, which a flat detector facing the
    beam cannot record; that includes every d below λ/2, where Bragg's law
    has no solution.
    """
    sin_theta = style.wavelength_a / (2.0 * d)
    if sin_theta >= math.sqrt(0.5):
        return None
    two_theta = 2.0 * math.asin(sin_theta)
    return (
        style.detector_distance_mm
        * math.tan(two_theta)
        / style.detector_half_diagonal_mm
    )


def ice_d_spacings(d_min: float) -> list[float]:
    """Return the distinct d-spacings of ice Ih down to ``d_min``, in Å.

    Enumerates every reflection of the hexagonal cell, where
    1/d² = 4(h² + hk + k²)/(3a²) + l²/c², and drops the absences of
    P6₃/mmc: the hh(-2h)l family, and with it 000l, needs l even. The
    family appears as (h, k) = (h, h), (h, -2h) and (-2h, h) up to sign.
    Matches cctbx's build_miller_set for the same cell and space group,
    which is what DIALS masks.
    """
    h_max = int(ICE_CELL_A / d_min) + 2
    l_max = int(ICE_CELL_C / d_min) + 2
    spacings = set()
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(l_max + 1):  # noqa: E741 - Miller index
                if h == k == l == 0:
                    continue
                if (h == k or k == -2 * h or h == -2 * k) and l % 2:
                    continue
                inverse_d_squared = (
                    4.0 * (h * h + h * k + k * k) / (3.0 * ICE_CELL_A**2)
                    + l * l / ICE_CELL_C**2
                )
                d = 1.0 / math.sqrt(inverse_d_squared)
                if d >= d_min:
                    # Rounding merges symmetry equivalents that differ only
                    # in floating-point noise.
                    spacings.add(round(d, 6))
    return sorted(spacings, reverse=True)


def _ice_ring_radii(grid: CodeGrid, style: DiffractionStyle) -> list[float]:
    """Return the normalised radii of the ice rings that reach the symbol.

    The limit is the resolution at the symbol's outer corner, half a
    module beyond the corner module's centre.
    """
    r_edge = (grid.corner_distance + math.sqrt(0.5)) / grid.corner_distance
    radii = (ring_radius(d, style) for d in ice_d_spacings(resolution(r_edge, style)))
    return [r for r in radii if r is not None]


def _encode(url: str) -> CodeGrid:
    """Encode ``url`` at level H, smallest version, with no built-in border."""
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_H, border=0)
    qr.add_data(url)
    qr.make(fit=True)
    size = len(qr.modules)
    finders, alignments = _pattern_centres(size, qr.version)
    reserved = _covered_modules(finders, FINDER_HALF_WIDTH) | _covered_modules(
        alignments, ALIGNMENT_HALF_WIDTH
    )
    return CodeGrid(
        modules=qr.modules,
        size=size,
        version=qr.version,
        finders=finders,
        alignments=alignments,
        reserved=frozenset(reserved),
    )


def _ice_ring_layer(grid: CodeGrid, style: DiffractionStyle, seed: int) -> list[str]:
    """Draw the ice rings as pale annuli, clipped to the symbol.

    The clip is the symbol square with the finder and alignment squares cut
    out, so the quiet zone and the patterns' light gaps stay white.
    """
    if not style.ice_rings:
        return []

    # One subpath per reserved square, wound against the symbol square so
    # each is a hole under both the nonzero and even-odd rules; cairosvg
    # ignores clip-rule, so the winding alone has to carry it.
    holes = [
        f"M{x - half} {y - half}v{2 * half + 1}h{2 * half + 1}v{-(2 * half + 1)}z"
        for centres, half in (
            (grid.finders, FINDER_HALF_WIDTH),
            (grid.alignments, ALIGNMENT_HALF_WIDTH),
        )
        for x, y in centres
    ]
    n = grid.size
    out = [
        f'<clipPath id="ice-clip"><path d="M0 0H{n}V{n}H0Z{"".join(holes)}"/>'
        "</clipPath>",
        f'<g clip-path="url(#ice-clip)" fill="none" stroke="{style.ice_ring_colour}" '
        f'stroke-width="{style.ice_ring_width:g}">',
    ]

    # Ring centres sit on the symbol centre, which in SVG coordinates is
    # half the side.
    for r in _ice_ring_radii(grid, style):
        out.append(
            f'<circle cx="{n / 2:g}" cy="{n / 2:g}" r="{r * grid.corner_distance:.3f}"/>'
        )
    out.append("</g>")
    return out


def _ice_boost(r: float, ring_radii: list[float], half_width: float) -> float:
    """Return the fractional diameter growth of a spot at normalised radius
    ``r``: ICE_SPOT_BOOST on a ring's centre line, tapering linearly to
    zero at its edge, ``half_width`` away."""
    return max(
        (
            ICE_SPOT_BOOST * (1.0 - abs(r - ring) / half_width)
            for ring in ring_radii
            if abs(r - ring) < half_width
        ),
        default=0.0,
    )


def wilson_intensity(d: float, b_factor: float) -> float:
    """Return the Wilson falloff exp(-B/(4d²)) at resolution ``d`` in Å, for
    an overall B factor in Å²; 1 at d = ∞.

    Wilson, A. J. C. (1942), Nature 150, 152, doi:10.1038/150152a0; the
    exponent is -2B(sin θ/λ)² with sin θ/λ = 1/(2d).
    """
    return math.exp(-b_factor / (4.0 * d * d))


def _falloff_diameter(r: float, style: DiffractionStyle) -> float:
    """Return the unjittered spot diameter at normalised radius ``r``.

    The radial falloff interpolates on r^FALLOFF_EXPONENT. The Wilson
    falloff interpolates on intensity, rescaled so the centre takes
    spot_diameter_centre and the corner module (r = 1) spot_diameter_edge
    whatever the B factor; B shapes the curve but never moves its ends.
    With no falloff every spot takes the midpoint of the two, keeping the
    overall ink coverage close to the falloff versions.
    """
    centre, edge = style.spot_diameter_centre, style.spot_diameter_edge
    if style.falloff == "none":
        return (centre + edge) / 2.0
    if style.falloff == "radial":
        return centre + (edge - centre) * r**FALLOFF_EXPONENT

    # With no decay to rescale over, every spot keeps the centre size.
    corner = wilson_intensity(resolution(1.0, style), style.wilson_b)
    if corner >= 1.0:
        return centre
    intensity = wilson_intensity(resolution(r, style), style.wilson_b)
    return edge + (centre - edge) * (intensity - corner) / (1.0 - corner)


def _spot_svg(
    cx: float, cy: float, width: float, length: float, angle: float, colour: str
) -> str:
    """Return one spot centred on (cx, cy): ``width`` across its long axis
    and ``length`` along it, which points ``angle`` degrees clockwise from
    +x. Unstreaked spots stay circles, so their markup is unchanged."""
    if length <= width:
        return (
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{width / 2:.3f}" fill="{colour}"/>'
        )
    return (
        f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{length / 2:.3f}" '
        f'ry="{width / 2:.3f}" transform="rotate({angle:.1f} {cx:.2f} {cy:.2f})" '
        f'fill="{colour}"/>'
    )


def _spot_layer(grid: CodeGrid, style: DiffractionStyle, seed: int) -> list[str]:
    """Draw a spot in every dark data module and speckle in some light ones.

    Uses ``random.Random(seed)`` directly rather than a per-layer stream
    so renders recorded before the layers were split stay reproducible.
    """
    rng = random.Random(seed)
    ring_radii = _ice_ring_radii(grid, style) if style.ice_rings else []
    ring_half_width = style.ice_ring_width / 2.0 / grid.corner_distance
    out = []
    for y in range(grid.size):
        for x in range(grid.size):
            if (x, y) in grid.reserved:
                continue
            cx, cy = x + 0.5, y + 0.5

            # Dark modules: large spots at the centre shrinking towards the
            # edge, jittered, and never below the safe sampling diameter.
            if grid.modules[y][x]:
                r = grid.radius(x, y)
                d = _falloff_diameter(r, style)
                d *= 1.0 + rng.uniform(-style.spot_jitter, style.spot_jitter)
                d = max(d, SAFE_SPOT_DIAMETER)

                # Spots on an ice ring grow, up to the cap; a spot already
                # past the cap is left as it is rather than shrunk.
                boost = _ice_boost(r, ring_radii, ring_half_width)
                if boost > 0.0:
                    d = min(d * (1.0 + boost), max(d, ICE_SPOT_CAP))

                # Streak across the radius: the radial axis keeps d, the
                # tangential axis lengthens with radius up to the cap.
                strength = style.streak_strength if style.streaks else 0.0
                length = min(d * (1.0 + strength * r), max(d, STREAK_CAP))
                angle = math.degrees(math.atan2(y - grid.centre, x - grid.centre))
                out.append(
                    _spot_svg(cx, cy, d, length, angle + 90.0, style.spot_colour)
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
    return out


def _pattern_layer(grid: CodeGrid, style: DiffractionStyle, seed: int) -> list[str]:
    """Draw the finders, then the alignment patterns, as rings."""
    out = []
    for fx, fy in grid.finders:
        out.extend(_finder_svg(fx + 0.5, fy + 0.5, style.spot_colour, grid.version))
    for ax, ay in grid.alignments:
        out.extend(_alignment_svg(ax + 0.5, ay + 0.5, style.spot_colour))
    return out


# A layer draws one kind of element over the whole grid and returns SVG
# elements in symbol coordinates. A layer that needs randomness seeds its
# own generator from the build seed and its name, so adding a layer
# never reshuffles another.
Layer = Callable[[CodeGrid, DiffractionStyle, int], list[str]]

# Layers in drawing order, bottom first.
LAYERS: tuple[Layer, ...] = (_ice_ring_layer, _spot_layer, _pattern_layer)


def build(
    url: str,
    seed: int = 0,
    style: DiffractionStyle | None = None,
    size_mm: float | None = None,
) -> StyledQr:
    """Encode ``url`` at error correction level H and render it as an SVG.

    The SVG viewBox is in module units. ``size_mm`` sets the printed side
    of the whole SVG, quiet zone included; without it the SVG is sized
    at one millimetre per module. ``seed`` fixes every random choice so a
    given URL and seed always render identically.
    """
    style = style or DiffractionStyle()
    grid = _encode(url)

    # A white page the size of the symbol plus its quiet zone, with the
    # layers drawn inside a group shifted by the quiet zone.
    width = grid.size + 2 * style.quiet_zone
    side = f"{size_mm:g}mm" if size_mm is not None else f"{width}mm"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {width}" '
        f'width="{side}" height="{side}">',
        f'<rect width="{width}" height="{width}" fill="#fff"/>',
        f'<g transform="translate({style.quiet_zone},{style.quiet_zone})">',
    ]
    for layer in LAYERS:
        out.extend(layer(grid, style, seed))
    out.append("</g></svg>")

    return StyledQr(
        svg="\n".join(out), size=grid.size, version=grid.version, width=width
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="data to encode")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for jitter and speckle"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("diffqr.svg"), help="SVG to write"
    )
    parser.add_argument(
        "--size-mm",
        type=float,
        help="printed side of the SVG in mm, quiet zone included",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="measure the blur margin of the result (needs the decode harness)",
    )

    # One switch per optional feature, defaulting to the style's own value.
    defaults = DiffractionStyle()
    features = parser.add_argument_group("optional features")
    for name, help_text in (
        ("speckle", "faint speckle in light modules"),
        ("ice_rings", "ice rings and the larger spots that sit on them"),
        ("streaks", "spots streaked across the radius"),
    ):
        features.add_argument(
            f"--{name.replace('_', '-')}",
            action=argparse.BooleanOptionalAction,
            default=getattr(defaults, name),
            help=help_text,
        )
    features.add_argument(
        "--falloff",
        choices=["wilson", "radial", "none"],
        default=defaults.falloff,
        help="how spot size falls off from the centre",
    )
    args = parser.parse_args(argv)
    style = dataclasses.replace(
        defaults,
        speckle=args.speckle,
        ice_rings=args.ice_rings,
        streaks=args.streaks,
        falloff=args.falloff,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    styled = build(args.url, seed=args.seed, style=style, size_mm=args.size_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(styled.svg)
    logger.info(
        "Wrote %s: version %d, %dx%d modules",
        args.output,
        styled.version,
        styled.size,
        styled.size,
    )

    # A physical size fixes the printed module, which is what a camera
    # has to resolve.
    if args.size_mm is not None:
        module_mm = args.size_mm / styled.width
        logger.info("Module size %.2f mm", module_mm)
        if module_mm < MIN_PRINT_MODULE_MM:
            logger.warning(
                "Module size %.2f mm is below the %.2f mm print minimum; "
                "shorten the URL or raise --size-mm to at least %.1f",
                module_mm,
                MIN_PRINT_MODULE_MM,
                MIN_PRINT_MODULE_MM * styled.width,
            )

    # The decode harness pulls in cairosvg, scipy and zxing-cpp, so it is
    # imported only when asked for.
    if args.check:
        from margin import MARGIN_FLOOR, blur_margin

        sigma = blur_margin(styled.svg, args.url)
        logger.info("Blur margin σ = %.3f module pitches", sigma)
        if sigma < MARGIN_FLOOR:
            logger.warning(
                "Blur margin σ = %.3f is below the %.3f floor; the code may not "
                "scan from a real camera, so shorten the URL or reduce styling",
                sigma,
                MARGIN_FLOOR,
            )


if __name__ == "__main__":
    main()
