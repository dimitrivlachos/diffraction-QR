#!/usr/bin/env bash
# Build the three QR codes on the back of the MX Data Analysis postcard.
#
# Each size is the white backing the code occupies on the card, measured
# from Data_Analysis_PostCard.svg, so the SVGs drop into place at 1:1.
# Run from anywhere with the project environment active.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p out

python diffraction.py \
    "https://dials.github.io/documentation/tutorials/index.html" \
    --size-mm 23.5 --check -o out/dials.svg

python diffraction.py \
    "https://xia2.github.io" \
    --size-mm 23.5 --check -o out/xia2.svg

python diffraction.py \
    "https://www.diamond.ac.uk/Instruments/Mx/MX-Data-Analysis/Interpreting-Experiment-Feedback.html" \
    --size-mm 29.4 --check -o out/ispyb.svg
