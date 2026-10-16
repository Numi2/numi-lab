#!/usr/bin/env python3
"""Extract a bounded envelope from the embedded raster in Nagasawa Fig. 9.

This intentionally does not attempt to assign overlapping traces to the ten
legend entries. It converts the colored-pixel envelope into a grouped estimate
and marks the low response as censored when it merges with the zero axis.
The source PDF and extracted image hashes are fixed so another plot cannot be
silently substituted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SOURCE_PDF_SHA256 = "364dbc9c02d1a5bd6f7b1c2b309a2f3d502de791151b474d0989655f6f3f3e17"
FIGURE_IMAGE_SHA256 = "fd26155edee7bd320ee47ad1ba00a9a9de4371803f6165961778f853e8d3c091"
DEFAULT_PDF = Path("/Users/home/cardboard-research-20261006/primary-access/jstage-scoring-article.pdf")
DEFAULT_IMAGE = Path("/Users/home/cardboard-research-20261006/primary-access/figures/nagasawa-2012-fig9-embedded-image.jpg")

# Pixel-center calibration read from the source plot frame. The y pixel axis
# points down in the image, while bending moment increases upward.
AXES = {
    "x_pixel_min": 117.0,
    "x_pixel_max": 616.0,
    "x_min_deg": 0.0,
    "x_max_deg": 100.0,
    "y_pixel_top": 21.0,
    "y_pixel_bottom": 324.0,
    "y_min_Nm_per_m": 0.0,
    "y_max_Nm_per_m": 5.0,
}
ANGLE_SAMPLE_DEG = tuple(float(angle) for angle in range(1, 91))
COLUMN_RADIUS_PX = 1
COLOR_CHROMA_MIN = 8
MAX_CHANNEL_EXCLUSIVE = 253
READOUT_RADIUS_PX = 2
ZERO_AXIS_CENSOR_AT_OR_BELOW_PX = 320


class DigitizeError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bmp_rgb(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Read the 24-bit uncompressed BMP emitted by macOS sips, top-down RGB."""
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise DigitizeError("sips output is not a Windows BMP")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size, width, signed_height = struct.unpack_from("<Iii", data, 14)
    planes, bits_per_pixel = struct.unpack_from("<HH", data, 26)
    compression = struct.unpack_from("<I", data, 30)[0]
    if dib_size < 40 or planes != 1 or bits_per_pixel != 24 or compression != 0:
        raise DigitizeError("expected uncompressed 24-bit BMP from sips")
    height = abs(signed_height)
    row_stride = ((width * 24 + 31) // 32) * 4
    if width <= 0 or height <= 0 or pixel_offset + row_stride * height > len(data):
        raise DigitizeError("BMP dimensions or pixel extent are invalid")
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        stored_y = y if signed_height < 0 else height - 1 - y
        row = pixel_offset + stored_y * row_stride
        for x in range(width):
            offset = row + x * 3
            blue, green, red = data[offset : offset + 3]
            pixels.append((red, green, blue))
    return width, height, pixels


def pixel_to_moment(y_px: float) -> float:
    top = AXES["y_pixel_top"]
    bottom = AXES["y_pixel_bottom"]
    return AXES["y_max_Nm_per_m"] - (y_px - top) * (
        AXES["y_max_Nm_per_m"] - AXES["y_min_Nm_per_m"]
    ) / (bottom - top)


def extract_rows(width: int, height: int, pixels: list[tuple[int, int, int]],
                 source_pdf_sha256: str = SOURCE_PDF_SHA256,
                 figure_image_sha256: str = FIGURE_IMAGE_SHA256) -> list[dict[str, Any]]:
    if (width, height) != (647, 376) or len(pixels) != width * height:
        raise DigitizeError("Figure 9 raster dimensions must be exactly 647 x 376")
    if source_pdf_sha256 != SOURCE_PDF_SHA256 or figure_image_sha256 != FIGURE_IMAGE_SHA256:
        raise DigitizeError("source PDF or embedded figure image hash is not registered")

    rows: list[dict[str, Any]] = []
    x0, x1 = AXES["x_pixel_min"], AXES["x_pixel_max"]
    for angle in ANGLE_SAMPLE_DEG:
        x_center = round(x0 + angle * (x1 - x0) / (AXES["x_max_deg"] - AXES["x_min_deg"]))
        selected_y: list[int] = []
        for x in range(x_center - COLUMN_RADIUS_PX, x_center + COLUMN_RADIUS_PX + 1):
            if not (0 <= x < width):
                continue
            for y in range(23, ZERO_AXIS_CENSOR_AT_OR_BELOW_PX + 1):
                red, green, blue = pixels[y * width + x]
                channels = (red, green, blue)
                if max(channels) < MAX_CHANNEL_EXCLUSIVE and max(channels) - min(channels) >= COLOR_CHROMA_MIN:
                    selected_y.append(y)
        if not selected_y:
            raise DigitizeError(f"no trace pixels detected at {angle:g} degrees")

        high_moment_y_px = min(selected_y)
        low_moment_y_px = max(selected_y)
        lower_censored = low_moment_y_px >= ZERO_AXIS_CENSOR_AT_OR_BELOW_PX
        lower_moment = 0.0 if lower_censored else pixel_to_moment(low_moment_y_px)
        upper_moment = pixel_to_moment(high_moment_y_px)
        if not (math.isfinite(lower_moment) and math.isfinite(upper_moment) and lower_moment <= upper_moment):
            raise DigitizeError(f"invalid envelope at {angle:g} degrees")

        rows.append({
            "condition_id": "nagasawa_2012_fig9_cP3p0_cC0p7_V2p92",
            "angle_deg": f"{angle:.1f}",
            "moment_envelope_min_Nm_per_m": f"{lower_moment:.6f}",
            "moment_envelope_max_Nm_per_m": f"{upper_moment:.6f}",
            "lower_envelope_censored_at_zero_axis": str(lower_censored).lower(),
            "sample_x_pixel": str(x_center),
            "lower_envelope_y_pixel": str(low_moment_y_px),
            "upper_envelope_y_pixel": str(high_moment_y_px),
            "readout_uncertainty_angle_deg": f"{READOUT_RADIUS_PX * (AXES['x_max_deg'] - AXES['x_min_deg']) / (x1 - x0):.6f}",
            "readout_uncertainty_moment_Nm_per_m": f"{READOUT_RADIUS_PX * (AXES['y_max_Nm_per_m'] - AXES['y_min_Nm_per_m']) / (AXES['y_pixel_bottom'] - AXES['y_pixel_top']):.6f}",
            "observation_class": "digitized_group_envelope_estimate",
            "condition_group": "one_Fig9_test_condition_all_10_traces_pooled",
            "source_page": "PDF page 6; printed page 1268; Figure 9",
            "source_pdf_sha256": source_pdf_sha256,
            "embedded_image_sha256": figure_image_sha256,
        })
    return rows


FIELDNAMES = [
    "condition_id", "angle_deg", "moment_envelope_min_Nm_per_m",
    "moment_envelope_max_Nm_per_m", "lower_envelope_censored_at_zero_axis",
    "sample_x_pixel", "lower_envelope_y_pixel", "upper_envelope_y_pixel",
    "readout_uncertainty_angle_deg", "readout_uncertainty_moment_Nm_per_m",
    "observation_class", "condition_group", "source_page", "source_pdf_sha256",
    "embedded_image_sha256",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--figure-image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256(args.source_pdf) != SOURCE_PDF_SHA256:
        raise DigitizeError("retained J-STAGE PDF hash differs from the registered source")
    if sha256(args.figure_image) != FIGURE_IMAGE_SHA256:
        raise DigitizeError("retained Figure 9 image hash differs from the registered image")
    with tempfile.TemporaryDirectory(prefix="nagasawa-fig9-") as temp:
        bitmap = Path(temp) / "figure.bmp"
        result = subprocess.run(
            ["sips", "-s", "format", "bmp", str(args.figure_image), "--out", str(bitmap)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise DigitizeError(f"sips image conversion failed: {result.stderr.strip()}")
        width, height, pixels = read_bmp_rgb(bitmap)
    rows = extract_rows(width, height, pixels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise DigitizeError(f"refusing to overwrite existing dataset {args.output}")
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} grouped-envelope estimates: {args.output}")
    print("No trace identities, replicate distribution, or confidence interval were inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
