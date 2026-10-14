#!/usr/bin/env python3
"""Render recorded synthetic-skin wound-lip traction samples as an MP4."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import tempfile


WIDTH = 1280
HEIGHT = 720

FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    ":": ("00000", "00110", "00110", "00000", "00110", "00110", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "%": ("11001", "11010", "00100", "01000", "10110", "00110", "00000"),
}


class Canvas:
    def __init__(self) -> None:
        self.pixels = bytearray([248, 249, 247] * WIDTH * HEIGHT)

    def pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            index = (y * WIDTH + x) * 3
            self.pixels[index : index + 3] = bytes(color)

    def line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            for oy in range(-(thickness // 2), thickness // 2 + 1):
                for ox in range(-(thickness // 2), thickness // 2 + 1):
                    self.pixel(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    def circle(
        self, x: int, y: int, radius: int, color: tuple[int, int, int]
    ) -> None:
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if ox * ox + oy * oy <= radius * radius:
                    self.pixel(x + ox, y + oy, color)

    def text(
        self,
        x: int,
        y: int,
        value: str,
        color: tuple[int, int, int],
        scale: int = 2,
    ) -> None:
        cursor = x
        for character in value.upper():
            glyph = FONT.get(character)
            if glyph is None:
                cursor += 4 * scale
                continue
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        for oy in range(scale):
                            for ox in range(scale):
                                self.pixel(
                                    cursor + column * scale + ox,
                                    y + row * scale + oy,
                                    color,
                                )
            cursor += 6 * scale

    def ppm(self, path: Path) -> None:
        with path.open("wb") as stream:
            stream.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
            stream.write(self.pixels)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def map_xy(
    x_m: float,
    y_m: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    x = left + round((x_m * 1000.0 + 10.0) / 20.0 * width)
    y = top + round((3.0 - y_m * 1000.0) / 6.0 * height)
    return x, y


def render_frame(
    lip_rows: list[dict[str, str]],
    sample: dict[str, str],
    samples: list[dict[str, str]],
    frame_index: int,
    control_samples: list[dict[str, str]] | None = None,
    failure_label: str | None = None,
) -> Canvas:
    canvas = Canvas()
    navy = (28, 45, 55)
    muted = (107, 122, 124)
    grid = (220, 226, 222)
    red = (177, 79, 74)
    blue = (61, 121, 151)
    green = (44, 132, 108)
    x0, y0, plot_width, plot_height = 96, 180, 620, 390
    chart_x, chart_y, chart_width, chart_height = 850, 260, 330, 230

    canvas.text(68, 46, "SYNTHETIC SKIN / WOUND-LIP TRACTION", navy, 3)
    canvas.text(70, 92, "PAIRED BITE FORCE  |  RESEARCH MODEL  |  NOT CALIBRATED", muted, 1)
    if failure_label:
        canvas.text(70, 114, failure_label, red, 1)
    canvas.text(76, 142, "TOP VIEW: OPEN INCISION LIPS", navy, 2)
    canvas.text(850, 142, "WOUND GAP RESPONSE", navy, 2)

    for x_tick in range(-10, 11, 2):
        px, _ = map_xy(x_tick / 1000.0, 0.0, x0, y0, plot_width, plot_height)
        canvas.line(px, y0, px, y0 + plot_height, grid)
        canvas.text(px - 8, y0 + plot_height + 18, str(x_tick), muted, 1)
    for y_tick in range(-3, 4):
        _, py = map_xy(0.0, y_tick / 1000.0, x0, y0, plot_width, plot_height)
        canvas.line(x0, py, x0 + plot_width, py, grid)
    canvas.line(x0, y0 + plot_height, x0 + plot_width, y0 + plot_height, muted)
    canvas.line(x0, y0, x0, y0 + plot_height, muted)
    canvas.text(x0 + plot_width // 2 - 15, y0 + plot_height + 48, "X MM", muted, 1)
    canvas.text(20, y0 + plot_height // 2, "Y", muted, 2)

    lower_points: list[tuple[int, int]] = []
    upper_points: list[tuple[int, int]] = []
    for row in sorted(lip_rows, key=lambda value: int(value["pair"])):
        lower_points.append(
            map_xy(
                float(row["lower_x_m"]),
                float(row["lower_y_m"]),
                x0,
                y0,
                plot_width,
                plot_height,
            )
        )
        upper_points.append(
            map_xy(
                float(row["upper_x_m"]),
                float(row["upper_y_m"]),
                x0,
                y0,
                plot_width,
                plot_height,
            )
        )
    for points, color in ((lower_points, red), (upper_points, blue)):
        for start, end in zip(points, points[1:]):
            canvas.line(start[0], start[1], end[0], end[1], color, 4)
        for point in points:
            canvas.circle(point[0], point[1], 4, color)

    for bite_x in (-0.004, 0.004):
        lower = min(
            lip_rows,
            key=lambda row: abs(float(row["rest_x_m"]) - bite_x),
        )
        center_x = float(lower["rest_x_m"])
        midpoint_y = 0.5 * (
            float(lower["lower_y_m"]) + float(lower["upper_y_m"])
        )
        for sign in (-1, 1):
            target_x, target_y = map_xy(
                center_x,
                midpoint_y + sign * 0.00055,
                x0,
                y0,
                plot_width,
                plot_height,
            )
            inward_y = map_xy(
                center_x,
                midpoint_y,
                x0,
                y0,
                plot_width,
                plot_height,
            )[1]
            canvas.line(target_x, target_y, target_x, inward_y, green, 2)
            canvas.circle(target_x, target_y, 5, green)

    canvas.text(105, 610, "LOWER LIP", red, 1)
    canvas.text(235, 610, "UPPER LIP", blue, 1)
    canvas.text(368, 610, "BITE SITES / INWARD TRACTION", green, 1)

    for index in range(6):
        px = chart_x + round(index / 5 * chart_width)
        canvas.line(px, chart_y, px, chart_y + chart_height, grid)
    gap_rows = list(samples)
    if control_samples:
        gap_rows.extend(control_samples)
    min_gap = min(float(row["mean_gap_m"]) for row in gap_rows)
    max_gap = max(float(row["mean_gap_m"]) for row in gap_rows)
    gap_span = max(max_gap - min_gap, 1.0e-8)
    for tick in range(5):
        py = chart_y + round(tick / 4 * chart_height)
        canvas.line(chart_x, py, chart_x + chart_width, py, grid)
    canvas.line(chart_x, chart_y + chart_height, chart_x + chart_width, chart_y + chart_height, muted)
    canvas.line(chart_x, chart_y, chart_x, chart_y + chart_height, muted)
    gap_points = []
    for index, row in enumerate(samples):
        x = chart_x + round(index / max(1, len(samples) - 1) * chart_width)
        y = chart_y + round(
            (max_gap - float(row["mean_gap_m"])) / gap_span * chart_height
        )
        gap_points.append((x, y))
    for start, end in zip(gap_points, gap_points[1:]):
        canvas.line(start[0], start[1], end[0], end[1], blue, 3)
    current = gap_points[frame_index]
    canvas.circle(current[0], current[1], 7, green)
    if control_samples:
        control_points = []
        for index, row in enumerate(control_samples):
            x = chart_x + round(index / max(1, len(control_samples) - 1) * chart_width)
            y = chart_y + round(
                (max_gap - float(row["mean_gap_m"])) / gap_span * chart_height
            )
            control_points.append((x, y))
        for start, end in zip(control_points, control_points[1:]):
            canvas.line(start[0], start[1], end[0], end[1], red, 2)
        canvas.line(850, 205, 880, 205, blue, 3)
        canvas.text(888, 198, "LOADED", blue, 1)
        canvas.line(1000, 205, 1030, 205, red, 2)
        canvas.text(1038, 198, "0 N CONTROL", red, 1)
    canvas.text(chart_x - 24, chart_y + chart_height + 20, "0", muted, 1)
    chart_end_s = float(samples[-1]["time_s"])
    canvas.text(chart_x + chart_width - 44, chart_y + chart_height + 20, f"{chart_end_s:.3f} S", muted, 1)
    canvas.text(850, 520, f"LOADED {float(sample['mean_gap_m']) * 1000:.3f} MM", blue, 2)
    if control_samples:
        control_gap = float(control_samples[frame_index]["mean_gap_m"])
        canvas.text(850, 554, f"CONTROL {control_gap * 1000:.3f} MM", red, 2)
        canvas.text(850, 588, f"T {float(sample['time_s']):.3f} S", muted, 2)
        canvas.text(850, 622, f"BITE FORCE {float(sample['bite_force_n']):.2E} N", green, 2)
    else:
        canvas.text(850, 562, f"T {float(sample['time_s']):.3f} S", muted, 2)
        canvas.text(850, 606, f"BITE FORCE {float(sample['bite_force_n']):.2E} N", green, 2)
    footer = (
        "EARLY STOP / 0.006 S SIMULATED AT 1 FPS / NO WOUND CLOSURE"
        if failure_label
        else "SIMULATION OUTPUT: EXPLICIT PAIRED FORCE; NO THREAD OR ROBOT IN THIS DEMO"
    )
    canvas.text(70, 680, footer, muted, 1)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--lips", type=Path, required=True)
    parser.add_argument("--control-samples", type=Path)
    parser.add_argument("--failure-label", help="persistent label for an incomplete or failed run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    samples = read_rows(args.samples)
    lip_rows = read_rows(args.lips)
    control_samples = read_rows(args.control_samples) if args.control_samples else None
    if not samples or not lip_rows:
        raise SystemExit("traction CSVs are empty")
    if control_samples is not None:
        control_by_step = {int(row["step"]): row for row in control_samples}
        missing_control = [int(row["step"]) for row in samples if int(row["step"]) not in control_by_step]
        if missing_control:
            raise SystemExit(f"control is missing matched loaded steps: {missing_control}")
        control_samples = [control_by_step[int(row["step"])] for row in samples]
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    by_step: dict[int, list[dict[str, str]]] = {}
    for row in lip_rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    missing_lip_steps = [
        int(row["step"]) for row in samples if int(row["step"]) not in by_step
    ]
    if missing_lip_steps:
        raise SystemExit(f"missing lip geometry for steps: {missing_lip_steps}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to encode the showcase MP4")
    with tempfile.TemporaryDirectory(prefix="skin-traction-video-") as temporary:
        frame_directory = Path(temporary)
        for index, sample in enumerate(samples):
            step = int(sample["step"])
            frame = render_frame(
                by_step.get(step, []), sample, samples, index, control_samples,
                args.failure_label,
            )
            frame.ppm(frame_directory / f"frame-{index:04d}.ppm")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_directory / "frame-%04d.ppm"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
        subprocess.run(command, check=True)
    print(f"video={args.output}")
    print(f"frames={len(samples)} fps={args.fps}")
    print(f"initial_mean_gap_mm={float(samples[0]['mean_gap_m']) * 1000:.6f}")
    print(f"minimum_mean_gap_mm={min(float(row['mean_gap_m']) for row in samples) * 1000:.6f}")
    print(f"final_mean_gap_mm={float(samples[-1]['mean_gap_m']) * 1000:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
