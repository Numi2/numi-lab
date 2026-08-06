#!/usr/bin/env python3
"""Render an exact Numi G1 solver-state trace as a compact GIF.

This is deliberately a presentation renderer, not a dynamics source: every
link point comes from ``g1_motion_retarget.npz`` produced by
``solver_trace_to_g1``.  It never interpolates, filters, floor-corrects, or
integrates the policy motion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHAINS = (
    ("pelvis", "left_hip_pitch_link", "left_hip_roll_link",
     "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link",
     "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link", "right_hip_roll_link",
     "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link",
     "right_ankle_roll_link"),
    ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"),
    ("torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
     "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
     "left_wrist_pitch_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
     "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
     "right_wrist_pitch_link", "right_wrist_yaw_link"),
)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retarget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=520)
    parser.add_argument("--fps", type=float, default=25.0)
    return parser.parse_args()


def project(point: np.ndarray, *, x0: float, x1: float, width: int, height: int) -> tuple[int, int]:
    # Fixed oblique orthographic view: forward displacement remains legible,
    # while left/right links do not collapse into one silhouette.
    margin_x = 48
    margin_y = 54
    horizontal = point[0] + 0.9 * point[1]
    px = margin_x + (horizontal - x0) / (x1 - x0) * (width - 2 * margin_x)
    py = height - margin_y - point[2] / 1.28 * (height - 2 * margin_y)
    return round(px), round(py)


def render_frame(
    transforms: np.ndarray,
    index: dict[str, int],
    frame: int,
    *,
    x0: float,
    x1: float,
    width: int,
    height: int,
    total: int,
) -> Image.Image:
    scale = 2
    image = Image.new("RGB", (width * scale, height * scale), (5, 12, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14 * scale)
    small = ImageFont.load_default(size=11 * scale)

    def point(name: str) -> tuple[int, int]:
        return project(
            transforms[frame, index[name], :3],
            x0=x0, x1=x1, width=width * scale, height=height * scale,
        )

    ground_y = height * scale - 54 * scale
    draw.line((0, ground_y, width * scale, ground_y), fill=(32, 96, 118), width=2 * scale)
    for meter in range(int(np.floor(x0)) - 1, int(np.ceil(x1)) + 2):
        grid_x, _ = project(
            np.array((float(meter), 0.0, 0.0)),
            x0=x0, x1=x1, width=width * scale, height=height * scale,
        )
        draw.line((grid_x, ground_y, grid_x, ground_y + 8 * scale), fill=(37, 76, 94), width=scale)
        draw.text((grid_x - 5 * scale, ground_y + 12 * scale), f"{meter}", font=small, fill=(105, 142, 156))

    colors = ((52, 209, 230), (255, 188, 71), (172, 232, 255), (180, 231, 255), (255, 207, 122))
    for chain, color in zip(CHAINS, colors):
        if not all(name in index for name in chain):
            continue
        points = [point(name) for name in chain]
        draw.line(points, fill=color, width=6 * scale, joint="curve")
        for position in points:
            radius = 4 * scale
            draw.ellipse(
                (position[0] - radius, position[1] - radius,
                 position[0] + radius, position[1] + radius),
                fill=(230, 245, 250), outline=color, width=2 * scale,
            )
    torso = point("torso_link")
    draw.ellipse(
        (torso[0] - 14 * scale, torso[1] - 16 * scale,
         torso[0] + 14 * scale, torso[1] + 16 * scale),
        fill=(212, 242, 252), outline=(70, 204, 230), width=3 * scale,
    )

    draw.rounded_rectangle(
        (18 * scale, 16 * scale, 378 * scale, 58 * scale), radius=8 * scale,
        fill=(8, 25, 42), outline=(38, 114, 140), width=scale,
    )
    draw.text((30 * scale, 25 * scale), "NUMI LAB  |  G1 LEGS LOCOMOTION", font=font, fill=(211, 243, 250))
    draw.text(
        (30 * scale, 43 * scale),
        f"accepted solver state  •  {frame + 1:03d}/{total:03d}",
        font=small, fill=(122, 181, 199),
    )
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    args = options()
    with np.load(args.retarget, allow_pickle=False) as archive:
        names = [str(value) for value in archive["link_names"]]
        transforms = np.asarray(archive["link_position_quaternion_xyzw"], dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[2] != 7 or not np.isfinite(transforms).all():
        raise ValueError("retarget artifact has invalid transforms")
    index = {name: value for value, name in enumerate(names)}
    if "pelvis" not in index or "torso_link" not in index:
        raise ValueError("retarget artifact is not a G1 link-state stream")
    x = (
        transforms[:, index["pelvis"], 0]
        + 0.9 * transforms[:, index["pelvis"], 1]
    )
    x0 = float(np.min(x) - 0.55)
    x1 = float(np.max(x) + 0.75)
    frames = [
        render_frame(transforms, index, frame, x0=x0, x1=x1,
                     width=args.width, height=args.height,
                     total=transforms.shape[0])
        for frame in range(transforms.shape[0])
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000.0 / args.fps),
        loop=0,
        disposal=2,
        optimize=False,
    )


if __name__ == "__main__":
    main()
