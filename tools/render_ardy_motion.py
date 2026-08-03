#!/usr/bin/env python3
"""Render a Numi ARDY motion proposal as a provenance-labelled MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RIGHT_ARM = {7, 8, 9, 10, 11, 12}


def _posed_joints(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    root = np.asarray(archive["root_positions"], dtype=np.float32)
    local = np.asarray(archive["local_joint_positions"], dtype=np.float32)
    joints = np.concatenate((root[:, None, :], local), axis=1)
    joints[:, 1:, 0] += root[:, None, 0]
    joints[:, 1:, 2] += root[:, None, 2]
    if joints.shape != (40, 27, 3) or not np.isfinite(joints).all():
        raise ValueError("expected one finite 40-frame cskel27 proposal")
    return joints


def render(proposal_directory: Path, output: Path, loops: int) -> None:
    evidence = json.loads(
        (proposal_directory / "evidence.json").read_text(encoding="utf-8")
    )
    with np.load(proposal_directory / "motion_proposal.npz") as archive:
        joints = _posed_joints(archive)
    parents = evidence["skeleton"]["joint_parents"]
    fps = int(evidence["fps"])
    if len(parents) != joints.shape[1]:
        raise ValueError("joint parent table does not match motion")

    display_fps = 60
    model_seconds = joints.shape[0] / fps
    frame_count = round(model_seconds * display_fps * loops)
    fig = plt.figure(figsize=(8, 10), dpi=150, facecolor="#050812")
    ax = fig.add_axes((0, 0, 1, 1), facecolor="#050812")
    ax.set_xlim(-0.92, 0.56)
    ax.set_ylim(-0.28, 1.94)
    ax.set_aspect("equal")
    ax.axis("off")

    for y in np.arange(0.0, 1.9, 0.2):
        ax.axhline(y, color="#18223a", linewidth=0.55, alpha=0.32, zorder=0)
    for x in np.arange(-0.8, 0.6, 0.2):
        ax.axvline(x, color="#18223a", linewidth=0.55, alpha=0.25, zorder=0)
    ax.axhline(0.0, color="#5de2ff", linewidth=1.1, alpha=0.48, zorder=0)

    bones = []
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        color = "#ffb45e" if child in RIGHT_ARM else "#d9f7ff"
        width = 5.2 if child in RIGHT_ARM else 4.2
        (line,) = ax.plot([], [], color=color, linewidth=width,
                          solid_capstyle="round", zorder=4)
        bones.append((child, parent, line))
    body = ax.scatter([], [], s=28, color="#f4fcff", edgecolor="#6ddcf7",
                      linewidth=0.7, zorder=5)
    hand = ax.scatter([], [], s=115, color="#ffb45e", edgecolor="#fff0d4",
                      linewidth=1.4, zorder=6)
    (trail,) = ax.plot([], [], color="#ffb45e", linewidth=2.2, alpha=0.35,
                       zorder=2)
    shadow = plt.Circle((0, 0.005), 0.27, color="#000000", alpha=0.38, zorder=1)
    ax.add_patch(shadow)

    ax.text(
        0.055, 0.953, "ARDY · APPLE SILICON", transform=ax.transAxes,
        color="#5de2ff", fontsize=12, weight="bold", ha="left", va="top",
    )
    ax.text(0.055, 0.912, evidence["prompt"], transform=ax.transAxes,
            color="#f5fbff", fontsize=20, weight="bold", ha="left", va="top")
    ax.text(0.055, 0.865,
            f"{evidence['skeleton']['id']}  ·  {fps} FPS  ·  {evidence['frame_count']} generated frames",
            transform=ax.transAxes, color="#8ca3bd", fontsize=10,
            ha="left", va="top")
    ax.text(0.055, 0.055,
            f"ONNX Runtime · {evidence['selected_provider'].upper()} provider · arm64 macOS",
            transform=ax.transAxes, color="#90a8c1", fontsize=9,
            ha="left", va="bottom")
    ax.text(0.945, 0.055, "Numi Lab motion proposal · physics not applied",
            transform=ax.transAxes, color="#90a8c1", fontsize=9,
            ha="right", va="bottom")
    progress_bg = plt.Rectangle((0.055, 0.028), 0.89, 0.004,
                                transform=ax.transAxes, color="#18283e",
                                linewidth=0, zorder=8)
    progress = plt.Rectangle((0.055, 0.028), 0.0, 0.004,
                             transform=ax.transAxes, color="#5de2ff",
                             linewidth=0, zorder=9)
    ax.add_patch(progress_bg)
    ax.add_patch(progress)

    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgba", "-s", "1200x1500", "-r", str(display_fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
            "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for display_index in range(frame_count):
            phase = (display_index / display_fps) % model_seconds
            source = phase * fps
            first = min(int(source), joints.shape[0] - 1)
            second = min(first + 1, joints.shape[0] - 1)
            blend = source - first
            pose = joints[first] * (1.0 - blend) + joints[second] * blend
            x = pose[:, 0] + 0.18 * pose[:, 2]
            y = pose[:, 1]
            for child, parent, line in bones:
                line.set_data((x[parent], x[child]), (y[parent], y[child]))
            body.set_offsets(np.column_stack((x, y)))
            hand.set_offsets([[x[11], y[11]]])
            history_start = max(0, first - 10)
            history = joints[history_start:first + 1, 11]
            trail.set_data(history[:, 0] + 0.18 * history[:, 2], history[:, 1])
            shadow.center = (x[0], 0.005)
            progress.set_width(0.89 * (source / (joints.shape[0] - 1)))
            fig.canvas.draw()
            assert ffmpeg.stdin is not None
            ffmpeg.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    finally:
        if ffmpeg.stdin is not None:
            ffmpeg.stdin.close()
        return_code = ffmpeg.wait()
        plt.close(fig)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proposal_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--loops", type=int, default=3)
    arguments = parser.parse_args()
    render(arguments.proposal_directory, arguments.output, arguments.loops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
