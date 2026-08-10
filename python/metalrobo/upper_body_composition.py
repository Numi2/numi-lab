"""Compose generated G1 upper-body intent with a qualified standing base."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from .ardy_interaction_convert import (
    _G1_JOINT_LOWER,
    _G1_JOINT_UPPER,
    _G1_JOINT_VELOCITY,
    convert,
)


STANDING_JOINTS = np.asarray(
    (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    ),
    dtype=np.float32,
)
UPPER_BODY_START = 15
COMPOSED_FRAME_COUNT = 16
FRAMES_PER_SECOND = 50.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose_upper_body(
    source_joints: np.ndarray,
    *,
    standing_joints: np.ndarray = STANDING_JOINTS,
    active_arms: tuple[str, ...] = ("left", "right"),
    shoulder_target: float = -0.65,
    frames_per_second: float = FRAMES_PER_SECOND,
) -> np.ndarray:
    source = np.asarray(source_joints, dtype=np.float32)
    base = np.asarray(standing_joints, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 29 or source.shape[0] < 2:
        raise ValueError("source_joints must have shape [frames>=2, 29]")
    if base.shape != (29,) or not np.all(np.isfinite(source)) or not np.all(np.isfinite(base)):
        raise ValueError("composition inputs must be finite G1 joint arrays")
    result = np.tile(base, (source.shape[0], 1))
    arm_slices = {"left": slice(15, 22), "right": slice(22, 29)}
    if not active_arms or any(arm not in arm_slices for arm in active_arms):
        raise ValueError("active_arms must contain left and/or right")
    if not np.isfinite(shoulder_target) or not np.isfinite(frames_per_second) or frames_per_second <= 0:
        raise ValueError("shoulder_target and frames_per_second must be finite")
    for arm in active_arms:
        joint_slice = arm_slices[arm]
        delta = source[:, joint_slice] - source[0, joint_slice]
        shoulder_delta = float(delta[-1, 0])
        if shoulder_delta >= -1.0e-3:
            raise ValueError(f"{arm} ARDY proposal does not raise its shoulder")
        requested_scale = (
            shoulder_target - float(base[joint_slice.start])
        ) / shoulder_delta
        maximum_scale = float("inf")
        for local, joint in enumerate(range(joint_slice.start, joint_slice.stop)):
            values = delta[:, local]
            positive = values > 0.0
            negative = values < 0.0
            if np.any(positive):
                maximum_scale = min(
                    maximum_scale,
                    float(np.min(
                        (_G1_JOINT_UPPER[joint] - base[joint]) /
                        values[positive]
                    )),
                )
            if np.any(negative):
                maximum_scale = min(
                    maximum_scale,
                    float(np.min(
                        (_G1_JOINT_LOWER[joint] - base[joint]) /
                        values[negative]
                    )),
                )
            peak_delta_velocity = float(
                np.max(np.abs(np.diff(values))) * frames_per_second
            )
            if peak_delta_velocity > 0.0:
                maximum_scale = min(
                    maximum_scale,
                    float(_G1_JOINT_VELOCITY[joint]) /
                    peak_delta_velocity,
                )
        scale = min(requested_scale, maximum_scale * (1.0 - 1.0e-5))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{arm} ARDY proposal has no feasible positive scale")
        result[:, joint_slice] += delta * scale
    return result


def _resample(source: np.ndarray, frame_count: int) -> np.ndarray:
    if frame_count < 2:
        raise ValueError("composed frame count must be at least two")
    source_phase = np.linspace(0.0, 1.0, source.shape[0])
    target_phase = np.linspace(0.0, 1.0, frame_count)
    return np.stack([
        np.interp(target_phase, source_phase, source[:, joint])
        for joint in range(source.shape[1])
    ], axis=1).astype(np.float32)


def compose_pack(
    *, source_retarget: Path, output_directory: Path, task_id: str,
    secondary_source_retarget: Path | None = None,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    with np.load(source_retarget, allow_pickle=False) as archive:
        source = np.asarray(archive["joint_positions"], dtype=np.float32)
    active_arms = {
        "raise-left-hand": ("left",),
        "raise-right-hand": ("right",),
        "raise-both-hands": ("left", "right"),
    }.get(task_id)
    if active_arms is None:
        raise ValueError("task-id must be raise-left-hand, raise-right-hand, or raise-both-hands")
    source = _resample(source, COMPOSED_FRAME_COUNT)
    source_paths = [source_retarget]
    if task_id == "raise-both-hands" and secondary_source_retarget is not None:
        with np.load(secondary_source_retarget, allow_pickle=False) as archive:
            secondary = _resample(
                np.asarray(archive["joint_positions"], dtype=np.float32),
                COMPOSED_FRAME_COUNT,
            )
        # Bilateral behavior is a literal composition of the two already
        # qualified unilateral ARDY proposals, not a third unrelated motion.
        source[:, 22:29] = secondary[:, 22:29]
        source_paths.append(secondary_source_retarget)
    composed = compose_upper_body(
        source,
        active_arms=active_arms,
        frames_per_second=FRAMES_PER_SECOND,
    )
    frame_count = composed.shape[0]
    root_wxyz = np.tile(
        np.asarray((0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        (frame_count, 1),
    )
    qpos = np.concatenate((root_wxyz, composed), axis=1)
    qpos_path = output_directory / "composed-qpos.csv"
    motion_path = output_directory / "composed-motion.npz"
    pack_path = output_directory / "motion.interactionpack"
    np.savetxt(qpos_path, qpos, delimiter=",", fmt="%.9g")
    np.savez(
        motion_path,
        foot_contacts=np.ones((frame_count, 4), dtype=np.float32),
        fps=np.asarray((FRAMES_PER_SECOND,), dtype=np.float32),
        text=np.asarray((task_id.replace("-", " "),)),
    )
    convert(SimpleNamespace(
        motion_npz=motion_path, qpos_csv=qpos_path, output=pack_path,
        id=f"pqi2-{task_id}", clip_id=task_id,
        desired_outcome=task_id.replace("-", " "),
        source_repository="TREEIndustries/ARDY-G1 + Numi qualified standing base",
        source_revision="+".join(_sha256(path) for path in source_paths),
        license="NVIDIA Open Model License",
        left_contact_group="left_foot_contact", right_contact_group="right_foot_contact",
        counterpart="locomotion_ground", contact_confidence=0.5,
        joint_limit_margin=1.0e-4, loop=False,
    ))
    evidence = {
        "schema": "numi.upper-body-composition.v1",
        "task": task_id,
        "source_retarget": str(source_retarget),
        "source_retarget_sha256": _sha256(source_retarget),
        "source_retargets": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in source_paths
        ],
        "interaction_pack": str(pack_path),
        "interaction_pack_sha256": _sha256(pack_path),
        "frame_count": frame_count,
        "generated_joint_indices": [
            index for arm in active_arms
            for index in ({"left": range(15, 22), "right": range(22, 29)}[arm])
        ],
        "fixed_joint_indices": [
            index for index in range(29)
            if not any(index in ({"left": range(15, 22), "right": range(22, 29)}[arm]) for arm in active_arms)
        ],
        "root_base": "fixed qualified standing reset",
        "lower_body_base": "fixed qualified standing reset",
        "shoulder_endpoints_rad": {
            arm: float(composed[-1, {"left": 15, "right": 22}[arm]])
            for arm in active_arms
        },
        "upper_body_semantics": "ARDY within-arm trajectory rebased at frame zero and uniformly scaled per arm to the closest position-and-velocity-qualified shoulder endpoint",
        "frames_per_second": FRAMES_PER_SECOND,
        "contact_semantics": "predicted bilateral support only; no force or pressure authored",
    }
    (output_directory / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    return evidence


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-retarget", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--secondary-source-retarget", type=Path)
    options = parser.parse_args(arguments)
    print(json.dumps(compose_pack(
        source_retarget=options.source_retarget,
        output_directory=options.output_directory,
        task_id=options.task_id,
        secondary_source_retarget=options.secondary_source_retarget,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
