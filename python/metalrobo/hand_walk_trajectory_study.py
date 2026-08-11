"""Qualify complete ARDY hand-to-walk trajectories without reward shortcuts.

The module deliberately evaluates one canonical trajectory at a time.  It is
the admission authority for a later correction dataset, not a trainer and not
a policy selector.  Solver traces are the source of every physical metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


_SCHEMA = "numi.pqi3-trajectory-correction-protocol.v1"
_WRIST_Z_COLUMN = {"left": 3, "right": 12}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tsv(path: Path, *, minimum_columns: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, comments="#", dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < minimum_columns:
        raise ValueError(f"{path} has an incompatible trace layout")
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{path} is non-finite or has non-monotonic steps")
    return values


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != _SCHEMA or protocol.get("status") != "preregistered":
        raise ValueError("trajectory-correction protocol must be preregistered v1")
    phases = protocol.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("protocol must declare ordered phases")
    last_frame = -1
    for phase in phases:
        first = phase.get("first_reference_frame")
        last = phase.get("last_reference_frame")
        if not isinstance(first, int) or not isinstance(last, int) or first != last_frame + 1 or last < first:
            raise ValueError("protocol phases must form one contiguous reference sequence")
        last_frame = last
    if last_frame + 1 != protocol["scope"].get("reference_frames"):
        raise ValueError("protocol phases must cover the complete reference")
    return protocol


def _tilt_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return root-up tilt, derived directly from accepted root orientation."""

    q = np.asarray(quaternion, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1.0e-8):
        raise ValueError("state trace contains a zero root quaternion")
    x, y, z, w = (q[:, index] / norms for index in range(4))
    root_up_dot_world_up = 1.0 - 2.0 * (x * x + y * y)
    return np.arccos(np.clip(root_up_dot_world_up, -1.0, 1.0))


def _phase_rows(trace: np.ndarray, frame_position: np.ndarray, first_frame: int, last_frame: int) -> np.ndarray:
    rows = trace[(frame_position >= float(first_frame)) & (frame_position <= float(last_frame))]
    return rows


def _interaction_frame_positions(protocol: dict[str, Any], rollout_pack_path: Path) -> np.ndarray:
    """Recover the solver-gated reference clock from the compiled observation ABI."""

    from metalrobo.mlx_policy_learning import read_policy_rollout_pack

    rollout = read_policy_rollout_pack(rollout_pack_path)
    columns = protocol["scope"].get("interaction_phase_observation_columns")
    if (
        rollout.environment_count != 1
        or not isinstance(columns, list)
        or len(columns) != 3
        or any(not isinstance(column, int) or column < 0 or column >= rollout.actor_observation_count for column in columns)
    ):
        raise ValueError("rollout does not expose the protocol's interaction-phase observation ABI")
    observations = rollout.actor_observations.reshape(
        rollout.control_step_count, rollout.actor_observation_count
    )
    phase = observations[:, columns]
    # The third component is the normalized selected-reference frame.  The
    # sine/cosine pair gives an independent contract check instead of trusting
    # an arbitrary observation column as phase.
    expected_sine = np.sin(2.0 * math.pi * phase[:, 2])
    expected_cosine = np.cos(2.0 * math.pi * phase[:, 2])
    if (
        not np.isfinite(phase).all()
        or np.any(phase[:, 2] < -1.0e-5)
        or np.any(phase[:, 2] > 1.0 + 1.0e-5)
        or np.max(np.abs(phase[:, 0] - expected_sine), initial=0.0) > 2.0e-3
        or np.max(np.abs(phase[:, 1] - expected_cosine), initial=0.0) > 2.0e-3
    ):
        raise ValueError("interaction-phase observation does not satisfy its sine/cosine contract")
    return phase[:, 2] * float(protocol["scope"]["reference_frames"] - 1)


def analyze(
    protocol_path: Path,
    *,
    interaction_pack: Path,
    native_evidence_path: Path,
    state_trace_path: Path,
    motion_trace_path: Path,
    rollout_pack_path: Path,
    replay_state_trace_path: Path | None = None,
    replay_motion_trace_path: Path | None = None,
) -> dict[str, Any]:
    """Produce an admission decision for one full sequence and optional replay."""

    protocol = _load_protocol(protocol_path)
    state = _load_tsv(state_trace_path, minimum_columns=8)
    motion = _load_tsv(motion_trace_path, minimum_columns=13)
    native = json.loads(native_evidence_path.read_text(encoding="utf-8"))
    if not interaction_pack.is_file():
        raise FileNotFoundError(interaction_pack)
    from metalrobo.ardy_interaction_convert import read_interaction_pack
    pack = read_interaction_pack(interaction_pack)
    matching_clips = [
        clip for clip in pack.clips
        if clip.id == protocol["scope"]["canonical_clip"]
    ]
    if len(matching_clips) != 1 or matching_clips[0].root_targets.shape[0] != protocol["scope"]["reference_frames"]:
        raise ValueError("interaction pack does not match the protocol's canonical reference")
    gates = protocol["gates"]
    frame_position = _interaction_frame_positions(protocol, rollout_pack_path)
    if state.shape[0] != frame_position.size or motion.shape[0] != frame_position.size:
        raise ValueError("state, motion, and interaction-phase traces must share one control horizon")
    trace_coverage = float(np.max(frame_position)) >= float(protocol["scope"]["reference_frames"] - 1) - 1.0e-4
    tilt = _tilt_from_xyzw(state[:, 4:8])
    phase_metrics: list[dict[str, Any]] = []
    wrist_gate_passed = True
    walk_metrics: dict[str, float] | None = None
    for phase in protocol["phases"]:
        first_frame = int(phase["first_reference_frame"])
        last_frame = int(phase["last_reference_frame"])
        motion_rows = _phase_rows(motion, frame_position, first_frame, last_frame)
        state_rows = _phase_rows(state, frame_position, first_frame, last_frame)
        reached = motion_rows.size > 0 and state_rows.size > 0
        if not reached:
            entry = {
                "id": phase["id"],
                "first_reference_frame": first_frame,
                "last_reference_frame": last_frame,
                "reached": False,
                "wrists": [],
            }
            wrist_gate_passed = wrist_gate_passed and not phase["required_wrists"]
            if phase["id"] == "walk":
                walk_metrics = {
                    "forward_displacement_m": None,
                    "lateral_displacement_m": None,
                }
                entry["walk"] = walk_metrics
            phase_metrics.append(entry)
            continue
        wrists = []
        for side in phase["required_wrists"]:
            heights = motion_rows[:, _WRIST_Z_COLUMN[side]]
            peak = float(np.max(heights))
            excursion = float(peak - heights[0])
            passed = peak >= gates["minimum_wrist_peak_height_m"] and excursion >= gates["minimum_wrist_upward_excursion_m"]
            wrists.append({"side": side, "peak_height_m": peak, "upward_excursion_m": excursion, "passed": passed})
            wrist_gate_passed = wrist_gate_passed and passed
        entry: dict[str, Any] = {
            "id": phase["id"],
            "first_reference_frame": first_frame,
            "last_reference_frame": last_frame,
            "reached": True,
            "minimum_root_height_m": float(np.min(state_rows[:, 3])),
            "maximum_tilt_rad": float(np.max(_tilt_from_xyzw(state_rows[:, 4:8]))),
            "wrists": wrists,
        }
        if phase["id"] == "walk":
            displacement = state_rows[-1, 1:3] - state_rows[0, 1:3]
            walk_metrics = {
                "forward_displacement_m": float(displacement[0]),
                "lateral_displacement_m": float(abs(displacement[1])),
            }
            entry["walk"] = walk_metrics
        phase_metrics.append(entry)
    if walk_metrics is None:
        raise ValueError("protocol must contain a walk phase")
    replay_hashes: dict[str, str] | None = None
    replay_verified = False
    if replay_state_trace_path is not None or replay_motion_trace_path is not None:
        if replay_state_trace_path is None or replay_motion_trace_path is None:
            raise ValueError("state and motion replay traces must be supplied together")
        replay_hashes = {
            "state_trace_sha256": _sha256(replay_state_trace_path),
            "motion_trace_sha256": _sha256(replay_motion_trace_path),
        }
        replay_verified = replay_hashes == {
            "state_trace_sha256": _sha256(state_trace_path),
            "motion_trace_sha256": _sha256(motion_trace_path),
        }
    native_failed_steps = int(native.get("failed_environment_steps", -1))
    native_terminations = int(native.get("termination_count", -1))
    stability_passed = (
        native_failed_steps == gates["failed_environment_steps"]
        and native_terminations == gates["termination_count"]
        and float(np.min(state[:, 3])) >= gates["minimum_root_height_m"]
        and float(np.max(tilt)) < gates["maximum_tilt_rad"]
    )
    walk_passed = (
        walk_metrics["forward_displacement_m"] is not None
        and walk_metrics["lateral_displacement_m"] is not None
        and walk_metrics["forward_displacement_m"] >= gates["minimum_walk_forward_displacement_m"]
        and walk_metrics["lateral_displacement_m"] <= gates["maximum_walk_lateral_displacement_m"]
    )
    decision = trace_coverage and stability_passed and wrist_gate_passed and walk_passed
    if gates["required_exact_replay"]:
        decision = decision and replay_verified
    return {
        "schema": "numi.pqi3-trajectory-admission.v1",
        "protocol_sha256": _sha256(protocol_path),
        "interaction_pack": str(interaction_pack),
        "interaction_pack_sha256": _sha256(interaction_pack),
        "native_evidence": str(native_evidence_path),
        "native_evidence_sha256": _sha256(native_evidence_path),
        "state_trace": str(state_trace_path),
        "state_trace_sha256": _sha256(state_trace_path),
        "motion_feature_trace": str(motion_trace_path),
        "motion_feature_trace_sha256": _sha256(motion_trace_path),
        "rollout_pack": str(rollout_pack_path),
        "rollout_pack_sha256": _sha256(rollout_pack_path),
        "trace_coverage": trace_coverage,
        "native_failed_environment_steps": native_failed_steps,
        "native_termination_count": native_terminations,
        "minimum_root_height_m": float(np.min(state[:, 3])),
        "maximum_tilt_rad": float(np.max(tilt)),
        "phase_metrics": phase_metrics,
        "walk": walk_metrics,
        "stability_passed": stability_passed,
        "wrist_gates_passed": wrist_gate_passed,
        "walk_passed": walk_passed,
        "replay": {"required": gates["required_exact_replay"], "verified": replay_verified, "hashes": replay_hashes},
        "trajectory_bank_admitted": bool(decision),
        "decision": "admit" if decision else "reject",
        "claim_boundary": "simulator trajectory evidence only; not hardware evidence and not authorization for a manuscript claim",
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interaction-pack", type=Path, required=True)
    parser.add_argument("--native-evidence", type=Path, required=True)
    parser.add_argument("--state-trace", type=Path, required=True)
    parser.add_argument("--motion-trace", type=Path, required=True)
    parser.add_argument("--rollout-pack", type=Path, required=True)
    parser.add_argument("--replay-state-trace", type=Path)
    parser.add_argument("--replay-motion-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    result = analyze(
        options.protocol,
        interaction_pack=options.interaction_pack,
        native_evidence_path=options.native_evidence,
        state_trace_path=options.state_trace,
        motion_trace_path=options.motion_trace,
        rollout_pack_path=options.rollout_pack,
        replay_state_trace_path=options.replay_state_trace,
        replay_motion_trace_path=options.replay_motion_trace,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
