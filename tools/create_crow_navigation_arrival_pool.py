#!/usr/bin/env python3
"""Compile current autonomous Crow route arrivals into Metal reset data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _finite(values: list[Any], count: int, label: str) -> list[float]:
    if len(values) != count:
        raise ValueError(f"{label} must contain {count} values")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _float(value: float) -> str:
    if value == 0.0:
        return "0.0f"
    rendered = f"{value:.10g}"
    if "e" not in rendered and "." not in rendered:
        rendered += ".0"
    return rendered + "f"


def _scene_position(scene: list[float], index: int) -> tuple[float, float, float]:
    base = index * 13
    return scene[base], scene[base + 1], scene[base + 2]


def _waypoint(
    scene: list[float],
    index: int,
) -> tuple[float, float, float]:
    # Scene body zero is the ground; the five authored course bodies follow.
    if index == 1:
        x, y, z = _scene_position(scene, 3)
        return x, y + (-0.75 if y >= 0.0 else 0.75), z + 0.25
    if index == 2:
        x, y, z = _scene_position(scene, 4)
        return x, y + (-0.75 if y >= 0.0 else 0.75), z + 0.25
    if index == 3:
        x, y, z = _scene_position(scene, 5)
        return x - 0.50, y, z + 0.30
    if index == 4:
        x, y, z = _scene_position(scene, 5)
        return x, y + 0.55, z + 0.12
    raise ValueError("arrival-pool compilation supports route waypoints 1...4")


def _rotate_inverse(
    orientation: list[float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = orientation
    vx, vy, vz = vector
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y + z * w) * vy
        + 2.0 * (x * z - y * w) * vz,
        2.0 * (x * y - z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z + x * w) * vz,
        2.0 * (x * z + y * w) * vx
        + 2.0 * (y * z - x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def _smoothstep(lower: float, upper: float, value: float) -> float:
    unit = min(max((value - lower) / (upper - lower), 0.0), 1.0)
    return unit * unit * (3.0 - 2.0 * unit)


def _command_and_phase(
    q: list[float],
    scene: list[float],
    step: int,
    target_waypoint: int,
) -> tuple[float, float, float, float]:
    target = _waypoint(scene, target_waypoint)
    delta = tuple(target[index] - q[index] for index in range(3))
    local = _rotate_inverse(q[3:7], delta)
    planar = math.hypot(local[0], local[1])
    if planar > 1.0e-5:
        direction = local[0] / planar, local[1] / planar
    else:
        direction = 1.0, 0.0
    speed = 0.35 * _smoothstep(0.10, 0.60, planar)
    yaw = max(min(1.40 * math.atan2(local[1], local[0]), 0.45), -0.45)
    phase = math.remainder(step * 0.02 * 4.6 * 2.0 * math.pi, 2.0 * math.pi)
    return speed * direction[0], speed * direction[1], yaw, phase


def _array(name: str, rows: list[list[float]], width: int) -> str:
    lines = [f"constant float {name}[{len(rows)}][{width}] = {{"]
    for row in rows:
        lines.append("    {" + ", ".join(_float(value) for value in row) + "},")
    lines.append("};")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-metal", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--arrival-waypoint",
        type=int,
        choices=(3, 4),
        default=3,
    )
    arguments = parser.parse_args()
    arrival_waypoint = arguments.arrival_waypoint
    stage_name = "StageThree" if arrival_waypoint == 3 else "StageFour"

    rows: list[dict[str, Any]] = []
    contract: tuple[str, str, str, str] | None = None
    sources: list[dict[str, Any]] = []
    for input_path in arguments.inputs:
        raw = input_path.read_bytes()
        envelope = json.loads(raw)
        if envelope.get("schema") != "numi.crow-navigation-arrivals.v1":
            raise ValueError(f"{input_path} has the wrong schema")
        payload = envelope["payload"]
        if int(payload.get("arrival_waypoint", 3)) != arrival_waypoint:
            raise ValueError(
                f"{input_path} is not a waypoint-{arrival_waypoint} arrival pack"
            )
        if payload.get("scheduled_resets"):
            raise ValueError(f"{input_path} used scheduled resets")
        fingerprint = tuple(
            str(payload[key])
            for key in (
                "world_fingerprint",
                "task_fingerprint",
                "observation_fingerprint",
                "action_fingerprint",
            )
        )
        if contract is None:
            contract = fingerprint
        elif contract != fingerprint:
            raise ValueError("arrival packs disagree on their native contract")
        frames = payload["frames"]
        if len(frames) != int(payload["frame_count"]):
            raise ValueError(f"{input_path} frame count is inconsistent")
        source = {
            "path": str(input_path.resolve()),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "payload_sha256": envelope["payload_sha256"],
            "seed": str(payload["seed"]),
            "frame_count": len(frames),
        }
        sources.append(source)
        for frame in frames:
            q = _finite(frame["q"], 20, "q")
            v = _finite(frame["v"], 19, "v")
            actions = _finite(frame["accepted_actions"], 15, "accepted actions")
            scene = _finite(frame["scene_states"], 6 * 13, "scene states")
            _finite(
                frame["actor_observation"],
                int(payload["actor_observation_count"]),
                "actor observation",
            )
            if (
                float(frame["outcomes"]["navigation_waypoints_reached"])
                < float(arrival_waypoint)
            ):
                raise ValueError(
                    f"arrival frame did not reach waypoint {arrival_waypoint}"
                )
            step = int(frame["step"])
            start = _waypoint(scene, arrival_waypoint - 1)
            incoming = _waypoint(scene, arrival_waypoint - 2)
            rows.append({
                "source_seed": source["seed"],
                "environment": int(frame["environment"]),
                "step": step,
                "q": q,
                "v": v,
                "actions": actions,
                "root_offset": [q[index] - start[index] for index in range(3)],
                "incoming_yaw": math.atan2(
                    start[1] - incoming[1],
                    start[0] - incoming[0],
                ),
                "command_and_phase": list(
                    _command_and_phase(q, scene, step, arrival_waypoint)
                ),
                "journey_phase": min(max(step / 1600.0, 0.0), 1.0),
                "maximum_waypoints": float(frame["maximum_waypoints_reached"]),
            })

    rows.sort(key=lambda row: (int(row["source_seed"]), row["environment"]))
    if not rows:
        raise ValueError("arrival-pool compilation requires at least one frame")
    count = len(rows)
    comment = [
        f"// Current autonomous waypoint-{arrival_waypoint} arrivals captured",
        "// from the accepted parent. Rows include both successful next-waypoint",
        "// continuations and failure-near misses; source",
        "// hashes and per-row provenance are retained in the generated manifest.",
    ]
    steps = ", ".join(f"{row['step']}u" for row in rows)
    metal = "\n".join(comment + [
        f"constant uint kCrowNavigation{stage_name}ArrivalCount = {count}u;",
        f"constant uint kCrowNavigation{stage_name}Step[{count}] = {{{steps}}};",
        _array(f"kCrowNavigation{stage_name}Q", [row["q"] for row in rows], 20),
        _array(f"kCrowNavigation{stage_name}V", [row["v"] for row in rows], 19),
        f"constant float3 kCrowNavigation{stage_name}RootOffset[" + str(count) + "] = {\n" +
        "\n".join(
            "    float3(" + ", ".join(_float(value) for value in row["root_offset"]) + "),"
            for row in rows
        ) + "\n};",
        f"constant float kCrowNavigation{stage_name}CapturedIncomingYaw[" + str(count) + "] = {\n    " +
        ", ".join(_float(row["incoming_yaw"]) for row in rows) + "\n};",
        _array(f"kCrowNavigation{stage_name}Action", [row["actions"] for row in rows], 15),
        f"constant float4 kCrowNavigation{stage_name}CommandAndPhase[" + str(count) + "] = {\n" +
        "\n".join(
            "    float4(" + ", ".join(_float(value) for value in row["command_and_phase"]) + "),"
            for row in rows
        ) + "\n};",
        f"constant float kCrowNavigation{stage_name}JourneyPhase[" + str(count) + "] = {\n    " +
        ", ".join(_float(row["journey_phase"]) for row in rows) + "\n};",
        "",
    ])
    arguments.output_metal.write_text(metal)
    manifest = {
        "schema": "numi.crow-navigation-arrival-pool.v1",
        "classification": "simulated current-parent curriculum reset states",
        "arrival_waypoint": arrival_waypoint,
        "contract": {
            "world": contract[0],
            "task": contract[1],
            "observation": contract[2],
            "action": contract[3],
        },
        "sources": sources,
        "row_count": count,
        "successful_next_waypoint_continuations": sum(
            row["maximum_waypoints"] >= float(arrival_waypoint + 1)
            for row in rows
        ),
        "failed_arrival_waypoint_continuations": sum(
            row["maximum_waypoints"] < float(arrival_waypoint + 1)
            for row in rows
        ),
        "rows": [
            {
                "index": index,
                "source_seed": row["source_seed"],
                "environment": row["environment"],
                "step": row["step"],
                "maximum_waypoints": row["maximum_waypoints"],
            }
            for index, row in enumerate(rows)
        ],
    }
    arguments.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
