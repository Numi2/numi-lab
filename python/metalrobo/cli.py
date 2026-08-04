"""Deployment and independent-engine verification tools.

Simulation, rollout scheduling, and PPO orchestration are intentionally absent
from this Python CLI. Use ``metalrobo_task_rollout`` and
``metalrobo_task_train`` for the native Swift/Metal execution path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .g1_policy import (
    UnitreeG1MuJoCoRunner,
    export_g1_coreml,
    export_g1_mlx,
    export_g1_onnx,
)
from .unitree_mjlab_policy import import_unitree_g1_mjlab_policy


def _g1_parser(parser: argparse.ArgumentParser) -> None:
    operations = parser.add_subparsers(
        dest="g1_operation",
        required=True,
    )
    export = operations.add_parser(
        "export",
        help="export a bundled G1 PolicyPack for deployment",
    )
    export.add_argument("--policy-pack", type=Path, required=True)
    export.add_argument(
        "--library",
        type=Path,
        help="native library containing the pinned G1 deployment contract",
    )
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument(
        "--formats",
        nargs="+",
        choices=("mlx", "onnx", "coreml"),
        default=("mlx", "onnx", "coreml"),
    )

    sim2sim = operations.add_parser(
        "sim2sim",
        help="run a PolicyPack in Unitree's pinned official MuJoCo model",
    )
    sim2sim.add_argument("--policy-pack", type=Path, required=True)
    sim2sim.add_argument(
        "--library",
        type=Path,
        help="native library containing the pinned G1 deployment contract",
    )
    sim2sim.add_argument(
        "--official-model",
        type=Path,
        required=True,
        help="pinned unitree_mujoco g1/scene_29dof.xml",
    )
    sim2sim.add_argument(
        "--velocity-command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "YAW"),
    )
    sim2sim.add_argument("--seconds", type=float, default=20.0)
    sim2sim.add_argument(
        "--zero-action",
        action="store_true",
        help="run the default-pose controller baseline instead of the actor",
    )

    import_unitree = operations.add_parser(
        "import-unitree",
        help="convert Unitree's pinned official G1 actor to PolicyPack",
    )
    import_unitree.add_argument(
        "--official-repo",
        type=Path,
        required=True,
        help="clean pinned unitreerobotics/unitree_rl_mjlab checkout",
    )
    import_unitree.add_argument(
        "--output-policy-pack",
        type=Path,
        required=True,
    )
    import_unitree.add_argument(
        "--library",
        type=Path,
        help="native MetalRobo library used to write the PolicyPack",
    )
    for name in (
        "world-fingerprint",
        "task-fingerprint",
        "observation-fingerprint",
        "action-fingerprint",
    ):
        import_unitree.add_argument(
            f"--{name}",
            type=lambda value: int(value, 0),
            required=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metalrobo",
        description=(
            "MetalRobo deployment artifacts and independent sim2sim checks"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _g1_parser(
        commands.add_parser(
            "g1",
            help="bundled Unitree G1 deployment utilities",
        )
    )
    return parser


def _export_g1(args: argparse.Namespace) -> int:
    output = args.output_dir.expanduser().resolve()
    results: dict[str, str] = {}
    if "mlx" in args.formats:
        results["mlx"] = str(
            export_g1_mlx(
                args.policy_pack,
                output,
                library_path=args.library,
            )
        )
    if "onnx" in args.formats:
        results["onnx"] = str(
            export_g1_onnx(
                args.policy_pack,
                output / "g1-locomotion.onnx",
                library_path=args.library,
            )
        )
    if "coreml" in args.formats:
        results["coreml"] = str(
            export_g1_coreml(
                args.policy_pack,
                output / "G1Locomotion.mlpackage",
                library_path=args.library,
            )
        )
    print(
        json.dumps(
            {
                "operation": "g1-export",
                "artifacts": results,
            },
            indent=2,
        )
    )
    return 0


def _sim2sim_g1(args: argparse.Namespace) -> int:
    runner = UnitreeG1MuJoCoRunner(
        args.policy_pack,
        args.official_model,
        library_path=args.library,
    )
    report = runner.run(
        np.asarray(
            args.velocity_command or (0.5, 0.0, 0.0),
            dtype=np.float32,
        ),
        seconds=args.seconds,
        zero_action=args.zero_action,
    )
    print(
        json.dumps(
            {
                "operation": "g1-sim2sim",
                "controller": (
                    "zero_action_baseline"
                    if args.zero_action
                    else "policy"
                ),
                **report,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def _import_unitree_g1(args: argparse.Namespace) -> int:
    report = import_unitree_g1_mjlab_policy(
        args.official_repo,
        args.output_policy_pack,
        world_fingerprint=args.world_fingerprint,
        task_fingerprint=args.task_fingerprint,
        observation_fingerprint=args.observation_fingerprint,
        action_fingerprint=args.action_fingerprint,
        library_path=args.library,
    )
    print(
        json.dumps(
            {
                "operation": "g1-import-unitree",
                **report,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "g1" and args.g1_operation == "export":
            return _export_g1(args)
        if args.command == "g1" and args.g1_operation == "sim2sim":
            return _sim2sim_g1(args)
        if args.command == "g1" and args.g1_operation == "import-unitree":
            return _import_unitree_g1(args)
        raise AssertionError("unhandled command")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"metalrobo: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
