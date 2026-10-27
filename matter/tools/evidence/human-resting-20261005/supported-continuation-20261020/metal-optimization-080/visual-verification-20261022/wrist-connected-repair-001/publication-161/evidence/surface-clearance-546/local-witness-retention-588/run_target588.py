#!/usr/bin/env python3
"""Run frozen 582 checks with only its witness-damping scope changed to components."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

from connected_witness_damping588 import (
    CORE582, CORE582_SHA256, install_damper588, sha256_file, validate_zero_boundary588,
)


def _load_core():
    actual = sha256_file(CORE582)
    if actual != CORE582_SHA256:
        raise ValueError(f"frozen 582 core hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("retain_local_witnesses582_for588", CORE582)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned 582 core: {CORE582}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_points_faces(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return archive["points_world_m"].copy(), archive["source_O_faces"].copy()


def _annotate_report(out: Path, metadata: dict, core_path: Path):
    path = out / "report.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text())
    pins = dict(report.get("source_pins", {}))
    for source in (core_path, Path(__file__), Path(__file__).with_name("connected_witness_damping588.py")):
        pins[str(source.resolve())] = sha256_file(source)
    report["source_pins"] = pins
    report["witness_damping_588"] = {
        "method": metadata["damping_method"],
        "mutable_component_count": metadata["mutable_component_count"],
        "mutable_component_sizes": metadata["mutable_component_sizes"],
        "mutable_node_count": int(sum(metadata["mutable_component_sizes"])),
        "boundary_nodes_bitwise_zero_displacement": True,
        "boundary_nodes_checked": int(metadata["boundary_nodes_checked"]),
        "events": metadata["events"],
        "construction_only": True,
        "final_545_130_159_gates_unchanged": True,
    }
    report["native_admission"] = False
    report["accepted_physical_steps"] = 0
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args, _ = parser.parse_known_args()
    state = json.loads(args.state.read_text())
    base_path = Path(state["candidate"])
    if sha256_file(base_path) != state.get("candidate_sha256"):
        raise ValueError("baseline candidate does not match its pinned state")
    base_points, faces = _load_points_faces(base_path)
    target_points, target_faces = _load_points_faces(args.target)
    if target_points.dtype != np.float32 or base_points.dtype != np.float32 or target_points.shape != base_points.shape:
        raise ValueError("base and target must share a Float32 point domain")
    if target_faces.shape != faces.shape or not np.array_equal(target_faces, faces):
        raise ValueError("582 retention requires unchanged face topology")
    mutable = np.any(base_points != target_points, axis=1)
    boundary = validate_zero_boundary588(base_points, target_points, faces)
    core = _load_core()
    metadata = install_damper588(core, faces, mutable)
    metadata["boundary_nodes_checked"] = int(len(boundary))
    # Core computes its own dependency paths before this override. Its report
    # now pins the actual 588 entrypoint as the implementation used this run.
    core.__file__ = str(Path(__file__).resolve())
    try:
        core.main()
    finally:
        _annotate_report(args.out, metadata, CORE582)


if __name__ == "__main__":
    main()
