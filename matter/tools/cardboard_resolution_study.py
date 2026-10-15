#!/usr/bin/env python3
"""Plan and run a source-bound, sequential cardboard resolution study.

This driver never launches a run while planning.  At execution it admits every
run through cardboard_convergence._load_run and every pair through
cardboard_convergence.analyze_pair before calculating the additional sequence
gates in this module.  Native runs are retained even when they fail.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "numi.cardboard.resolution-study.v2"
PLAN_SCHEMA = "numi.cardboard.resolution-plan.v1"
THRESHOLDS = {
    "peak_force_relative": 0.02,
    "peak_moment_relative": 0.02,
    "path_work_relative": 0.02,
    "normalized_force_curve_rms": 0.05,
    "normalized_moment_curve_rms": 0.05,
    "signed_springback_angle_difference_deg": 1.0,
    "paper_damage_area_fraction_difference": 0.02,
    "debond_area_fraction_difference": 0.02,
}
PHASE_ORDER = ("loading", "hold", "unloading", "relaxation", "release")
PHASE_CLI = {
    "loading": "--steps",
    "hold": "--hold-steps",
    "unloading": "--unload-steps",
    "relaxation": "--relax-steps",
    "release": "--release-steps",
}
CONTROLLED_FLAGS = {
    "--output", "--metallib", "--nx", "--ny", "--thickness-slices", "--dt",
    "--steps", "--hold-steps", "--unload-steps", "--relax-steps", "--release-steps",
}


class StudyError(Exception):
    pass


def _load_convergence():
    path = Path(__file__).with_name("cardboard_convergence.py")
    spec = importlib.util.spec_from_file_location("cardboard_convergence", path)
    if not spec or not spec.loader:
        raise StudyError(f"cannot import convergence admission module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cc = _load_convergence()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _plan_digest(plan: dict[str, Any]) -> str:
    payload = dict(plan)
    payload.pop("plan_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _finite_positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StudyError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise StudyError(f"{label} must be a finite positive number")
    return number


def _baseline_contract(path: Path) -> Any:
    try:
        run = cc._load_run(path)
    except Exception as exc:
        raise StudyError(f"baseline is not admitted by cardboard_convergence._load_run: {exc}") from exc
    mesh = run.manifest["mesh"]
    dims = (mesh["nx_per_pitch"], mesh["ny"], mesh["thickness_slices"])
    if dims != (8, 2, 1):
        raise StudyError(f"study baseline must be the declared coarse mesh (8,2,1), found {dims}")
    protocol = cc._protocol(run.manifest["solver"])
    missing = [phase for phase in PHASE_ORDER if protocol[phase] <= 0]
    if missing:
        raise StudyError(f"baseline must include positive loading, hold, withdrawal, relaxation, and release steps; missing {missing}")
    if "release" not in run.phases:
        raise StudyError("baseline has no accepted free-release phase")
    return run


def _baseline_material_files(run: Any) -> list[dict[str, str]]:
    materials = run.manifest.get("materials", {})
    entries = []
    for path_key, digest_key in (("liner_path", "liner_sha256"),
                                 ("medium_path", "medium_sha256"),
                                 ("glue_path", "glue_sha256")):
        path_string = materials.get(path_key)
        expected = materials.get(digest_key)
        if not path_string or not expected:
            raise StudyError(f"baseline manifest lacks source path/digest for {path_key}")
        path = Path(path_string).expanduser().resolve()
        if not path.is_file():
            raise StudyError(f"material source is not a regular file: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise StudyError(f"material digest mismatch for {path}: expected {expected}, got {actual}")
        entries.append({"path": str(path), "sha256": actual, "role": path_key.removesuffix("_path")})
    return entries


def _default_source_paths(repo_root: Path) -> list[Path]:
    relative = (
        "matter/tools/cardboard_probe.mm",
        "matter/tools/cardboard_convergence.py",
        "matter/tools/cardboard_resolution_study.py",
        "matter/tools/cardboard_glue_mesh.hpp",
        "matter/tools/cardboard_starch_interface_mesh.hpp",
        "matter/src/runtime.mm",
        "matter/src/compiler_impl.cpp",
        "matter/src/validation_impl.cpp",
        "matter/src/metal/fem.metalinc",
        "matter/include/numi/matter/shared.h",
    )
    result = []
    for item in relative:
        path = repo_root / item
        if not path.is_file():
            raise StudyError(f"required source binding is missing: {path}")
        result.append(path.resolve())
    result.extend(path.resolve() for path in sorted((repo_root / "matter/src/metal").glob("*.metalinc")))
    result.extend(path.resolve() for path in sorted((repo_root / "matter/include/numi/matter").glob("*.h")))
    return result


def _hash_bindings(paths: Iterable[Path]) -> list[dict[str, str]]:
    entries = []
    seen: set[Path] = set()
    for original in paths:
        path = original.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise StudyError(f"source binding is not a regular file: {path}")
        entries.append({"path": str(path), "sha256": _sha256_file(path)})
    return sorted(entries, key=lambda item: item["path"])


def _native_base_arguments(run: Any) -> list[str]:
    manifest = run.manifest
    geometry = manifest["geometry"]
    solver = manifest["solver"]
    materials = manifest["materials"]
    tooling = manifest.get("tooling", {})
    if tooling.get("enabled") is not False:
        raise StudyError("this resolution study currently admits the end-grip coupon only; native score/contact mode is a separate study")
    if solver.get("runtime_execution") is not True:
        raise StudyError("baseline is not a runtime execution")

    def mm(key: str) -> str:
        return format(float(geometry[key]) * 1000.0, ".17g")

    args = [
        "--preset", str(manifest.get("preset", "literature2009")),
        "--length-mm", mm("length_m"), "--width-mm", mm("width_m"),
        "--height-mm", mm("total_height_m"), "--pitch-mm", mm("pitch_m"),
        "--liner-thickness-mm", mm("liner_thickness_m"),
        "--medium-thickness-mm", mm("medium_normal_thickness_m"),
        "--liner-material", str(Path(materials["liner_path"]).resolve()),
        "--medium-material", str(Path(materials["medium_path"]).resolve()),
        "--glue-material", str(Path(materials["glue_path"]).resolve()),
        "--glue-gap-mm", format(float(geometry["glue_minimum_gap_m"]) * 1000.0, ".17g"),
        "--bond-width-mm", format(float(geometry["bond_width_m"]) * 1000.0, ".17g"),
        "--upper-glue-gap-mm", format(float(geometry["upper_glue_gap_m"]) * 1000.0, ".17g"),
        "--upper-bond-width-mm", format(float(geometry["upper_bond_width_m"]) * 1000.0, ".17g"),
        "--bend-angle", format(float(solver["bend_angle_deg"]), ".17g"),
        "--newton-iterations", str(solver["newton_iteration_budget"]),
        "--material-iterations", str(solver["local_material_newton_iterations"]),
        "--fgmres-iterations", str(solver["fgmres_iteration_budget"]),
        "--fgmres-restart", str(solver["fgmres_restart"]),
        "--line-search-steps", str(solver["line_search_steps"]),
        "--residual-tolerance", format(float(solver["relative_residual_tolerance"]), ".17g"),
    ]
    preconditioner = solver.get("fgmres_preconditioner")
    preconditioner_flags = {
        "scalar_diagonal": "scalar",
        "regional_tangent_fem_diagonal": "regional-tangent-diagonal",
        "regional_galerkin_rap": "regional-galerkin",
        "regional_galerkin_block_jacobi": "regional-galerkin-block",
        "regional_galerkin_contact_rap": "regional-galerkin-contact",
        "regional_galerkin_graph_schwarz_conditioning_only": "regional-galerkin-schwarz",
    }
    if preconditioner not in preconditioner_flags:
        raise StudyError(f"unsupported or missing preconditioner in baseline: {preconditioner!r}")
    args += ["--fgmres-preconditioner", preconditioner_flags[preconditioner]]
    if not 1 <= int(solver["fgmres_restart"]) <= 16:
        raise StudyError("baseline restart exceeds the declared ceiling of16")
    for key, flag in (("fgmres_algorithmic_tangent_cache", "--cache-fem-algorithmic-tangent"),
                      ("paper_compensated_arithmetic", "--compensated-paper-arithmetic"),
                      ("safeguarded_paper_hill_initializer", "--safeguarded-paper-initializer")):
        value = solver.get(key, False)
        if type(value) is not bool:
            raise StudyError(f"baseline {key} must be boolean")
        if value:
            args.append(flag)
    if not solver.get("deformable_self_contact", False):
        args.append("--without-self-contact")
    if solver.get("deformation_gradient") == "persistent_reference_displacement":
        args.append("--reference-displacement-gradient")
    elif solver.get("deformation_gradient") not in (None, "absolute_position"):
        raise StudyError("baseline has an unsupported FEM kinematic formulation")
    projection = manifest.get("paper_projection_residual_scaling", {})
    if projection.get("enabled") is True:
        args.append("--scaled-paper-residuals")
    elif projection and projection.get("enabled") is not False:
        raise StudyError("baseline paper residual scaling flag is ambiguous")
    accurate = manifest.get("paper_hill_accurate_projection", {}).get("enabled", False)
    if type(accurate) is not bool:
        raise StudyError("baseline accurate projection flag is ambiguous")
    if accurate:
        if solver.get("paper_compensated_arithmetic") is not True:
            raise StudyError("accurate baseline lacks explicit compensated arithmetic policy")
        args.append("--accurate-paper-projection")
    if projection.get("enabled") is True and any(key not in solver for key in
            ("paper_compensated_arithmetic", "safeguarded_paper_hill_initializer")):
        raise StudyError("scaled baseline lacks explicit arithmetic/initializer policy")
    try:
        source_projection = cc._source_projection_policy(manifest)
    except cc.EvidenceError as error:
        raise StudyError(str(error)) from error
    if source_projection:
        if (accurate or projection.get("enabled") is True
                or solver.get("safeguarded_paper_hill_initializer", False)):
            raise StudyError("source15 projection cannot use 13-state Hill projection controls")
        args.append("--compensated-source-paper-projection")
        if manifest["compensated_source_paper_projection"].get("accurate_bilinear15_projection", False):
            args.append("--accurate-source-paper-projection")
    for feature in ("retained_matrix_damage_overlay", "corrugator_starch_interface",
                    "generic_compensated_implicit_projection"):
        if manifest.get(feature, {}).get("enabled", False):
            raise StudyError(f"end-grip study cannot reconstruct enabled {feature}; use its explicit stage study")
    return args


def _controlled_arg_check(args: list[str]) -> None:
    for arg in args:
        flag = arg.split("=", 1)[0]
        if flag in CONTROLLED_FLAGS:
            raise StudyError(f"common arguments cannot override planned axis {flag}")


def _spatial_cases() -> list[dict[str, Any]]:
    cases = []
    for nx in (16, 32):
        cases.append({"id": f"nx{nx}-ny2-t1", "group": "nx", "mesh": [nx, 2, 1], "axis": "nx"})
    for ny in (4, 8):
        cases.append({"id": f"nx8-ny{ny}-t1", "group": "ny", "mesh": [8, ny, 1], "axis": "ny"})
    for thickness in (2, 4):
        cases.append({"id": f"nx8-ny2-t{thickness}", "group": "thickness", "mesh": [8, 2, thickness], "axis": "thickness_slices"})
    cases.append({"id": "nx32-ny8-t4", "group": "combined_refined", "mesh": [32, 8, 4], "axis": "combined"})
    return cases


def _time_counts(protocol: dict[str, Any], factor: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for phase in PHASE_ORDER:
        value = protocol[phase]
        if value <= 0:
            raise StudyError(f"time refinement requires a positive {phase} phase")
        counts[phase] = value * factor
        if counts[phase] > 10000:
            raise StudyError(f"{phase} count {counts[phase]} exceeds the native limit of 10000")
    return counts


def _case_pairs(slower_load: bool) -> list[dict[str, Any]]:
    pairs = []
    for group, ids in (
        ("nx", ["coarse-reference", "nx16-ny2-t1", "nx32-ny2-t1"]),
        ("ny", ["coarse-reference", "nx8-ny4-t1", "nx8-ny8-t1"]),
        ("thickness", ["coarse-reference", "nx8-ny2-t2", "nx8-ny2-t4"]),
    ):
        pairs.append({"group": group, "kind": "spatial", "coarse_middle": [ids[0], ids[1]], "middle_fine": [ids[1], ids[2]], "triple": ids})
    pairs.append({"group": "combined_refined", "kind": "spatial", "coarse_middle": ["coarse-reference", "nx32-ny8-t4"], "middle_fine": None, "triple": None})
    pairs.append({"group": "temporal", "kind": "temporal", "coarse_middle": ["nx32-ny8-t4", "time-half"], "middle_fine": ["time-half", "time-quarter"], "triple": ["nx32-ny8-t4", "time-half", "time-quarter"]})
    if slower_load:
        pairs.append({"group": "slower_load", "kind": "load_rate", "coarse_middle": ["nx32-ny8-t4", "load-slower"], "middle_fine": None, "triple": None})
    return pairs


def make_plan(*, baseline_dir: Path, probe: Path, metallib: Path, run_root: Path,
              timeout_seconds: float, slower_load_factor: int | None = None,
              max_free_force_residual_n: float | None = None,
              source_paths: Iterable[Path] | None = None) -> dict[str, Any]:
    """Create the immutable prediction and sequential native case plan."""
    baseline_dir = baseline_dir.expanduser().resolve()
    probe = probe.expanduser().resolve()
    metallib = metallib.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    timeout = _finite_positive(timeout_seconds, "timeout_seconds")
    force_limit = (None if max_free_force_residual_n is None else
                   _finite_positive(max_free_force_residual_n, "max_free_force_residual_n"))
    run = _baseline_contract(baseline_dir)
    for path, label in ((probe, "probe executable"), (metallib, "Metal library")):
        if not path.is_file():
            raise StudyError(f"{label} is not a regular file: {path}")
    if not os.access(probe, os.X_OK):
        raise StudyError(f"probe executable is not executable: {probe}")
    if run_root.exists():
        raise StudyError(f"run root already exists; evidence is never overwritten: {run_root}")
    if slower_load_factor is not None and slower_load_factor < 2:
        raise StudyError("slower_load_factor must be an integer >= 2")

    repo_root = Path(__file__).resolve().parents[2]
    sources = _default_source_paths(repo_root)
    if source_paths is not None:
        sources.extend(source_paths)
    sources.extend((Path(__file__).resolve(), Path(cc.__file__).resolve()))
    source_bindings = _hash_bindings(sources)
    material_bindings = _baseline_material_files(run)
    binary_binding = {"path": str(probe), "sha256": _sha256_file(probe)}
    metallib_binding = {"path": str(metallib), "sha256": _sha256_file(metallib)}
    source_bindings = _hash_bindings([Path(item["path"]) for item in source_bindings] +
                                    [Path(item["path"]) for item in material_bindings])

    protocol = cc._protocol(run.manifest["solver"])
    common_args = _native_base_arguments(run)
    cases = []
    for item in _spatial_cases():
        cases.append({**item, "kind": "spatial", "dt_s": protocol["dt_s"],
                      "relative_residual_tolerance": float(run.manifest["solver"]["relative_residual_tolerance"]),
                      "phase_steps": {p: protocol[p] for p in PHASE_ORDER}})
    # The combined-refined spatial case is the dt reference for a nested
    # temporal triplet.  Counts are multiplied uniformly so every physical
    # phase duration and the imposed loading rate stay fixed.
    for factor, label in ((2, "half"), (4, "quarter")):
        counts = _time_counts(protocol, factor)
        cases.append({"id": f"time-{label}", "group": "temporal", "kind": "temporal",
                      "mesh": [32, 8, 4], "dt_s": protocol["dt_s"] / factor,
                      "relative_residual_tolerance": float(run.manifest["solver"]["relative_residual_tolerance"]) / factor,
                      "phase_steps": counts, "time_refinement_factor": factor})
    if slower_load_factor is not None:
        counts = {p: protocol[p] for p in PHASE_ORDER}
        counts["loading"] *= slower_load_factor
        if counts["loading"] > 10000:
            raise StudyError("slower-load case exceeds the native loading-step limit of 10000")
        cases.append({"id": "load-slower", "group": "slower_load", "kind": "load_rate",
                      "mesh": [32, 8, 4], "dt_s": protocol["dt_s"],
                      "relative_residual_tolerance": float(run.manifest["solver"]["relative_residual_tolerance"]),
                      "phase_steps": counts, "loading_duration_factor": slower_load_factor})

    capability = _capability_screen(run.manifest, run.result)
    plan = {
        "schema": PLAN_SCHEMA,
        "created_unix_utc": int(time.time()),
        "prediction": {
            "ordered_cases": [case["id"] for case in cases],
            "expected_direction": "for each three-level sequence, the middle-to-fine discrepancy should not exceed coarse-to-middle for every required numerical observable",
            "acceptance_thresholds": THRESHOLDS,
            "no_plateau_claim": "a missing or failed level is inconclusive; no extrapolated plateau or tolerance-derived absolute force accuracy is inferred",
            "spatial_axes": {"nx": [8, 16, 32], "ny": [2, 4, 8], "thickness_slices": [1, 2, 4], "combined_refined": [32, 8, 4]},
            "temporal_axis": "dt, dt/2, dt/4 at 32x8x4 with all five physical phase durations and loading speed unchanged",
            "force_accuracy": "hold relative residual tolerance / dt fixed by tightening tolerance on time refinement; report accepted native free-force residuals in newtons separately; this ratio is not an absolute force error bound because the residual normalization scale is not exported",
            "max_free_force_residual_n": force_limit,
        },
        "baseline": {
            "path": str(baseline_dir),
            "input_sha256": dict(sorted(run.files.items())),
            "mesh": [8, 2, 1],
            "protocol": {**protocol, "phase_durations_s": cc._phase_durations(run)},
            "compiled_world_fingerprint": run.result["compiled_world_fingerprint"],
            "materials": material_bindings,
            "scope_capabilities": capability,
        },
        "execution": {
            "probe": binary_binding,
            "metallib": metallib_binding,
            "source_files": source_bindings,
            "timeout_seconds": timeout,
            "run_root": str(run_root),
            "parallelism": 1,
            "lock_path": "/tmp/numi-cardboard-resolution-study.gpu.lock",
            "common_native_args": common_args,
            "controlled_flags": sorted(CONTROLLED_FLAGS),
            "failure_policy": "stop at first failed case; preserve output, stdout, stderr, prediction, and prior accepted runs",
        },
        "cases": cases,
        "comparisons": _case_pairs(slower_load_factor is not None),
        "stage_scope": {
            "required_capabilities_for_added_stage_resolution_claim": ["crease_tool_contact", "paper_damage", "glue_debond"],
            "capability_status": "complete" if all(capability.values()) else "incomplete_or_unreported",
            "rule": "pairwise numerical screens cannot qualify a crease/contact/damage/debond stage absent explicit native feature certificates and requested area observables",
        },
    }
    plan["plan_sha256"] = _plan_digest(plan)
    return plan


def _capability_screen(manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, bool]:
    """Require explicit native evidence flags; never infer from a material label."""
    features = result.get("resolution_features", {})
    return {
        "crease_tool_contact": features.get("crease_tool_contact_certified") is True,
        "paper_damage": features.get("paper_damage_area_certified") is True,
        "glue_debond": features.get("glue_debond_area_certified") is True,
    }


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def create_plan(args: argparse.Namespace) -> Path:
    root = Path(args.run_root).expanduser().resolve()
    plan = make_plan(
        baseline_dir=Path(args.baseline), probe=Path(args.probe), metallib=Path(args.metallib),
        run_root=root, timeout_seconds=args.timeout,
        slower_load_factor=args.slower_load_factor,
        max_free_force_residual_n=args.max_free_force_residual_n,
        source_paths=[Path(path) for path in args.source_file] if args.source_file else None,
    )
    root.mkdir(parents=True, exist_ok=False)
    plan_path = root / "study-plan.json"
    _write_json_new(plan_path, plan)
    return plan_path


def _verify_bindings(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("plan_sha256") != _plan_digest(plan):
        raise StudyError("study plan schema/digest is invalid or the prediction was modified")
    execution = plan["execution"]
    bindings = [execution["probe"], execution["metallib"], *execution["source_files"]]
    bindings.extend(plan["baseline"]["materials"])
    for entry in bindings:
        path = Path(entry["path"])
        if not path.is_file() or _sha256_file(path) != entry["sha256"]:
            raise StudyError(f"source binding changed or disappeared: {path}")
    baseline = Path(plan["baseline"]["path"])
    admitted = cc._load_run(baseline)
    if dict(sorted(admitted.files.items())) != plan["baseline"]["input_sha256"]:
        raise StudyError("baseline evidence changed after the prediction was sealed")


def _native_args(case: dict[str, Any], common: list[str], output: Path,
                 metallib: Path) -> list[str]:
    args = list(common)
    if "relative_residual_tolerance" in case:
        index = args.index("--residual-tolerance")
        args[index + 1] = format(float(case["relative_residual_tolerance"]), ".17g")
    args.extend(["--nx", str(case["mesh"][0]), "--ny", str(case["mesh"][1]),
                 "--thickness-slices", str(case["mesh"][2]),
                 "--dt", format(float(case["dt_s"]), ".17g")])
    for phase in PHASE_ORDER:
        args.extend([PHASE_CLI[phase], str(case["phase_steps"][phase])])
    args.extend(["--metallib", str(metallib), "--output", str(output)])
    return args


def _all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _absolute_linear_path_work(moment_a: float, moment_b: float, delta_angle_rad: float) -> float:
    """Integrate |M| exactly for a linearly interpolated moment over one angle interval."""
    a, b, width = float(moment_a), float(moment_b), abs(float(delta_angle_rad))
    if not _all_finite((a, b, width)):
        raise StudyError("nonfinite moment/rotation input to absolute-work integration")
    abs_a, abs_b = abs(a), abs(b)
    scale = max(abs_a, abs_b)
    if scale == 0.0 or width == 0.0:
        return 0.0
    scaled_a, scaled_b = abs_a / scale, abs_b / scale
    if (a < 0.0 < b) or (b < 0.0 < a):
        # Split at the exact zero crossing. The two normalized triangle areas
        # avoid squaring large finite moments or overflowing their sum.
        average_factor = 0.5 * (scaled_a * scaled_a + scaled_b * scaled_b) / (scaled_a + scaled_b)
    else:
        average_factor = 0.5 * (scaled_a + scaled_b)
    value = (scale * average_factor) * width
    if not math.isfinite(value):
        raise StudyError("absolute moment-path work overflowed")
    return value


def _run_metrics(run: Any) -> dict[str, Any]:
    protocol = cc._protocol(run.manifest["solver"])
    rows = [run.row_by_key[(step, 0)] for step in range(sum(protocol[p] for p in PHASE_ORDER))]
    forces = [math.sqrt(row["reaction_x_N"] ** 2 + row["reaction_y_N"] ** 2 + row["reaction_z_N"] ** 2) for row in rows]
    moments = [abs(row["reaction_moment_y_Nm"]) for row in rows]
    if not _all_finite(forces + moments):
        raise StudyError("nonfinite accepted force/moment observations")
    path_work_segments = []
    signed_work = 0.0
    for a, b in zip(rows, rows[1:]):
        theta_a = a.get("measured_right_grip_angle_deg")
        theta_b = b.get("measured_right_grip_angle_deg")
        if theta_a is None or theta_b is None:
            raise StudyError("accepted observations lack measured grip angles required for force-work integration")
        dtheta = math.radians(theta_b - theta_a)
        moment_a = a["reaction_moment_y_Nm"]
        moment_b = b["reaction_moment_y_Nm"]
        increment = 0.5 * (moment_a + moment_b) * dtheta
        signed_work += increment
        path_work_segments.append(_absolute_linear_path_work(moment_a, moment_b, dtheta))
    try:
        path_work = math.fsum(path_work_segments)
    except OverflowError as error:
        raise StudyError("absolute moment-path work overflowed during accumulation") from error
    if not _all_finite((signed_work, path_work)):
        raise StudyError("nonfinite integrated moment work")
    release = cc._release_response(run)
    if not release or release.get("boundary_verified") is not True:
        raise StudyError("free-release boundary response is missing or unverified")
    residual_rows = list(run.rows)
    l2_values = [row.get("free_force_imbalance_l2_N") for row in residual_rows]
    max_values = [row.get("free_force_imbalance_max_N") for row in residual_rows]
    residual_present = all(value is not None for value in l2_values + max_values)
    residual_summary = None
    if residual_present:
        numeric = [float(v) for v in l2_values + max_values]
        if not _all_finite(numeric):
            raise StudyError("native free-force residual observations are nonfinite")
        residual_summary = {
            "max_l2_N": max(float(v) for v in l2_values),
            "max_component_N": max(float(v) for v in max_values),
            "sample_count_across_environments": len(residual_rows),
            "interpretation": "observed accepted free-force imbalance; not an absolute error bound because the residual normalization scale is not exported",
        }
    return {
        "peak_resultant_reaction_force_N": max(forces),
        "peak_absolute_reaction_moment_Nm": max(moments),
        "signed_external_moment_work_J": signed_work,
        "absolute_moment_path_work_J": path_work,
        "springback_angle_change_deg": abs(float(release["right_grip_angle_change_deg"])) if release["right_grip_angle_change_deg"] is not None else None,
        "signed_springback_angle_change_deg": float(release["right_grip_angle_change_deg"]) if release["right_grip_angle_change_deg"] is not None else None,
        "free_force_residuals": residual_summary,
    }


def _relative_difference(a: float, b: float) -> float | None:
    denom = max(abs(a), abs(b))
    if denom <= 1.0e-30:
        return 0.0 if a == b else None
    return abs(b - a) / denom


def _paired_full_curve(run_a: Any, run_b: Any, kind: str) -> dict[str, Any]:
    matched = []
    if kind == "load_rate":
        for phase in PHASE_ORDER:
            points_a = [(cc._phase_coordinate(run_a, row, phase), row) for row in run_a.rows
                        if row["environment"] == 0 and row["phase"] == phase]
            points_b = [(cc._phase_coordinate(run_b, row, phase), row) for row in run_b.rows
                        if row["environment"] == 0 and row["phase"] == phase]
            if not points_a or not points_b:
                raise StudyError(f"load-rate pair lacks accepted {phase} samples")
            lower = max(min(x for x, _ in points_a), min(x for x, _ in points_b))
            upper = min(max(x for x, _ in points_a), max(x for x, _ in points_b))
            if lower > upper:
                raise StudyError(f"load-rate pair has no common normalized {phase} interval")
            coordinates = sorted(x for x in (set(x for x, _ in points_a) | set(x for x, _ in points_b))
                                 if lower <= x <= upper)
            for x in coordinates:
                a = {field: cc._interpolate(points_a, x, field) for field in
                     ("reaction_x_N", "reaction_y_N", "reaction_z_N", "reaction_moment_y_Nm")}
                b = {field: cc._interpolate(points_b, x, field) for field in
                     ("reaction_x_N", "reaction_y_N", "reaction_z_N", "reaction_moment_y_Nm")}
                matched.append((a, b))
    else:
        rows_a = [run_a.row_by_key[(step, 0)] for step in sorted({key[0] for key in run_a.row_by_key if key[1] == 0})]
        rows_b = [run_b.row_by_key[(step, 0)] for step in sorted({key[0] for key in run_b.row_by_key if key[1] == 0})]
        b_by_time = {cc._time_key(row["time_s"]): row for row in rows_b}
        for row_a in rows_a:
            row_b = b_by_time.get(cc._time_key(row_a["time_s"]))
            if row_b is None:
                # Spatial/temporal pair admission should already establish exact
                # common samples. Keep the gate independently fail-closed.
                raise StudyError(f"pair lacks a matching physical time {row_a['time_s']:.12g}s")
            matched.append((row_a, row_b))
    f_delta2 = 0.0
    f_base2 = 0.0
    m_delta2 = 0.0
    m_base2 = 0.0
    for a, b in matched:
        fa = (a["reaction_x_N"], a["reaction_y_N"], a["reaction_z_N"])
        fb = (b["reaction_x_N"], b["reaction_y_N"], b["reaction_z_N"])
        f_delta2 += sum((fb[i] - fa[i]) ** 2 for i in range(3))
        f_base2 += sum(value ** 2 for value in fa)
        m_delta2 += (b["reaction_moment_y_Nm"] - a["reaction_moment_y_Nm"]) ** 2
        m_base2 += a["reaction_moment_y_Nm"] ** 2
    force_norm = math.sqrt(f_delta2 / max(len(matched), 1))
    moment_norm = math.sqrt(m_delta2 / max(len(matched), 1))
    force_scale = math.sqrt(f_base2 / max(len(matched), 1))
    moment_scale = math.sqrt(m_base2 / max(len(matched), 1))
    return {
        "matched_physical_time_samples": len(matched),
        "normalized_force_curve_rms": force_norm / force_scale if force_scale > 1.0e-30 else (0.0 if force_norm == 0 else None),
        "normalized_moment_curve_rms": moment_norm / moment_scale if moment_scale > 1.0e-30 else (0.0 if moment_norm == 0 else None),
        "normalization": ("RMS difference across all accepted environment-0 common phase-coordinate samples divided by baseline RMS; each phase is normalized independently"
                          if kind == "load_rate" else
                          "RMS difference across all common accepted environment-0 physical-time samples divided by baseline RMS"),
    }


def _damage_areas(run: Any) -> dict[str, Any]:
    features = run.result.get("resolution_features", {})
    paper_enabled = features.get("paper_damage_area_enabled") is True
    debond_enabled = features.get("glue_debond_area_enabled") is True
    summary = run.result.get("damage_and_debond_area_fractions")
    if not paper_enabled and not debond_enabled:
        return {"enabled": False, "paper_enabled": False, "debond_enabled": False,
                "paper_damage_area_fraction": None, "debond_area_fraction": None}
    if not isinstance(summary, dict):
        return {"enabled": True, "paper_enabled": paper_enabled, "debond_enabled": debond_enabled,
                "missing": "damage_and_debond_area_fractions"}
    output = {"enabled": True, "paper_enabled": paper_enabled, "debond_enabled": debond_enabled}
    for key, enabled in (("paper_damage_area_fraction", paper_enabled), ("debond_area_fraction", debond_enabled)):
        if not enabled:
            output[key] = None
            continue
        value = summary.get(key)
        if value is None or not math.isfinite(float(value)) or not (0.0 <= float(value) <= 1.0):
            output[key] = None
        else:
            output[key] = float(value)
    return output


def _pair_metrics(run_a: Any, run_b: Any, kind: str) -> dict[str, Any]:
    a, b = _run_metrics(run_a), _run_metrics(run_b)
    curve = _paired_full_curve(run_a, run_b, kind)
    metrics = {
        "peak_force_relative": _relative_difference(a["peak_resultant_reaction_force_N"], b["peak_resultant_reaction_force_N"]),
        "peak_moment_relative": _relative_difference(a["peak_absolute_reaction_moment_Nm"], b["peak_absolute_reaction_moment_Nm"]),
        "path_work_relative": _relative_difference(a["absolute_moment_path_work_J"], b["absolute_moment_path_work_J"]),
        "normalized_force_curve_rms": curve["normalized_force_curve_rms"],
        "normalized_moment_curve_rms": curve["normalized_moment_curve_rms"],
        "springback_angle_difference_deg": abs(a["springback_angle_change_deg"] - b["springback_angle_change_deg"])
            if a["springback_angle_change_deg"] is not None and b["springback_angle_change_deg"] is not None else None,
        "signed_springback_angle_difference_deg": abs(
            a["signed_springback_angle_change_deg"] - b["signed_springback_angle_change_deg"]
        ) if a["signed_springback_angle_change_deg"] is not None and b["signed_springback_angle_change_deg"] is not None else None,
        "free_force_residuals": {"baseline": a["free_force_residuals"], "candidate": b["free_force_residuals"]},
        "damage_debond": {"baseline": _damage_areas(run_a), "candidate": _damage_areas(run_b)},
        "raw_summary": {"baseline": a, "candidate": b},
    }
    da, db = metrics["damage_debond"]["baseline"], metrics["damage_debond"]["candidate"]
    metrics["area_feature_enablement_matches"] = (
        da.get("paper_enabled") == db.get("paper_enabled") and
        da.get("debond_enabled") == db.get("debond_enabled")
    )
    for feature, key in (("paper_damage_area_fraction_difference", "paper_damage_area_fraction"),
                         ("debond_area_fraction_difference", "debond_area_fraction")):
        metrics[feature] = abs(da.get(key) - db.get(key)) if da.get(key) is not None and db.get(key) is not None else None
    return metrics


def _gate_pair(metrics: dict[str, Any]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    gates["area_feature_enablement"] = {
        "status": "pass" if metrics.get("area_feature_enablement_matches") is True else "incompatible_feature_enablement"
    }
    for key, limit in THRESHOLDS.items():
        value = metrics.get(key)
        if key in ("paper_damage_area_fraction_difference", "debond_area_fraction_difference"):
            area_key = "paper_enabled" if key.startswith("paper_") else "debond_enabled"
            enabled = (metrics.get("damage_debond", {}).get("baseline", {}).get(area_key) or
                       metrics.get("damage_debond", {}).get("candidate", {}).get(area_key))
            if not enabled:
                gates[key] = {"status": "not_enabled", "observed": None, "limit": limit}
                continue
            if metrics.get("area_feature_enablement_matches") is not True:
                gates[key] = {"status": "incompatible_feature_enablement", "observed": value, "limit": limit}
                continue
        if value is None or not math.isfinite(float(value)):
            gates[key] = {"status": "inconclusive_missing_metric", "observed": value, "limit": limit}
        else:
            gates[key] = {"status": "pass" if float(value) <= limit else "exceeds_limit", "observed": float(value), "limit": limit}
    residuals = metrics.get("free_force_residuals", {})
    if not residuals.get("baseline") or not residuals.get("candidate"):
        gates["observed_free_force_residuals"] = {"status": "inconclusive_missing_observation", "absolute_accuracy_claim": False}
    else:
        bound = metrics.get("max_free_force_residual_limit_N")
        observed = max(
            residuals["baseline"]["max_l2_N"], residuals["baseline"]["max_component_N"],
            residuals["candidate"]["max_l2_N"], residuals["candidate"]["max_component_N"],
        )
        gates["observed_free_force_residuals"] = {
            "status": ("inconclusive_no_absolute_bound_declared" if bound is None else
                       ("pass" if observed <= bound else "exceeds_declared_absolute_bound")),
            "baseline_max_l2_N": residuals["baseline"]["max_l2_N"],
            "baseline_max_component_N": residuals["baseline"]["max_component_N"],
            "candidate_max_l2_N": residuals["candidate"]["max_l2_N"],
            "candidate_max_component_N": residuals["candidate"]["max_component_N"],
            "maximum_observed_N": observed,
            "declared_limit_N": bound,
            "residual_limit_satisfied": (None if bound is None else observed <= bound),
            "absolute_force_error_bound": False,
        }
    return gates


def _gate_triple(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    keys = list(THRESHOLDS)
    gates: dict[str, Any] = {}
    for key in keys:
        coarse = first.get(key)
        fine = second.get(key)
        if key in ("paper_damage_area_fraction_difference", "debond_area_fraction_difference"):
            area_key = "paper_enabled" if key.startswith("paper_") else "debond_enabled"
            enabled = any(
                pair.get("damage_debond", {}).get(endpoint, {}).get(area_key) is True
                for pair in (first, second) for endpoint in ("baseline", "candidate")
            )
            if not enabled:
                gates[key] = {"status": "not_enabled", "coarse_middle": None, "middle_fine": None}
                continue
            matching = all(
                pair.get("area_feature_enablement_matches") is True
                for pair in (first, second)
            )
            if not matching:
                gates[key] = {"status": "incompatible_feature_enablement", "coarse_middle": coarse, "middle_fine": fine}
                continue
        if coarse is None or fine is None:
            gates[key] = {"status": "inconclusive_missing_metric", "coarse_middle": coarse, "middle_fine": fine}
        else:
            gates[key] = {"status": "pass" if float(fine) < float(coarse) else "difference_not_decreasing",
                          "coarse_middle": float(coarse), "middle_fine": float(fine)}
    return gates


def _load_case_runs(plan: dict[str, Any], case_paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, str]]:
    runs: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    runs["coarse-reference"] = cc._load_run(Path(plan["baseline"]["path"]))
    statuses["coarse-reference"] = "admitted_by_convergence_loader"
    for case in plan["cases"]:
        path = case_paths.get(case["id"])
        if path is None:
            statuses[case["id"]] = "not_executed"
            continue
        try:
            runs[case["id"]] = cc._load_run(path)
            statuses[case["id"]] = "admitted_by_convergence_loader"
        except Exception as exc:
            statuses[case["id"]] = f"rejected_by_convergence_loader: {type(exc).__name__}: {exc}"
    return runs, statuses


def analyze_study(plan: dict[str, Any], case_paths: dict[str, Path]) -> dict[str, Any]:
    """Admit every pair and apply three-level plus finest-pair gates."""
    _verify_bindings(plan)
    runs, admissions = _load_case_runs(plan, case_paths)
    pair_results: dict[str, Any] = {}
    scope_by_run = {}
    baseline_run = cc._load_run(Path(plan["baseline"]["path"]))
    scope_by_run["coarse-reference"] = _capability_screen(baseline_run.manifest, baseline_run.result)
    for case_id, run in runs.items():
        if case_id != "coarse-reference":
            scope_by_run[case_id] = _capability_screen(run.manifest, run.result)
    required_capabilities = ("crease_tool_contact", "paper_damage", "glue_debond")
    scope_complete = len(scope_by_run) == len(plan["cases"]) + 1 and all(
        capabilities.get(key) is True
        for capabilities in scope_by_run.values() for key in required_capabilities
    )
    all_pairs = []
    for comparison in plan["comparisons"]:
        refs = [comparison["coarse_middle"], comparison.get("middle_fine")]
        for pair in refs:
            if not pair:
                continue
            all_pairs.append((comparison, pair))
    for comparison, pair in all_pairs:
        left, right = pair
        pair_name = f"{left}__{right}"
        if left not in runs or right not in runs:
            pair_results[pair_name] = {"status": "inconclusive_missing_run", "left": left, "right": right}
            continue
        kind = comparison["kind"]
        report = cc.analyze_pair(runs[left].path, runs[right].path, kind, list(PHASE_ORDER),
                                 force_matched_tolerance=(kind == "temporal"))
        if report.get("status") != "analyzed_descriptive":
            pair_results[pair_name] = {"status": report.get("status"), "admission": report}
            continue
        metrics = _pair_metrics(runs[left], runs[right], kind)
        metrics["max_free_force_residual_limit_N"] = plan["prediction"].get("max_free_force_residual_n")
        pair_results[pair_name] = {"status": "admitted_descriptive", "admission": report,
                                   "metrics": metrics, "gates": _gate_pair(metrics)}

    groups = {}
    for comparison in plan["comparisons"]:
        group = comparison["group"]
        coarse_pair = comparison["coarse_middle"]
        middle_pair = comparison.get("middle_fine")
        coarse_id = f"{coarse_pair[0]}__{coarse_pair[1]}"
        middle_id = f"{middle_pair[0]}__{middle_pair[1]}" if middle_pair else None
        coarse_result = pair_results.get(coarse_id, {})
        middle_result = pair_results.get(middle_id, {}) if middle_id else {}
        if not middle_pair:
            groups[group] = {"status": "inconclusive_single_pair_no_triple", "coarse_to_fine": coarse_result}
            continue
        if coarse_result.get("status") != "admitted_descriptive" or middle_result.get("status") != "admitted_descriptive":
            groups[group] = {"status": "inconclusive_pair_admission", "coarse_to_middle": coarse_result,
                             "middle_to_fine": middle_result}
            continue
        coarse_metrics = coarse_result["metrics"]
        middle_metrics = middle_result["metrics"]
        monotonic = _gate_triple(coarse_metrics, middle_metrics)
        finest = middle_result["gates"]
        group_pass = all(item["status"] in ("pass", "not_enabled") for item in monotonic.values()) and all(
            gate["status"] in ("pass", "not_enabled")
            for gate in finest.values()
        )
        residual_bound_statuses = [
            pair_result.get("gates", {}).get("observed_free_force_residuals", {}).get("status")
            for pair_result in (coarse_result, middle_result)
        ]
        group_pass = group_pass and all(status == "pass" for status in residual_bound_statuses)
        all_gate_statuses = [item["status"] for item in monotonic.values()]
        all_gate_statuses += [item["status"] for item in finest.values()]
        all_gate_statuses += residual_bound_statuses
        if any(str(status).startswith("inconclusive") for status in all_gate_statuses):
            group_status = "inconclusive_missing_gate_or_force_residual_limit"
        elif not group_pass:
            group_status = "numerical_screen_failed"
        elif not scope_complete:
            group_status = "numerical_screen_only_stage_scope_incomplete"
        else:
            group_status = "within_declared_numerical_screen"
        groups[group] = {
            "status": group_status,
            "decreasing_difference_gate": monotonic,
            "finest_pair_threshold_gate": finest,
            "coarse_to_middle": coarse_result,
            "middle_to_fine": middle_result,
            "physical_validation": False,
        }

    return {
        "schema": SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "source_bindings": plan["execution"]["source_files"],
        "binary_binding": plan["execution"]["probe"],
        "metallib_binding": plan["execution"]["metallib"],
        "case_admissions": admissions,
        "pair_comparisons": pair_results,
        "groups": groups,
        "scope_gate": {
            "status": "complete" if scope_complete else "inconclusive_missing_stage_certificates",
            "capabilities_by_run": scope_by_run,
            "meaning": "current end-grip coupon convergence cannot qualify the added scoring/contact/damage/debond stages without explicit accepted native certificates and area fields",
        },
        "force_accuracy_limit": "temporal pairs require matching relative residual tolerance / dt, with tighter tolerances on smaller timesteps; observed free-force imbalance is separately bounded in N for every pair. The tolerance ratio alone is not an absolute force-error bound because the residual normalization scale is not exported",
        "physical_validation": False,
    }


def _write_execution_prediction(case_dir: Path, plan: dict[str, Any], case: dict[str, Any]) -> None:
    prediction = {
        "schema": "numi.cardboard.resolution-case-prediction.v1",
        "plan_sha256": plan["plan_sha256"],
        "case": case,
        "expected_observation": "each accepted stage should be complete; no outcome is assumed",
        "expected_direction": "resolution differences should decrease on three-level sequences; otherwise retain and report the contrary result",
        "acceptance_thresholds": THRESHOLDS,
        "created_before_native_execution": True,
    }
    _write_json_new(case_dir / "prediction.json", prediction)


def run_plan(plan_path: Path) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise StudyError("native one-GPU execution requires the configured macOS Metal host")
    plan_path = plan_path.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _verify_bindings(plan)
    run_root = Path(plan["execution"]["run_root"])
    if (run_root / "cases").exists() or (run_root / "study-result.json").exists():
        raise StudyError("this plan already has execution artifacts; retries require a new plan and run root")
    lock_path = Path(plan["execution"]["lock_path"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StudyError(f"another cardboard GPU study owns {lock_path}; no cases launched") from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(json.dumps({"pid": os.getpid(), "plan_sha256": plan["plan_sha256"], "started_unix": time.time()}) + "\n")
        lock_file.flush()
        case_paths: dict[str, Path] = {}
        attempted_case_ids: list[str] = []
        failed_at: str | None = None
        for ordinal, case in enumerate(plan["cases"], start=1):
            try:
                _verify_bindings(plan)
            except Exception as exc:
                failed_at = f"source-binding-before-{case['id']}"
                _write_json_new(run_root / "stopped-before-case.json", {
                    "status": "source_binding_changed_before_case",
                    "next_case": case["id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "physical_validation": False,
                })
                break
            case_dir = run_root / "cases" / f"{ordinal:02d}-{case['id']}"
            case_dir.mkdir(parents=True, exist_ok=False)
            output_dir = case_dir / "native"
            _write_execution_prediction(case_dir, plan, case)
            args = [str(Path(plan["execution"]["probe"]["path"]))]
            args += _native_args(case, plan["execution"]["common_native_args"], output_dir,
                                 Path(plan["execution"]["metallib"]["path"]))
            _write_json_new(case_dir / "command.json", {"argv": args, "cwd": str(run_root), "plan_sha256": plan["plan_sha256"]})
            attempted_case_ids.append(case["id"])
            start = time.time()
            outcome: dict[str, Any]
            try:
                with (case_dir / "stdout.log").open("xb") as stdout, (case_dir / "stderr.log").open("xb") as stderr:
                    completed = subprocess.run(args, cwd=run_root, stdout=stdout, stderr=stderr,
                                               timeout=float(plan["execution"]["timeout_seconds"]), check=False)
                outcome = {"status": "process_exited", "returncode": completed.returncode}
                if output_dir.is_dir():
                    case_paths[case["id"]] = output_dir
                else:
                    outcome["status"] = "no_native_output_directory"
            except subprocess.TimeoutExpired as exc:
                outcome = {"status": "timeout", "timeout_seconds": plan["execution"]["timeout_seconds"],
                           "partial_stdout": str(exc.stdout) if exc.stdout else None,
                           "partial_stderr": str(exc.stderr) if exc.stderr else None}
                if output_dir.is_dir():
                    case_paths[case["id"]] = output_dir
            except Exception as exc:
                outcome = {"status": "launch_error", "error": f"{type(exc).__name__}: {exc}"}
                if output_dir.is_dir():
                    case_paths[case["id"]] = output_dir
            outcome["elapsed_seconds"] = time.time() - start
            outcome["source_integrity_after_run"] = "verified" if all(
                Path(item["path"]).is_file() and _sha256_file(Path(item["path"])) == item["sha256"]
                for item in [plan["execution"]["probe"], plan["execution"]["metallib"],
                             *plan["execution"]["source_files"], *plan["baseline"]["materials"]]
            ) else "changed"
            if outcome["status"] != "process_exited" or outcome.get("returncode") != 0 or outcome["source_integrity_after_run"] != "verified":
                failed_at = case["id"]
                _write_json_new(case_dir / "runner-result.json", outcome)
                break
            # The native process can return zero after exporting a failed
            # protocol. Admit immediately so later expensive cases do not run.
            admission_error = None
            try:
                cc._load_run(output_dir)
            except Exception as exc:
                admission_error = f"rejected: {type(exc).__name__}: {exc}"
            if admission_error:
                outcome["admission"] = admission_error
                failed_at = case["id"]
                _write_json_new(case_dir / "runner-result.json", outcome)
                break
            outcome["admission"] = "admitted_by_convergence_loader"
            _write_json_new(case_dir / "runner-result.json", outcome)
        analysis = None
        try:
            # Also retain pair-admission diagnostics for a failed/incomplete
            # case when all source bindings remain intact.
            _verify_bindings(plan)
            analysis = analyze_study(plan, case_paths)
            _write_json_new(run_root / "analysis.json", analysis)
        except Exception as exc:
            if failed_at is None:
                failed_at = "post-run-analysis"
            _write_json_new(run_root / "analysis-rejected.json", {
                "status": "inconclusive_analysis_binding_or_admission_error",
                "error": f"{type(exc).__name__}: {exc}",
                "physical_validation": False,
            })
        summary = {
            "schema": SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "status": "completed_all_cases" if failed_at is None else "stopped_preserving_failure",
            "failed_case": failed_at,
            "attempted_cases": attempted_case_ids,
            "unattempted_cases": [case["id"] for case in plan["cases"] if case["id"] not in attempted_case_ids],
            "analysis": "analysis.json" if analysis is not None else None,
            "physical_validation": False,
        }
        _write_json_new(run_root / "study-result.json", summary)
        return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="seal a source-bound sequential prediction without launching native work")
    plan.add_argument("--baseline", required=True, type=Path)
    plan.add_argument("--probe", required=True, type=Path)
    plan.add_argument("--metallib", required=True, type=Path)
    plan.add_argument("--run-root", required=True, type=Path)
    plan.add_argument("--timeout", required=True, type=float)
    plan.add_argument("--max-free-force-residual-n", type=float,
                      help="optional declared absolute limit for observed accepted free-force L2 and max residuals")
    plan.add_argument("--slower-load-factor", type=int,
                      help="optional separate load-rate comparison; multiplies loading duration only")
    plan.add_argument("--source-file", action="append", type=Path,
                      help="additional source binding; default core source set is always included")
    run = sub.add_parser("run", help="execute the sealed cases sequentially under the one-GPU process lock")
    run.add_argument("--plan", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "plan":
            path = create_plan(args)
            print(json.dumps({"status": "planned_only", "plan": str(path), "gpu_launched": False}))
            return 0
        result = run_plan(args.plan)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "completed_all_cases" else 2
    except (StudyError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "rejected", "error": f"{type(exc).__name__}: {exc}", "gpu_launched": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
