#!/usr/bin/env python3
"""Compare host-scaled operator-FD captures with frozen-031 native baseline.

This is a post-execution diagnostic, not a native solve, material oracle,
tangent-cause attribution, or convergence test. It intentionally verifies the
different source/build identities rather than claiming the input maps match.
Run only after the sealed 002 and 003 receipts exist.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
BASELINE = "solver/operator-fd-baseline-001"
SCALED_CASES = {
    "scale8": "solver/operator-fd-002",
    "scale16": "solver/operator-fd-003",
}
EXPECTED_CHANGED_INPUTS = {
    "build-cardboard-frozenX/numi-matter-cardboard-probe",
    "matter/src/runtime.mm",
    "matter/tools/cardboard_cohesive_face_check.mm",
}
EXPECTED_NEW_INPUTS = {"matter/tools/cardboard_fem_operator_fd.md"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_build_path(path: str) -> str:
    for name in ("build-cardboard-frozen031", "build-cardboard-frozen033"):
        path = path.replace(name + "/", "build-cardboard-frozenX/")
    return path


def case(root: Path, rel: str) -> dict[str, Any]:
    p = root / rel
    prediction_path = p / "prediction.json"
    receipt_path = p / "receipt.json"
    if not prediction_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"sealed case is incomplete; wait for receipt: {rel}")
    prediction = load(prediction_path)
    receipt = load(receipt_path)
    output = p / "output"
    required = [output / "observations.csv", output / "result.json", output / "compiled.nmatterpack"]
    missing = [str(q) for q in required if not q.is_file()]
    if missing:
        raise RuntimeError(f"case output is incomplete for {rel}: {missing}")

    hashes = prediction.get("inputs_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimeError(f"missing input hash map: {rel}")
    invalid = []
    for name, expected in hashes.items():
        source = p / "inputs" / name
        actual = digest(source) if source.is_file() else None
        if actual != expected:
            invalid.append({"path": name, "expected": expected, "actual": actual})
    return {
        "path": rel,
        "directory": p,
        "prediction": prediction,
        "receipt": receipt,
        "input_hashes": hashes,
        "input_verification": {
            "expected_count": len(hashes),
            "verified_count": len(hashes) - len(invalid),
            "all_verified": not invalid,
            "invalid": invalid,
        },
        "observations": output / "observations.csv",
        "result": output / "result.json",
        "compiled_pack": output / "compiled.nmatterpack",
        "stderr": p / "stderr.log",
    }


def normalized_inputs(entry: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, value in entry["input_hashes"].items():
        key = normalize_build_path(path)
        if key in result:
            raise RuntimeError(f"normalized input path collision in {entry['path']}: {key}")
        result[key] = value
    return result


def input_delta(baseline: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    a, b = normalized_inputs(baseline), normalized_inputs(diagnostic)
    common = a.keys() & b.keys()
    changed = [
        {"path": name, "baseline_sha256": a[name], "diagnostic_sha256": b[name]}
        for name in sorted(common) if a[name] != b[name]
    ]
    only_baseline = sorted(a.keys() - b.keys())
    only_diagnostic = sorted(b.keys() - a.keys())
    changed_paths = {item["path"] for item in changed}
    return {
        "input_maps_identical": a == b,
        "build_directory_names_normalized": True,
        "baseline_input_count": len(a),
        "diagnostic_input_count": len(b),
        "same_path_same_hash_count": sum(a[name] == b[name] for name in common),
        "changed_hashes": changed,
        "only_baseline_paths": only_baseline,
        "only_diagnostic_paths": only_diagnostic,
        "expected_host_source_delta_only": (
            changed_paths == EXPECTED_CHANGED_INPUTS
            and set(only_diagnostic) == EXPECTED_NEW_INPUTS
            and not only_baseline
        ),
        "production_metal_binary_hash_equal": any(
            item["path"] == "build-cardboard-frozenX/shaders/NumiMatter.metallib"
            and item["baseline_sha256"] == item["diagnostic_sha256"]
            for item in changed
        ) is False and a.get(
            "build-cardboard-frozenX/shaders/NumiMatter.metallib"
        ) == b.get("build-cardboard-frozenX/shaders/NumiMatter.metallib"),
    }


def source_snapshot_delta(baseline: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    a_root = Path(baseline["prediction"].get("source_snapshot", ""))
    b_root = Path(diagnostic["prediction"].get("source_snapshot", ""))
    if not a_root.is_dir() or not b_root.is_dir():
        raise RuntimeError(f"missing source snapshot: {a_root} or {b_root}")

    def inventory(directory: Path) -> dict[str, str]:
        return {
            path.relative_to(directory).as_posix(): digest(path)
            for path in directory.rglob("*") if path.is_file()
        }

    a, b = inventory(a_root), inventory(b_root)
    common = sorted(a.keys() & b.keys())
    changed = [
        {"path": name, "baseline_sha256": a[name], "diagnostic_sha256": b[name]}
        for name in common if a[name] != b[name]
    ]
    return {
        "baseline_snapshot": str(a_root),
        "diagnostic_snapshot": str(b_root),
        "baseline_file_count": len(a),
        "diagnostic_file_count": len(b),
        "baseline_files_missing_from_diagnostic": sorted(a.keys() - b.keys()),
        "diagnostic_only_file_count": len(b.keys() - a.keys()),
        "common_file_count": len(common),
        "changed_common_files": changed,
        "note": "The 033 snapshot is broader than the retained 031 snapshot. Changed-common-file list is the direct content comparison; diagnostic-only files are not asserted to be new edits relative to the complete 031 checkout.",
    }


def frozen_manifest_delta(root: Path) -> dict[str, Any]:
    old_path = root / "solver" / "frozen031.json"
    new_path = root / "solver" / "frozen033.json"
    if not old_path.is_file() or not new_path.is_file():
        raise RuntimeError("missing frozen031/frozen033 runtime manifest")
    old_raw = load(old_path).get("runtime_inputs_sha256", {})
    new_raw = load(new_path).get("sha256", {})

    def norm(key: str) -> str:
        for prefix in ("build-cardboard-frozen031/", "build-cardboard-frozen033/"):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    old = {norm(key): value for key, value in old_raw.items()}
    new = {norm(key): value for key, value in new_raw.items()}
    common = old.keys() & new.keys()
    changed = [
        {"path": key, "frozen031_sha256": old[key], "frozen033_sha256": new[key]}
        for key in sorted(common) if old[key] != new[key]
    ]
    shader_paths = [key for key in common if key.startswith("shaders/")]
    return {
        "frozen031_manifest": str(old_path.relative_to(root)),
        "frozen033_manifest": str(new_path.relative_to(root)),
        "frozen031_artifact_count": len(old),
        "frozen033_artifact_count": len(new),
        "not_listed_in_frozen033": sorted(old.keys() - new.keys()),
        "newly_listed_in_frozen033": sorted(new.keys() - old.keys()),
        "changed_artifact_hashes": changed,
        "all_common_shader_hashes_identical": all(old[key] == new[key] for key in shader_paths),
        "common_shader_count": len(shader_paths),
        "production_matter_metallib_identical": old.get("shaders/NumiMatter.metallib") == new.get("shaders/NumiMatter.metallib"),
        "scope_note": "Host executables can change when relinked against the host-only diagnostic runtime; shader identity is checked separately. Missing manifest entries mean that target was not listed in that frozen artifact inventory, not that it was necessarily removed from source.",
    }


def emit_source_patch(root: Path, baseline: dict[str, Any], diagnostic: dict[str, Any], delta: dict[str, Any]) -> Path:
    a_root = Path(baseline["prediction"]["source_snapshot"])
    b_root = Path(diagnostic["prediction"]["source_snapshot"])
    chunks: list[str] = []
    for item in delta["changed_common_files"]:
        name = item["path"]
        a = (a_root / name).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        b = (b_root / name).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        chunks.extend(difflib.unified_diff(a, b, fromfile=f"frozen031/{name}", tofile=f"frozen033/{name}"))
    patch_path = root / "operator-fd" / "source-diff-031-vs-033.patch"
    patch_path.write_text("".join(chunks), encoding="utf-8")
    return patch_path


def read_fd_trace(entry: dict[str, Any]) -> dict[str, Any]:
    if not entry["stderr"].is_file():
        raise RuntimeError(f"missing stderr trace: {entry['path']}")
    traces = []
    for line in entry["stderr"].read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("fem_operator_fd_iteration="):
            traces.append(json.loads(line.split("=", 1)[1]))
    if len(traces) != 1:
        raise RuntimeError(f"expected exactly one FD trace in {entry['path']}, got {len(traces)}")
    return traces[0]


def epsilon_pair(trace: dict[str, Any], environment: int, value: float) -> dict[str, Any]:
    try:
        pairs = trace["environments"][environment]["epsilon_pairs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"missing FD environment {environment}") from exc
    matches = [p for p in pairs if abs(float(p["epsilon"]) - value) <= 1e-12]
    if len(matches) != 1:
        raise RuntimeError(f"expected one epsilon={value} pair in environment {environment}, got {len(matches)}")
    return matches[0]


def output_parity(baseline: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    a, b = baseline["directory"] / "output", diagnostic["directory"] / "output"

    def obj_map(directory: Path) -> dict[str, str]:
        return {
            path.relative_to(directory).as_posix(): digest(path)
            for path in directory.rglob("*.obj")
        }

    meshes_a, meshes_b = obj_map(a), obj_map(b)
    return {
        "observations_csv_identical": digest(baseline["observations"]) == digest(diagnostic["observations"]),
        "result_json_identical": digest(baseline["result"]) == digest(diagnostic["result"]),
        "compiled_material_pack_identical": digest(baseline["compiled_pack"]) == digest(diagnostic["compiled_pack"]),
        "mesh_count_baseline": len(meshes_a),
        "mesh_count_diagnostic": len(meshes_b),
        "all_mesh_hashes_identical": meshes_a == meshes_b,
        "mesh_hashes_baseline": meshes_a,
        "mesh_hashes_diagnostic": meshes_b,
        "native_baseline_result": load(baseline["result"]),
        "native_diagnostic_result": load(diagnostic["result"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    base = case(root, BASELINE)
    scaled = {name: case(root, rel) for name, rel in SCALED_CASES.items()}
    for entry in [base, *scaled.values()]:
        if not entry["input_verification"]["all_verified"]:
            raise RuntimeError(f"sealed input verification failed for {entry['path']}: {entry['input_verification']['invalid'][:5]}")

    source_delta = source_snapshot_delta(base, scaled["scale8"])
    manifest_delta = frozen_manifest_delta(root)
    source_patch = emit_source_patch(root, base, scaled["scale8"], source_delta)
    traces = {name: read_fd_trace(entry) for name, entry in scaled.items()}
    base_result = load(base["result"])
    base_native_fp = base_result.get("compiled_world_fingerprint")
    for name, entry in scaled.items():
        result = load(entry["result"])
        if entry["prediction"].get("environment", {}).get("NM_FEM_OPERATOR_FD_EPSILON_SCALE") != ("8" if name == "scale8" else "16"):
            raise RuntimeError(f"sealed epsilon scale does not match expected case: {entry['path']}")
        if result.get("compiled_world_fingerprint") != base_native_fp:
            raise RuntimeError(f"compiled native-world fingerprint changed: {entry['path']}")

    common8 = epsilon_pair(traces["scale8"], 0, 0.25)
    common16 = epsilon_pair(traces["scale16"], 0, 0.25)
    compared_fields = (
        "dt", "position_delta", "free_node_count", "residual_plus_l2", "residual_minus_l2",
        "operator_action_l2", "centered_fd_l2", "error_l2", "relative_error",
        "action_dot_fd", "action_fd_cosine", "stage_status_codes_plus_minus",
        "minimum_j_plus_minus", "operator_status_code", "input_integrity_flag",
        "direction_nonzero", "maximum_delta_velocity", "maximum_position_perturbation",
    )
    common_equal = {field: common8.get(field) == common16.get(field) for field in compared_fields}
    parities = {name: output_parity(base, entry) for name, entry in scaled.items()}
    input_deltas = {name: input_delta(base, entry) for name, entry in scaled.items()}

    result = {
        "schema": "numi.cardboard.operator-fd-scaled-epsilon-comparison.v1",
        "derived_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "POSTRUN_DIAGNOSTIC_SENSITIVITY_ONLY",
        "limitations": [
            "This does not identify the cause of operator-versus-residual disagreement; projection, active-set behavior, and FP32 effects remain possible contributors.",
            "The source/runtime differs intentionally for host diagnostic epsilon scaling; input maps are verified and compared, not claimed identical.",
            "No accepted native path, material law, physical validation, or convergence admission is implied.",
            "The second environment has a zero correction direction and is only a vacuous control.",
        ],
        "cases": {
            "baseline": {"path": BASELINE, "receipt": base["receipt"], "source_snapshot": base["prediction"].get("source_snapshot"), "runtime_dir": base["prediction"].get("runtime_dir"), "input_verification": base["input_verification"]},
            **{
                name: {
                    "path": entry["path"], "receipt": entry["receipt"],
                    "epsilon_scale": entry["prediction"].get("environment", {}).get("NM_FEM_OPERATOR_FD_EPSILON_SCALE"),
                    "source_snapshot": entry["prediction"].get("source_snapshot"),
                    "runtime_dir": entry["prediction"].get("runtime_dir"),
                    "input_verification": entry["input_verification"],
                }
                for name, entry in scaled.items()
            },
        },
        "host_source_change": {
            "source_snapshot_comparison": source_delta,
            "frozen_artifact_manifest_comparison": manifest_delta,
            "source_diff_patch": str(source_patch.relative_to(root)),
            "normalized_sealed_input_deltas": input_deltas,
            "production_metal_hash_same_in_each_comparison": all(item["production_metal_binary_hash_equal"] for item in input_deltas.values()),
            "expected_compared_source_diffs": sorted(EXPECTED_CHANGED_INPUTS),
            "expected_diagnostic_only_doc": sorted(EXPECTED_NEW_INPUTS),
            "scope_note": "runtime.mm adds a bounded host-only diagnostic epsilon scale; cardboard_cohesive_face_check.mm changes the per-environment kinematic target count in its separate fixture. The compiled host probe changes. Production NumiMatter.metallib stays byte-identical to frozen031.",
        },
        "native_path_parity_to_frozen031_baseline": parities,
        "fd_traces": traces,
        "shared_epsilon_0_25_repeat": {
            "comparison": "environment 0, epsilon 0.25 in scale8 first pair versus scale16 second pair",
            "scale8_pair": common8,
            "scale16_pair": common16,
            "field_equality": common_equal,
            "all_compared_fields_exactly_equal": all(common_equal.values()),
        },
        "interpretation": "The two scaled captures test perturbation-size sensitivity while retaining an unchanged native outcome relative to frozen031. The shared epsilon 0.25 directly checks repeatability across the two host scales. This can bound epsilon sensitivity at the selected failure state but cannot establish the correct cause or certify the production Jacobian.",
    }
    out = root / "operator-fd" / "scaled-epsilon-comparison.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "comparison": str(out),
        "source_patch": str(source_patch),
        "inputs_verified": {name: value["input_verification"]["all_verified"] for name, value in {"baseline": base, **scaled}.items()},
        "native_path_parity": {name: {k: v for k, v in data.items() if k not in ("mesh_hashes_baseline", "mesh_hashes_diagnostic", "native_baseline_result", "native_diagnostic_result")} for name, data in parities.items()},
        "common_epsilon_0_25_exact_repeat": all(common_equal.values()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
