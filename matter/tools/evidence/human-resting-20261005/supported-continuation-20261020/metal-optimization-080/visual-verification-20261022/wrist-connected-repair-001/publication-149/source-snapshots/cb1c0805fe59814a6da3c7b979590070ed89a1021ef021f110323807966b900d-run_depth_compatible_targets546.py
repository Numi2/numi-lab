"""Inert depth-compatible point-target adapter over the pinned 545 builder.

Only proposal directions change: each uses the registered-source-outward normal
projected to satisfy the matched point-depth cosine floor. The frozen 545
all-185 ray and endpoint checks remain the target certifier.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
BUILDER = ROOT / "D/contained-clearance-545/build_joint_source_targets545.py"
BUILDER_SHA256 = "f4066f7db73c37911ed75ef48eca92116e6a15ea8d1fe6543af597a5655398e0"
DIRECT_ADAPTER = ROOT / "D/surface-clearance-546/direct-depth-targets-546/run_direct_depth_targets546.py"
DIRECT_ADAPTER_SHA256 = "0708d259c7bc7dd6341b8cac737f6516ac611011e6d4f599e05530cf91106ac5"
EXTENSION = ROOT / "D/source-outward-proposal-500/build_outward_targets_multi_source_546.py"
EXTENSION_SHA256 = "dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
OWNER500_SHA256 = "454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf"
MODE = "depth-compatible"
MINIMUM_DEPTH_COSINE = 0.05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loadmod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned module " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_invocation(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--direction-mode", default=None)
    parser.add_argument("--minimum-depth-cosine", type=float, default=MINIMUM_DEPTH_COSINE)
    parser.add_argument("--joint-source-patch", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.direction_mode != MODE:
        raise ValueError("this adapter requires explicit --direction-mode depth-compatible")
    if not math.isfinite(args.minimum_depth_cosine) or args.minimum_depth_cosine != MINIMUM_DEPTH_COSINE:
        raise ValueError("this adapter pins --minimum-depth-cosine 0.05")
    if args.joint_source_patch:
        raise ValueError("joint-source-patch is outside this target-mode adapter")
    return {"requested_direction_mode": args.direction_mode,
            "requested_minimum_depth_cosine": args.minimum_depth_cosine}


def validate_target_output(report, arrays_path, requested):
    if report.get("direction_mode") != requested["requested_direction_mode"]:
        raise ValueError("545 actual direction mode differs from requested mode")
    if report.get("minimum_depth_cosine") != requested["requested_minimum_depth_cosine"]:
        raise ValueError("545 actual depth cosine floor differs from pinned 0.05")
    targets = report.get("targets")
    if (report.get("status") != "PASS_FIRST_UNION_EXIT_TARGETS_ONLY"
            or not isinstance(targets, list)
            or len(targets) != report.get("query_rows")
            or report.get("passing_targets") != len(targets)):
        raise ValueError("545 did not certify every target")
    for row in targets:
        if row.get("status") != "PASS_FIRST_UNION_EXIT":
            raise ValueError("a target lacks a certified first union exit")
        if row.get("target_exactly_outside_all185") is not True or row.get("exact_endpoint_bank_count") != 185:
            raise ValueError("a target lacks exact outside classification against all185 banks")
        direction = row.get("direction_compatibility")
        if not isinstance(direction, dict):
            raise ValueError("a target lacks the depth-compatible direction certificate")
        depth_cos = direction.get("selected_depth_cosine")
        source_cos = direction.get("selected_source_hemisphere_cosine")
        if (isinstance(depth_cos, bool) or not isinstance(depth_cos, (int, float))
                or not math.isfinite(depth_cos) or depth_cos < MINIMUM_DEPTH_COSINE):
            raise ValueError("a target violates the matched depth cosine floor")
        if (isinstance(source_cos, bool) or not isinstance(source_cos, (int, float))
                or not math.isfinite(source_cos) or source_cos <= 0.0):
            raise ValueError("a target leaves the positive source-outward hemisphere")
    with __import__("numpy").load(arrays_path, allow_pickle=False) as z:
        if len(z["query_rows"]) != len(targets):
            raise ValueError("target arrays and report have different row counts")
        if not z["pass_mask"].all() or not z["target_exactly_outside_all185"].all():
            raise ValueError("target arrays do not certify every exact endpoint")
        if not z["direction_compatibility_mask"].all():
            raise ValueError("target arrays omit a depth-compatibility result")
    return {
        "actual_direction_mode": report["direction_mode"],
        "actual_minimum_depth_cosine": report["minimum_depth_cosine"],
        "target_count": len(targets),
        "all185_exact_endpoint_rows": len(targets),
        "minimum_observed_depth_cosine": min(
            float(row["direction_compatibility"]["selected_depth_cosine"]) for row in targets),
        "minimum_observed_source_hemisphere_cosine": min(
            float(row["direction_compatibility"]["selected_source_hemisphere_cosine"]) for row in targets),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    requested = validate_invocation(argv)
    for path, expected, label in (
        (BUILDER, BUILDER_SHA256, "545 joint target builder"),
        (DIRECT_ADAPTER, DIRECT_ADAPTER_SHA256, "pinned 546 source303 adapter"),
        (EXTENSION, EXTENSION_SHA256, "546 multi-parent source extension"),
    ):
        if sha(path) != expected:
            raise ValueError("pinned " + label + " changed")
    adapter = loadmod(DIRECT_ADAPTER, "direct_depth_adapter_for_compatible_mode")
    extension = loadmod(EXTENSION, "multi_parent_extension_for_compatible_mode")
    owner500 = Path(extension.OWNER500)
    if sha(owner500) != OWNER500_SHA256 or extension.OWNER500_SHA != OWNER500_SHA256:
        raise ValueError("pinned 500 source-normal owner mismatch")
    builder = loadmod(BUILDER, "joint_source_target_builder545_for_compatible_mode")
    face_provenance = {}
    original_loadmod = builder.loadmod

    def load_with_multisource(path, name):
        module = original_loadmod(path, name)
        if Path(path).resolve() == Path(builder.SOURCE303).resolve():
            adapter.install_source303_extension(module, extension, face_provenance)
        return module

    builder.loadmod = load_with_multisource
    builder.main(argv)

    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--output-dir", type=Path, required=True)
    args, _ = cli.parse_known_args(argv)
    output = args.output_dir.resolve()
    report_path = output / "nearest-union-exits.json"
    arrays_path = output / "nearest-union-exits.npz"
    if not report_path.is_file() or not arrays_path.is_file():
        raise RuntimeError("545 target builder omitted its report or target arrays")
    report = json.loads(report_path.read_text())
    cert = validate_target_output(report, arrays_path, requested)
    provenance_json = json.dumps(face_provenance, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()
    report["depth_compatible_adapter"] = {
        "requested_direction_mode": requested["requested_direction_mode"],
        "actual_direction_mode": cert["actual_direction_mode"],
        "requested_minimum_depth_cosine": requested["requested_minimum_depth_cosine"],
        "actual_minimum_depth_cosine": cert["actual_minimum_depth_cosine"],
        "direction_contract": (
            "Use the closest unit vector to each registered source-outward normal "
            "that satisfies the matched 357 depth-covector dot-product floor of 0.05 "
            "and remains in the positive source-outward hemisphere."
        ),
        "target_certification": cert,
        "multi_parent_source_normal_extension": {
            "reused_installer_path": str(DIRECT_ADAPTER),
            "reused_installer_sha256": DIRECT_ADAPTER_SHA256,
            "pinned_545_builder_path": str(BUILDER),
            "pinned_545_builder_sha256": BUILDER_SHA256,
            "pinned_546_extension_path": str(EXTENSION),
            "pinned_546_extension_sha256": EXTENSION_SHA256,
            "pinned_500_owner_path": str(owner500),
            "pinned_500_owner_sha256": OWNER500_SHA256,
            "single_parent_303_normal_function_preserved": True,
            "multi_parent_face_rows": sorted(face_provenance),
            "multi_parent_face_count": len(face_provenance),
            "multi_parent_provenance_sha256": hashlib.sha256(provenance_json).hexdigest(),
            "all_source_parent_provenance_retained": True,
        },
        "exact_185_bank_ray_and_endpoint_predicates_unchanged": True,
        "proposal_only": True,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "report": str(report_path), "report_sha256": sha(report_path),
        "target_npz": str(arrays_path), "target_npz_sha256": sha(arrays_path),
        "requested_direction_mode": requested["requested_direction_mode"],
        "actual_direction_mode": cert["actual_direction_mode"],
        "target_count": cert["target_count"], "status": report["status"],
        "native_admission": False, "accepted_physical_steps": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
