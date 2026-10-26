"""Inert coherent-depth target adapter over the pinned 545 builder.

This uses only 545's existing joint-source-patch/coherent-depth mode, with
the same tested multi-parent 303 normal adapter. Partial group failures are
retained as diagnostics; no all-target success is forced or inferred.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
BUILDER = ROOT / "D/contained-clearance-545/build_joint_source_targets545.py"
BUILDER_SHA256 = "f4066f7db73c37911ed75ef48eca92116e6a15ea8d1fe6543af597a5655398e0"
DIRECT_ADAPTER = ROOT / "D/surface-clearance-546/direct-depth-targets-546/run_direct_depth_targets546.py"
DIRECT_ADAPTER_SHA256 = "0708d259c7bc7dd6341b8cac737f6516ac611011e6d4f599e05530cf91106ac5"
EXTENSION = ROOT / "D/source-outward-proposal-500/build_outward_targets_multi_source_546.py"
EXTENSION_SHA256 = "dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
OWNER500_SHA256 = "454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf"


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
    parser.add_argument("--coherent-direction", default=None)
    parser.add_argument("--minimum-depth-cosine", type=float, default=0.05)
    parser.add_argument("--joint-source-patch", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.direction_mode != "coherent-depth":
        raise ValueError("this adapter requires explicit --direction-mode coherent-depth")
    if args.coherent_direction != "depth":
        raise ValueError("this adapter requires explicit --coherent-direction depth")
    if not args.joint_source_patch:
        raise ValueError("this adapter requires --joint-source-patch")
    if args.minimum_depth_cosine != 0.05:
        raise ValueError("this adapter pins the builder's minimum-depth-cosine metadata at 0.05")
    return {
        "requested_direction_mode": args.direction_mode,
        "requested_coherent_direction": args.coherent_direction,
        "joint_source_patch": True,
        "requested_minimum_depth_cosine_metadata": args.minimum_depth_cosine,
    }


def summarize_group_certificates(report, arrays_path, requested):
    if report.get("direction_mode") != requested["requested_direction_mode"]:
        raise ValueError("545 actual direction mode differs from requested mode")
    if report.get("coherent_direction_policy") != requested["requested_coherent_direction"]:
        raise ValueError("545 actual coherent direction differs from requested mode")
    if report.get("joint_source_support_patch") is not True:
        raise ValueError("545 did not use the requested joint source patch")
    if report.get("minimum_depth_cosine") != requested["requested_minimum_depth_cosine_metadata"]:
        raise ValueError("545 direction-mode metadata differs from requested value")

    targets = report.get("targets")
    rows = [int(row.get("query_row", -1)) for row in targets] if isinstance(targets, list) else []
    if (not isinstance(targets, list) or len(rows) != report.get("query_rows")
            or len(set(rows)) != len(rows) or any(row < 0 for row in rows)):
        raise ValueError("545 target rows are absent, duplicated, or incomplete")
    groups = report.get("coherent_group_rows")
    if not isinstance(groups, list) or not groups:
        raise ValueError("545 report lacks coherent group membership")
    normalized_groups = []
    seen = set()
    for group in groups:
        if not isinstance(group, list) or not group:
            raise ValueError("545 coherent group is empty or malformed")
        normalized = [int(row) for row in group]
        if normalized != sorted(set(normalized)):
            raise ValueError("545 coherent group rows are not unique and sorted")
        if seen.intersection(normalized):
            raise ValueError("545 coherent groups overlap")
        seen.update(normalized)
        normalized_groups.append(normalized)
    if seen != set(rows):
        raise ValueError("545 coherent groups do not exactly cover target rows")
    # This comparator explicitly requests one joint source-support patch.
    if len(normalized_groups) != 1 or normalized_groups[0] != sorted(rows):
        raise ValueError("joint-source-patch output is not one group covering all target rows")

    by_row = {int(target["query_row"]): target for target in targets}
    group_records = []
    for group in normalized_groups:
        failures = []
        translations = []
        for row in group:
            target = by_row[row]
            status = target.get("status")
            if status != "PASS_FIRST_UNION_EXIT":
                failures.append({"query_row": row, "status": status})
                continue
            if target.get("coherent_group_rows") != group:
                failures.append({"query_row": row, "status": "MISSING_SHARED_GROUP_CERTIFICATE"})
                continue
            if target.get("coherent_exact_endpoint_bank_count") != 185:
                failures.append({"query_row": row, "status": "MISSING_EXACT_ALL185_SHARED_ENDPOINT_CERTIFICATE"})
                continue
            selection = target.get("selection", {})
            if selection.get("coherent_endpoint_exact_outside_all185") is not True:
                failures.append({"query_row": row, "status": "SHARED_ENDPOINT_NOT_CERTIFIED_OUTSIDE_ALL185"})
                continue
            direction = target.get("direction_compatibility", {})
            depth_cos = direction.get("selected_depth_cosine")
            if isinstance(depth_cos, bool) or not isinstance(depth_cos, (int, float)) or depth_cos <= 0:
                failures.append({"query_row": row, "status": "NONPOSITIVE_SHARED_DEPTH_DIRECTION"})
                continue
            vector = target.get("coherent_translation_world_m")
            if not isinstance(vector, list) or len(vector) != 3:
                failures.append({"query_row": row, "status": "MISSING_SHARED_TRANSLATION"})
                continue
            translations.append(np.asarray(vector, dtype=np.float64))
        # The owner must have emitted one bitwise-identical translation for all
        # passing members before we report this group as coherently certified.
        if not failures and len(translations) == len(group):
            first = translations[0]
            if not all(np.array_equal(value, first) for value in translations[1:]):
                failures.append({"query_row": None, "status": "SHARED_TRANSLATION_MISMATCH"})
        group_records.append({
            "group_rows": group,
            "status": "PASS_SHARED_TRANSLATION_ALL185" if not failures else "FAILED_GROUP_CERTIFICATE",
            "failed_rows": failures,
            "certified_target_count": 0 if failures else len(group),
            "all185_exact_endpoint_certified": not failures,
            "shared_translation_world_m": translations[0].tolist() if translations and not failures else None,
        })

    arrays_path = Path(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if not np.array_equal(np.asarray(arrays["query_rows"], dtype=np.int64), np.asarray(rows, dtype=np.int64)):
            raise ValueError("target array row identities differ from report")
        pass_mask = np.asarray(arrays["pass_mask"], dtype=bool)
        exact_mask = np.asarray(arrays["target_exactly_outside_all185"], dtype=bool)
        if pass_mask.shape != (len(rows),) or exact_mask.shape != pass_mask.shape:
            raise ValueError("target array masks have invalid dimensions")
        expected_pass = np.asarray([by_row[row].get("status") == "PASS_FIRST_UNION_EXIT" for row in rows])
        if not np.array_equal(pass_mask, expected_pass) or not np.array_equal(exact_mask, expected_pass):
            raise ValueError("target array masks do not reflect final target statuses")
        if not np.asarray(arrays["direction_compatibility_mask"], dtype=bool).all():
            raise ValueError("coherent-depth target array omits depth-direction provenance")

    group_pass = all(group["status"] == "PASS_SHARED_TRANSLATION_ALL185" for group in group_records)
    return {
        "requested_direction_mode": requested["requested_direction_mode"],
        "actual_direction_mode": report["direction_mode"],
        "requested_coherent_direction": requested["requested_coherent_direction"],
        "actual_coherent_direction": report["coherent_direction_policy"],
        "joint_source_patch": report["joint_source_support_patch"],
        "target_count": len(rows),
        "group_count": len(group_records),
        "passing_group_count": sum(group["status"] == "PASS_SHARED_TRANSLATION_ALL185" for group in group_records),
        "failed_group_count": sum(group["status"] != "PASS_SHARED_TRANSLATION_ALL185" for group in group_records),
        "groups": group_records,
        "all_groups_exact_all185": group_pass,
        "all_target_statuses_pass": all(by_row[row].get("status") == "PASS_FIRST_UNION_EXIT" for row in rows),
        "native_admission": False,
        "accepted_physical_steps": 0,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    requested = validate_invocation(argv)
    for path, expected, label in (
        (BUILDER, BUILDER_SHA256, "545 joint target builder"),
        (DIRECT_ADAPTER, DIRECT_ADAPTER_SHA256, "existing 546 source303 installer"),
        (EXTENSION, EXTENSION_SHA256, "546 multi-parent source extension"),
    ):
        if sha(path) != expected:
            raise ValueError("pinned " + label + " changed")
    adapter = loadmod(DIRECT_ADAPTER, "direct_depth_adapter_for_coherent_mode")
    extension = loadmod(EXTENSION, "multi_parent_extension_for_coherent_mode")
    owner500 = Path(extension.OWNER500)
    if sha(owner500) != OWNER500_SHA256 or extension.OWNER500_SHA != OWNER500_SHA256:
        raise ValueError("pinned 500 source-normal owner mismatch")
    builder = loadmod(BUILDER, "joint_source_target_builder545_for_coherent_mode")
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
    group_summary = summarize_group_certificates(report, arrays_path, requested)
    provenance_json = json.dumps(face_provenance, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()
    report["coherent_depth_adapter"] = {
        **group_summary,
        "direction_contract": (
            "Reuse 545 coherent-depth grouping and shared maximum-minimum depth direction; "
            "with --joint-source-patch the registered-source support must be one connected group. "
            "Record failed groups and keep their individual/status masks unchanged."
        ),
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
    }
    report["native_admission"] = False
    report["accepted_physical_steps"] = 0
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "report": str(report_path), "report_sha256": sha(report_path),
        "target_npz": str(arrays_path), "target_npz_sha256": sha(arrays_path),
        "requested_direction_mode": requested["requested_direction_mode"],
        "actual_direction_mode": group_summary["actual_direction_mode"],
        "target_count": group_summary["target_count"],
        "passing_group_count": group_summary["passing_group_count"],
        "failed_group_count": group_summary["failed_group_count"],
        "status": report["status"], "native_admission": False,
        "accepted_physical_steps": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
