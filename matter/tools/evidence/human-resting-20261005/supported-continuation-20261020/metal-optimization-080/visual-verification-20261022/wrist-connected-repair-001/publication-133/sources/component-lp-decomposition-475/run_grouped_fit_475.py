"""Opt-in 475 component LP wrapper around the frozen 458/408 fit.

Requires an exact matched-input JSON pin and the explicit CLI flag
--component-decomposition-475. It leaves 458 validation and exact gates intact.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
RUNNER_458 = D / "D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
OWNER_408 = D / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OBJECTIVE_458 = RUNNER_458.with_name("fixed_bone_outward_plane_458.py")
DECOMPOSITION_475 = Path(__file__).with_name("component_lp_decomposition_475.py")
RUNNER_458_SHA = "08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93"
OWNER_408_SHA = "5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
OBJECTIVE_458_SHA = "056c8e60dad54f92a9c3ab69498dbfac0dc2c0152be272eadea2a58a3b6ea0e0"
SCHEMA = "numi.human.matched_inputs_component_fit_475.v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha_pin(record, label, expected_path):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"matched input {label} must contain exactly path and sha256")
    path = Path(record["path"]).resolve()
    if path != Path(expected_path).resolve():
        raise ValueError(f"matched input {label} path differs from actual fit argv")
    if not path.is_file() or sha(path) != record["sha256"]:
        raise ValueError(f"matched input {label} file/hash mismatch")
    return {"path": str(path), "sha256": record["sha256"]}


def validate_matched_inputs(config_path, actual_inputs, *, source_paths=None):
    """Validate exact data and implementation paths/hashes before loading 458."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("unknown matched-input schema")
    if set(config) != {"schema", "inputs", "source_pins"}:
        raise ValueError("matched-input config has unexpected or missing top-level fields")
    if set(actual_inputs) != {"candidate", "audit159", "face_target_npz", "point_target_npz"}:
        raise ValueError("actual matched input set is incomplete")
    inputs = config["inputs"]
    if set(inputs) != set(actual_inputs):
        raise ValueError("matched-input config roles do not match fit inputs")
    checked_inputs = {
        role: _sha_pin(inputs[role], role, actual_inputs[role]) for role in sorted(actual_inputs)
    }
    expected_sources = {
        "runner458": (RUNNER_458, RUNNER_458_SHA),
        "owner408": (OWNER_408, OWNER_408_SHA),
        "objective458": (OBJECTIVE_458, OBJECTIVE_458_SHA),
        "decomposition475": (DECOMPOSITION_475, sha(DECOMPOSITION_475)),
    }
    if source_paths is not None:
        for role, (path, digest) in source_paths.items():
            expected_sources[role] = (Path(path), digest)
    pins = config["source_pins"]
    if set(pins) != set(expected_sources):
        raise ValueError("matched-input source pin roles do not match active owners")
    checked_sources = {}
    for role, (expected_path, expected_digest) in expected_sources.items():
        rec = pins[role]
        if not isinstance(rec, dict) or set(rec) != {"path", "sha256"}:
            raise ValueError(f"source pin {role} must contain exactly path and sha256")
        actual_path = Path(rec["path"]).resolve()
        if actual_path != Path(expected_path).resolve() or not actual_path.is_file():
            raise ValueError(f"source pin {role} path mismatch or missing file")
        actual_digest = sha(actual_path)
        if rec["sha256"] != actual_digest or actual_digest != expected_digest:
            raise ValueError(f"source pin {role} SHA mismatch")
        checked_sources[role] = {"path": str(actual_path), "sha256": actual_digest}
    return {
        "schema": SCHEMA, "config_path": str(config_path.resolve()),
        "config_sha256": sha(config_path), "inputs": checked_inputs,
        "source_pins": checked_sources,
    }


def _fit_args():
    argv = list(sys.argv[1:])
    flag = "--component-decomposition-475"
    if argv.count(flag) != 1:
        raise ValueError("475 wrapper requires exactly one --component-decomposition-475 opt-in")
    argv.remove(flag)
    sys.argv[:] = [sys.argv[0], *argv]
    def value(name, required=True):
        if name not in argv:
            if required:
                raise ValueError(f"fit argv missing {name}")
            return None
        i = argv.index(name) + 1
        if i >= len(argv):
            raise ValueError(f"fit argv missing value after {name}")
        return Path(argv[i])
    base = value("--base-candidate")
    face = value("--face-target-npz")
    point = value("--point-target-npz")
    out = value("--output")
    audit_env = os.environ.get("NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458")
    audit = Path(audit_env) if audit_env else base.parent / "audit-159/report.json"
    config = os.environ.get("NUMI_MATCHED_INPUTS_JSON_475")
    if not config:
        raise ValueError("NUMI_MATCHED_INPUTS_JSON_475 must point to exact matched input pins")
    return {"candidate": base, "audit159": audit, "face_target_npz": face,
            "point_target_npz": point}, out, Path(config)


def main():
    actual_inputs, output_dir, config_path = _fit_args()
    matched = validate_matched_inputs(config_path, actual_inputs)
    # Fail before importing or running the fit if frozen owners drifted.
    for path, digest in ((RUNNER_458, RUNNER_458_SHA), (OWNER_408, OWNER_408_SHA),
                         (OBJECTIVE_458, OBJECTIVE_458_SHA)):
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"frozen 458/408 source pin mismatch: {path}")
    if not DECOMPOSITION_475.is_file():
        raise FileNotFoundError(DECOMPOSITION_475)
    limit_text = os.environ.get("NUMI_COMPONENT_LP_MAX_TOTAL_WALL_S_475", "120")
    max_total_wall_s = float(limit_text)
    if not (0 < max_total_wall_s <= 120.0):
        raise ValueError("475 total solver budget must be in (0, 120] seconds")
    variant = _load("fixed_bone_outward_runner_458_for_475", RUNNER_458)
    variant.EXPECTED_CANDIDATE_SHA = matched["inputs"]["candidate"]["sha256"]
    variant.EXPECTED_AUDIT_REPORT_SHA = matched["inputs"]["audit159"]["sha256"]
    variant.EXPECTED_FACE_TARGET_SHA = matched["inputs"]["face_target_npz"]["sha256"]
    variant.EXPECTED_POINT_TARGET_SHA = matched["inputs"]["point_target_npz"]["sha256"]
    decomposition = _load("component_lp_decomposition_475_runtime", DECOMPOSITION_475)
    loaded = {}
    original_load = variant.load
    def load_with_component_install(name, path):
        module = original_load(name, path)
        if Path(path).resolve() == OWNER_408.resolve():
            install_receipt = decomposition.install(module, max_total_wall_s=max_total_wall_s)
            loaded["owner"] = module
            loaded["install"] = install_receipt
        return module
    variant.load = load_with_component_install
    variant.main()
    report_path = output_dir / "report.json"
    report = json.loads(report_path.read_text())
    if "owner" not in loaded:
        raise RuntimeError("frozen 458 fit completed without installing 475 solver")
    report["component_lp_decomposition_475"] = {
        "classification": "opt-in construction-only sparse component LP; full exact downstream gates remain mandatory",
        "matched_inputs": matched,
        "solver_installation": loaded["install"],
        "helper_path": str(DECOMPOSITION_475),
        "helper_sha256": sha(DECOMPOSITION_475),
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": sha(__file__),
        "frozen_458_path": str(RUNNER_458),
        "frozen_458_sha256": sha(RUNNER_458),
        "max_total_wall_s": max_total_wall_s,
        "global_stage2_l1_optimality_claim": False,
        "native_admission": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pins_after = {str(p): sha(p) for p in (RUNNER_458, OWNER_408, OBJECTIVE_458,
                                                   DECOMPOSITION_475, Path(__file__))}
    report["component_lp_decomposition_475"]["source_pins_after_run"] = pins_after
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if any(sha(Path(p)) != d for p, d in pins_after.items()):
        raise RuntimeError("475 source changed during fit")


if __name__ == "__main__":
    main()
