"""Strict matched-input delegator for the frozen 485 component fit.

This wrapper accepts the existing 475 matched-input JSON schema, validates
actual CLI paths and every source pin, then loads 485 with those exact input
hashes. It does not change the solver or any downstream acceptance gate.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
SCHEMA = "numi.human.matched_inputs_component_fit_475.v1"

# Frozen entrypoint and every owner/helper it can exercise.
RUNNER_485 = D / "D/D/component-lp-decomposition-485/run_matched_component_fit_485.py"
HELPER_485 = D / "D/D/component-lp-decomposition-485/component_lp_decomposition_485.py"
SOFT_470 = D / "D/D/soft-target-nonregression-470/soft_target_nonregression_470.py"
RUNNER_458 = D / "D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
OWNER_408 = D / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OBJECTIVE_458 = D / "D/D/fixed-bone-outward-plane-458/fixed_bone_outward_plane_458.py"
FIXED_458 = D / "D/D/fixed-bone-outward-plane-458/fixed_bone_outward_plane_458.py"
DECOMPOSITION_475 = D / "D/D/component-lp-decomposition-475/component_lp_decomposition_475.py"
WRAPPER_475 = D / "D/D/component-lp-decomposition-475/run_grouped_fit_475.py"
EMITTER_314 = D / "D/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py"
AUDIT_159 = D / "audit_skin_bone_surfaces_159.py"
BANK_READER = D / "publication-084/canonical/matter/tools/human_skin_fat_bone_audit.py"
EXACT_159 = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")

# Four source roles required by the existing 475 config schema.
MATCHED_SOURCE_PINS = {
    "runner458": (RUNNER_458, "08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93"),
    "owner408": (OWNER_408, "5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"),
    "objective458": (OBJECTIVE_458, "056c8e60dad54f92a9c3ab69498dbfac0dc2c0152be272eadea2a58a3b6ea0e0"),
    "decomposition475": (DECOMPOSITION_475, "af0b6a6bf40757a403a8b7c359233694241d34e1e016b6dd3a37d7c98789c46d"),
}
# Broader source lock, including the 485 wrapper/helper, soft-target shim and
# 314/159/bank owners that are exercised downstream.
SOURCE_LOCK = {
    RUNNER_485: "3e504da05965405506382338da3b71c81442f40d8c0eb7b16c612b9375633a5f",
    HELPER_485: "b8632e0734ff0163e854b9c0c5fbfe33c6d12c7b17cbb7935c5028c8df78ef74",
    SOFT_470: "990e126932186e33c0a5363a6ac34fa967e211f4bc55977567582e2b89b8bb79",
    RUNNER_458: MATCHED_SOURCE_PINS["runner458"][1],
    OWNER_408: MATCHED_SOURCE_PINS["owner408"][1],
    OBJECTIVE_458: MATCHED_SOURCE_PINS["objective458"][1],
    FIXED_458: "056c8e60dad54f92a9c3ab69498dbfac0dc2c0152be272eadea2a58a3b6ea0e0",
    DECOMPOSITION_475: MATCHED_SOURCE_PINS["decomposition475"][1],
    WRAPPER_475: "7312e292b8711a8d6b0914d3519b7db762fe446cf897d59bd0d00be3476c76b1",
    EMITTER_314: "3422f30a61db0724d234c487d70fc3d74237a802261c2f9a72af9e15843a1c78",
    AUDIT_159: "de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1",
    BANK_READER: "4b2e084df442e1fc7491188696c410609228215b70b98db43640688b466f3e81",
    EXACT_159: "f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _record(record, label, expected_path):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} record must contain exactly path and sha256")
    digest = record["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} SHA-256 is malformed")
    path = Path(record["path"]).resolve()
    if path != Path(expected_path).resolve():
        raise ValueError(f"{label} path does not match the actual fit input")
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"{label} file/hash mismatch")
    return {"path": str(path), "sha256": digest}


def validate_matched_inputs(config_path, actual_inputs, *, source_pins=None):
    """Validate the existing 475 JSON contract against exact active paths."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(), object_pairs_hook=_pairs_no_duplicates)
    if config.get("schema") != SCHEMA:
        raise ValueError("unknown matched-input schema")
    if set(config) != {"schema", "inputs", "source_pins"}:
        raise ValueError("matched-input config has unexpected or missing top-level fields")
    roles = {"candidate", "audit159", "face_target_npz", "point_target_npz"}
    if set(actual_inputs) != roles or set(config["inputs"]) != roles:
        raise ValueError("matched input roles are incomplete or unexpected")
    checked_inputs = {role: _record(config["inputs"][role], role, actual_inputs[role])
                      for role in sorted(roles)}
    expected = dict(MATCHED_SOURCE_PINS if source_pins is None else source_pins)
    pins = config["source_pins"]
    if set(pins) != set(expected):
        raise ValueError("matched-input source pin roles differ from the frozen 475 contract")
    checked_sources = {}
    for role, (path, digest) in expected.items():
        record = pins[role]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"source pin {role} must contain exactly path and sha256")
        actual_path = Path(record["path"]).resolve()
        if actual_path != Path(path).resolve() or not actual_path.is_file():
            raise ValueError(f"source pin {role} path mismatch or missing file")
        actual_digest = sha(actual_path)
        if record["sha256"] != digest or actual_digest != digest:
            raise ValueError(f"source pin {role} hash mismatch")
        checked_sources[role] = {"path": str(actual_path), "sha256": actual_digest}
    audit = json.loads(Path(checked_inputs["audit159"]["path"]).read_text())
    if audit.get("candidate_sha256") != checked_inputs["candidate"]["sha256"]:
        raise ValueError("159 audit does not bind to the matched candidate")
    return {
        "schema": SCHEMA, "config_path": str(config_path),
        "config_sha256": sha(config_path), "inputs": checked_inputs,
        "source_pins": checked_sources,
    }


def validate_source_lock():
    checked = {}
    for path, expected in SOURCE_LOCK.items():
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"490 frozen source pin mismatch: {path}")
        checked[str(path.resolve())] = expected
    return checked


def _take_value(argv, flag, *, required=True):
    if argv.count(flag) > 1:
        raise ValueError(f"{flag} may appear at most once")
    if flag not in argv:
        if required:
            raise ValueError(f"missing required {flag}")
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"missing value after {flag}")
    value = argv[index + 1]
    del argv[index:index + 2]
    return Path(value)


def extract_fit_inputs(argv):
    """Remove 490-only options and derive the exact underlying 458 inputs."""
    remaining = list(argv)
    config_path = _take_value(remaining, "--matched-inputs-json-490")
    audit_path = _take_value(remaining, "--audit159-490", required=False)
    def value(flag):
        if remaining.count(flag) != 1:
            raise ValueError(f"expected exactly one {flag}")
        i = remaining.index(flag)
        if i + 1 >= len(remaining):
            raise ValueError(f"missing value after {flag}")
        return Path(remaining[i + 1])
    candidate = value("--base-candidate")
    face = value("--face-target-npz")
    point = value("--point-target-npz")
    output = value("--output")
    if audit_path is None:
        env_audit = os.environ.get("NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458")
        audit_path = Path(env_audit) if env_audit else candidate.parent / "audit-159/report.json"
    return remaining, config_path, output, {
        "candidate": candidate, "audit159": audit_path,
        "face_target_npz": face, "point_target_npz": point,
    }


def _load_module(path):
    spec = importlib.util.spec_from_file_location("frozen_component_fit_485_for_490", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen 485 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    args, config_path, output_dir, actual_inputs = extract_fit_inputs(sys.argv[1:])
    matched = validate_matched_inputs(config_path, actual_inputs)
    source_before = validate_source_lock()
    self_path = Path(__file__).resolve()
    source_before[str(self_path)] = sha(self_path)
    config_sha_before = sha(config_path)
    runner = _load_module(RUNNER_485)
    runner.EXPECTED_CANDIDATE = matched["inputs"]["candidate"]["sha256"]
    runner.EXPECTED_AUDIT = matched["inputs"]["audit159"]["sha256"]
    runner.EXPECTED_FACE = matched["inputs"]["face_target_npz"]["sha256"]
    runner.EXPECTED_POINT = matched["inputs"]["point_target_npz"]["sha256"]
    os.environ["NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458"] = str(actual_inputs["audit159"].resolve())
    os.environ["NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458"] = matched["inputs"]["audit159"]["sha256"]
    os.environ["NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458"] = matched["inputs"]["candidate"]["sha256"]
    sys.argv[:] = [sys.argv[0], *args]
    runner.main()
    for role, record in matched["inputs"].items():
        if not Path(record["path"]).is_file() or sha(record["path"]) != record["sha256"]:
            raise RuntimeError(f"matched input changed during 490 fit: {role}")
    if sha(config_path) != config_sha_before:
        raise RuntimeError("matched-input config changed during 490 fit")
    source_after = validate_source_lock()
    source_after[str(self_path)] = sha(self_path)
    if source_after != source_before:
        raise RuntimeError("frozen owner source changed during 490 fit")
    report_path = output_dir / "report.json"
    report = json.loads(report_path.read_text())
    report["matched_component_fit_490"] = {
        "classification": "strictly matched construction-only delegation to frozen 485",
        "matched_inputs": matched,
        "frozen_source_pins_before": source_before,
        "frozen_source_pins_after": source_after,
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": sha(__file__),
        "delegated_485_path": str(RUNNER_485),
        "delegated_485_sha256": sha(RUNNER_485),
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
