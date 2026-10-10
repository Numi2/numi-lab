#!/usr/bin/env python3
"""Verify the closed accepted-force observer paired smoke; never runs simulation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/accepted-force-observer-1253/paired-smoke-001")
OUT = Path(__file__).with_name("verification.json")
HUMAN_SRC = Path("/Users/n/numi-human-q-audit-provenance-1240/src")
COMPARE_PATH = Path("/Users/n/numi-human-retained-delivery-20261009/q-integration-audit-window-1240/root-comparison-001/compare_short_arms.py")
PACK_OWNER = HUMAN_SRC / "numilab_human/torso_anatomy_audit.py"
RIGID_OWNER = HUMAN_SRC / "numilab_human/resting_scene.py"
MODEL_MANIFEST = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json")
RIGID_PATH = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
TENDON_PATH = Path("/Users/n/numi-human-resting-evidence-20261005/tendon-semantic-foot-migration-834/numi-human-tendon-attachments.nhtendon")
TENDON_MANIFEST = TENDON_PATH.with_name("numi-human-tendon-attachments.manifest.json")
EXPECTED_Q_STEPS = list(range(464, 513))
COMMON_CSV = [
    "resting-coupled.csv",
    "resting-com-momentum-diagnostic.csv",
    "resting-com-support-impulses.csv",
    "resting-surface-audit.csv",
    "resting-com-q-integration.csv",
    "resting-com-q-index-map.csv",
    "resting-com-q-support-slip.csv",
]
FORCE_CSV = "resting-com-q-force-contributions.csv"
EXPECTED_UNITS = ("N for translational generalized coordinates; N*m for rotational generalized "
                  "coordinates; DOF identity is local_v_index in resting-com-q-index-map.csv")
failures: list[str] = []
checks: dict[str, object] = {}
input_hashes: dict[str, str] = {}

def require(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def pin(path: Path) -> str:
    path = path.resolve()
    key = str(path)
    if key in input_hashes:
        return input_hashes[key]
    value = sha(path)
    input_hashes[key] = value
    return value

def read_csv(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)

def import_function(path: Path, name: str, function: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load owner source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)

def load_arm(arm: str):
    root = BASE / arm
    run = root / "native-run"
    decl_path = root / "run-declaration.json"
    execution_path = root / "execution.json"
    invocation_path = run / "invocation.json"
    metadata_path = run / "run-metadata.json"
    start_path = root / "execution-start.json"
    for p in (decl_path, execution_path, invocation_path, metadata_path, start_path):
        require(p.is_file(), f"{arm}: required receipt missing: {p}")
        if p.is_file():
            pin(p)
    decl = json.loads(decl_path.read_text())
    execution = json.loads(execution_path.read_text())
    invocation = json.loads(invocation_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    start = json.loads(start_path.read_text())
    declared_assets = decl.get("immutable_assets", {})
    observed_assets = metadata.get("asset_sha256", {})
    asset_subset_matches = (isinstance(declared_assets, dict) and isinstance(observed_assets, dict)
                            and bool(observed_assets)
                            and all(path in declared_assets and declared_assets[path] == digest
                                    for path, digest in observed_assets.items()))
    require(asset_subset_matches, f"{arm}: run-metadata asset pins are not a subset of declaration pins")
    asset_files_match = True
    for asset_path, expected_sha in declared_assets.items():
        path = Path(asset_path)
        if not path.is_file():
            asset_files_match = False
            failures.append(f"{arm}: declared immutable asset is missing: {asset_path}")
            continue
        actual_sha = pin(path)
        if actual_sha != expected_sha:
            asset_files_match = False
            failures.append(f"{arm}: declared immutable asset hash mismatch: {asset_path}")
    require(asset_files_match, f"{arm}: one or more declared immutable assets failed live hash verification")
    require(decl.get("accepted_steps") == 512 and decl.get("capture_steps") == [0, 512],
            f"{arm}: declaration horizon/capture schedule mismatch")
    require(decl.get("q_audit") == {"enabled": True, "expected_rows": 49, "first_accepted_step": 464, "last_accepted_step": 512},
            f"{arm}: declaration Q window mismatch: {decl.get('q_audit')}")
    expected_force = "1" if arm == "window" else "0"
    require(decl.get("force_audit", {}).get("enabled") is (arm == "window"),
            f"{arm}: declaration force-audit enabled flag mismatch")
    require(decl.get("force_audit", {}).get("expected_rows") == (49 if arm == "window" else 0),
            f"{arm}: declaration expected force-row count mismatch")
    require(execution.get("changed_inputs") == {},
            f"{arm}: native execution reports changed inputs")
    require(execution.get("native_argv_matches_prepared_cli_preview") is True and
            execution.get("native_invocation_present") is True,
            f"{arm}: native invocation did not match prepared CLI preview")
    require(metadata.get("exit_code") == 0 and
            metadata.get("source_files_changed_during_run") == [],
            f"{arm}: run metadata exit/source-change status is not clean")
    require(metadata.get("argv") == invocation.get("argv") ==
            execution.get("observed_native_argv"),
            f"{arm}: invocation/metadata/execution argv disagree")
    require(metadata.get("environment") == invocation.get("environment"),
            f"{arm}: invocation and run-metadata environments disagree")
    env = metadata.get("environment", {})
    require(env.get("NUMI_HUMAN_ACCEPTED_FORCE_AUDIT") == expected_force,
            f"{arm}: force-audit invocation flag is not {expected_force}")
    require(env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT") == "1" and
            env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP") == "464" and
            env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP") == "512",
            f"{arm}: Q audit invocation bounds differ")
    start_argv = start.get("argv", [])
    require(f"NUMI_HUMAN_ACCEPTED_FORCE_AUDIT={expected_force}" in start_argv,
            f"{arm}: start receipt does not record force-audit flag {expected_force}")
    require(f"NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP=464" in start_argv and
            f"NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP=512" in start_argv,
            f"{arm}: start receipt does not record exact Q bounds")
    checks[arm + "_run"] = {
        "exit_code": metadata.get("exit_code"),
        "wall_seconds": metadata.get("wall_seconds"),
        "changed_inputs": execution.get("changed_inputs"),
        "source_files_changed_during_run": metadata.get("source_files_changed_during_run"),
        "force_audit_flag": expected_force,
        "q_window": [464, 512],
        "run_metadata_asset_pin_count": len(observed_assets),
        "run_metadata_asset_pins_match_declaration": asset_subset_matches,
        "all_declared_immutable_asset_files_match": asset_files_match,
        "declared_immutable_asset_count": len(declared_assets),
    }
    return root, run, decl, execution, invocation, metadata, start

def normalize_arm_path(value: str) -> str:
    return value.replace("/control/", "/ARM/").replace("/window/", "/ARM/")

def parse_vector(text: str, label: str):
    values = text.split(";")
    require(len(values) == 128, f"{label}: expected 128 semicolon-separated values, got {len(values)}")
    numbers = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except Exception:
            failures.append(f"{label}: nonnumeric value at index {index}")
            continue
        if not math.isfinite(number):
            failures.append(f"{label}: nonfinite value at index {index}")
        numbers.append(number)
    return numbers

def main() -> int:
    require(not OUT.exists(), f"refusing to overwrite existing output: {OUT}")
    require(BASE.is_dir(), f"paired smoke directory missing: {BASE}")
    if BASE.is_dir():
        pin(Path(__file__))
    try:
        compare_csv_fn = import_function(COMPARE_PATH, "accepted_force_pair_csv_compare", "compare_csv")
        sys.path.insert(0, str(HUMAN_SRC))
        from numilab_human.torso_anatomy_audit import _pack_sections as pack_sections_fn
        from numilab_human.resting_scene import load_rigid as load_rigid_fn
    except Exception as exc:
        failures.append(f"owner parser import failed: {type(exc).__name__}: {exc}")
        compare_csv_fn = None
        pack_sections_fn = None
        load_rigid_fn = None
    if compare_csv_fn is not None and pack_sections_fn is not None and load_rigid_fn is not None:
        for p in (COMPARE_PATH, PACK_OWNER, RIGID_OWNER):
            pin(p)
    arms = {}
    for arm in ("control", "window"):
        try:
            arms[arm] = load_arm(arm)
        except Exception as exc:
            failures.append(f"{arm}: receipt load failed: {type(exc).__name__}: {exc}")
    if len(arms) != 2:
        report = {"schema": "numi.human.accepted-force-paired-smoke-verification.v1",
                  "status": "failed", "failures": failures, "checks": checks,
                  "input_sha256": input_hashes}
        with OUT.open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False); f.write("\n")
        print(json.dumps({"status": "failed", "failures": failures, "output": str(OUT)}, indent=2))
        return 1

    croot, crun, cdecl, cexec, cinv, cmeta, cstart = arms["control"]
    wroot, wrun, wdecl, wexec, winv, wmeta, wstart = arms["window"]

    # The prepared immutable source set must be the same in both arms.
    cpins, wpins = cdecl.get("immutable_assets", {}), wdecl.get("immutable_assets", {})
    require(cpins == wpins, "control/window declared immutable asset maps differ")
    checks["declared_immutable_assets"] = {
        "control_count": len(cpins), "window_count": len(wpins), "maps_equal": cpins == wpins
    }
    # Normalize only expected arm-output locations when comparing native invocations.
    cargv = [normalize_arm_path(str(x)) for x in cmeta.get("argv", [])]
    wargv = [normalize_arm_path(str(x)) for x in wmeta.get("argv", [])]
    require(cargv == wargv, "native argv differ beyond control/window output paths")
    cenv = {k: normalize_arm_path(str(v)) for k, v in cmeta.get("environment", {}).items()}
    wenv = {k: normalize_arm_path(str(v)) for k, v in wmeta.get("environment", {}).items()}
    env_keys = sorted(k for k in set(cenv) | set(wenv) if cenv.get(k) != wenv.get(k))
    require(env_keys == ["NUMI_HUMAN_ACCEPTED_FORCE_AUDIT"],
            f"environment differs beyond force-observer toggle: {env_keys}")
    checks["arm_difference_scope"] = {
        "normalized_native_argv_equal": cargv == wargv,
        "environment_difference_keys": env_keys,
        "only_expected_force_observer_toggle": env_keys == ["NUMI_HUMAN_ACCEPTED_FORCE_AUDIT"],
        "same_declared_immutable_assets": cpins == wpins,
    }

    csv_reports = {}
    for name in COMMON_CSV:
        cp, wp = crun / name, wrun / name
        require(cp.is_file() and wp.is_file(), f"common CSV missing: {name}")
        if cp.is_file() and wp.is_file():
            hc, hw = pin(cp), pin(wp)
            parsed = compare_csv_fn(cp, wp)
            exact_bytes = hc == hw
            require(exact_bytes and parsed.get("all_cells_equal") is True,
                    f"common CSV differs between arms: {name}")
            csv_reports[name] = {
                "control_sha256": hc, "window_sha256": hw,
                "raw_bytes_equal": exact_bytes, "parsed_cells_equal": parsed.get("all_cells_equal"),
                "rows": parsed.get("control_rows"),
            }

    q_name = "resting-com-q-integration.csv"
    q_header, qrows = read_csv(wrun / q_name)
    cq_header, cqrows = read_csv(crun / q_name)
    qsteps = [int(row["accepted_step"]) for row in qrows]
    require(qsteps == EXPECTED_Q_STEPS, f"window Q rows do not exactly cover steps 464..512: {qsteps[:3]}..{qsteps[-3:]}")
    require([int(row["accepted_step"]) for row in cqrows] == EXPECTED_Q_STEPS,
            "control Q rows do not exactly cover steps 464..512")
    require(len(qrows) == 49 and len(cqrows) == 49, "Q audit row count is not 49 in both arms")
    index_header, index_rows = read_csv(wrun / "resting-com-q-index-map.csv")
    velocity_rows = [r for r in index_rows if r.get("local_v_index", "") != ""]
    local_v = [int(r["local_v_index"]) for r in velocity_rows]
    require(sorted(local_v) == list(range(128)) and len(local_v) == len(set(local_v)),
            "Q index map does not have exactly one row for each local_v index 0..127")
    rigid_manifest = json.loads(MODEL_MANIFEST.read_text())
    rigid = load_rigid_fn(RIGID_PATH) if load_rigid_fn else {}
    myo_file = RIGID_PATH.parent / rigid_manifest["payloads"]["muscles"]["file"]
    myo_sha = pin(myo_file)
    myo_declared_sha = rigid_manifest["payloads"]["muscles"]["sha256"]
    require(myo_sha == myo_declared_sha, "NHMYO payload hash differs from fullbody manifest")
    raw_myo = myo_file.read_bytes()
    if len(raw_myo) >= struct.calcsize("<8s9I32s"):
        magic, abi, body_count, muscle_count = struct.unpack_from("<8s3I", raw_myo)
    else:
        magic, abi, body_count, muscle_count = b"", 0, 0, 0
    tendon_manifest = json.loads(TENDON_MANIFEST.read_text())
    tendon_sha = pin(TENDON_PATH)
    require(tendon_sha == tendon_manifest.get("payload", {}).get("sha256"),
            "NHTENDON payload hash differs from tendon manifest")
    model_muscles = int(rigid_manifest["model"]["source_muscle_count"])
    tendon_muscles = int(tendon_manifest["coverage"]["muscle_count"])
    tendon_bindings = int(tendon_manifest["coverage"]["mechanical_endpoint_count"])
    nv = int(rigid.get("nv", -1)); nq = int(rigid.get("nq", -1))
    require((nv, nq) == (128, 129), f"current NHRIGID dimensions differ: nq={nq}, nv={nv}")
    require(muscle_count == model_muscles == 416 and tendon_muscles == 416 and tendon_bindings == 832,
            f"current model counts differ: NHMYO={muscle_count}, manifest muscles={model_muscles}, "
            f"tendon muscles={tendon_muscles}, tendon endpoints={tendon_bindings}")
    force_path = wrun / FORCE_CSV
    control_force_path = crun / FORCE_CSV
    require(force_path.is_file(), "window force observer CSV is missing")
    require(not control_force_path.exists(), "control unexpectedly emitted force observer CSV")
    force_rows = []
    force_report = {}
    if force_path.is_file():
        force_header, force_rows = read_csv(force_path)
        force_steps = [int(row["accepted_step"]) for row in force_rows]
        require(force_steps == EXPECTED_Q_STEPS and len(force_rows) == 49,
                f"force rows do not exactly cover steps464..512: {force_steps[:3]}..{force_steps[-3:]}")
        expected_vector_fields = [
            "mujoco_muscle_generalized_force_row_sum_cpu_f64_by_local_v_semicolon",
            "tendon_transfer_generalized_correction_sum_cpu_f64_by_local_v_semicolon",
            "post_consumer_mujoco_force_workspace_slice_f32_by_local_v_semicolon",
        ]
        finite_rows = 0
        fingerprint_matches = 0
        time_matches = 0
        vector_extrema = {field: {"min": None, "max": None} for field in expected_vector_fields}
        q_by_step = {int(row["accepted_step"]): row for row in qrows}
        for row in force_rows:
            step = int(row["accepted_step"])
            require(int(row["local_v_count"]) == nv, f"step{step}: local_v_count does not equal runtime nv")
            require(int(row["muscle_count"]) == model_muscles, f"step{step}: muscle_count differs from payload")
            require(int(row["tendon_binding_count"]) == tendon_bindings,
                    f"step{step}: tendon binding count differs from payload")
            require(row["generalized_effort_unit_convention"] == EXPECTED_UNITS,
                    f"step{step}: unexpected generalized-effort unit convention")
            qrow = q_by_step.get(step)
            require(qrow is not None, f"step{step}: no matching Q integration row")
            if qrow is not None:
                fp_equal = row["source_pre_step_q_fingerprint_fnv64"] == qrow["q_before_fingerprint_fnv64"]
                fingerprint_matches += int(fp_equal)
                require(fp_equal, f"step{step}: force row source q fingerprint does not match Q pre-step q")
                time_equal = row["accepted_time_s"] == qrow["time_s"]
                time_matches += int(time_equal)
                require(time_equal, f"step{step}: force row time differs from accepted Q row time")
            row_ok = True
            for field in expected_vector_fields:
                values = parse_vector(row[field], f"force step{step} {field}")
                row_ok = row_ok and len(values) == 128 and all(math.isfinite(v) for v in values)
                if values:
                    lo, hi = min(values), max(values)
                    old = vector_extrema[field]
                    old["min"] = lo if old["min"] is None else min(old["min"], lo)
                    old["max"] = hi if old["max"] is None else max(old["max"], hi)
            finite_rows += int(row_ok)
        require(finite_rows == 49, f"only {finite_rows}/49 force rows contain finite 128-vectors")
        force_report = {
            "header": force_header,
            "rows": len(force_rows), "step_range": [force_steps[0], force_steps[-1]] if force_steps else None,
            "q_pre_step_fingerprint_matches": fingerprint_matches,
            "accepted_time_string_matches": time_matches,
            "all_vectors_have_128_finite_values": finite_rows == 49,
            "vector_component_min_max": vector_extrema,
            "counts_from_current_payloads": {"nq": nq, "nv": nv, "muscles": model_muscles,
                                             "tendon_bindings": tendon_bindings},
            "unit_convention": EXPECTED_UNITS,
            "semantics": [
                "muscle vector is a CPU float64 row-sum reduction of already-collected per-muscle generalized force rows",
                "tendon vector is a CPU float64 row-sum reduction of already-collected tendon transfer correction rows",
                "workspace vector is the post-consumer generalized-force workspace slice stored as float32",
                "these are distinct observations; none is labeled as total or net generalized force",
            ],
        }
    checks["common_csv_exact"] = csv_reports
    checks["q_and_index_map"] = {
        "window_q_steps": qsteps, "control_q_steps": [int(r["accepted_step"]) for r in cqrows],
        "q_rows_both_49": len(qrows) == len(cqrows) == 49,
        "local_v_indices_complete": sorted(local_v) == list(range(128)),
        "unnamed_joint_labels_preserved": "joint_name" in index_header,
        "runtime_dimensions": {"nq": nq, "nv": nv},
        "current_model_counts": {"muscles": model_muscles, "tendon_muscles": tendon_muscles,
                                 "tendon_bindings": tendon_bindings, "myo_magic": magic.decode("ascii", "replace"),
                                 "myo_abi": abi, "myo_body_count": body_count},
    }
    checks["force_observer"] = {
        "enabled_only_in_window": not control_force_path.exists() and force_path.is_file(),
        "rows": force_report,
    }

    # Compare only MRVPACK geometry/data sections 2-5; section 1 is metadata.
    mrv_reports = {}
    if pack_sections_fn:
        for step in (0, 512):
            rel = Path("accepted-geometry") / f"step-{step}.mrvpack"
            cp, wp = crun / rel, wrun / rel
            require(cp.is_file() and wp.is_file(), f"MRV capture missing at accepted step {step}")
            if not (cp.is_file() and wp.is_file()):
                continue
            csha, wsha = pin(cp), pin(wp)
            csections = pack_sections_fn(cp)
            wsections = pack_sections_fn(wp)
            section_checks = {}
            for kind in (2, 3, 4, 5):
                cv, wv = csections.get(kind), wsections.get(kind)
                same = cv is not None and wv is not None and cv == wv
                require(same, f"MRV step{step} nonmetadata section{kind} differs")
                section_checks[str(kind)] = {
                    "equal": same,
                    "control_bytes": len(cv[0]) if cv else None,
                    "window_bytes": len(wv[0]) if wv else None,
                    "elements": cv[1] if cv else None,
                    "stride": cv[2] if cv else None,
                }
            mrv_reports[str(step)] = {
                "control_pack_sha256": csha, "window_pack_sha256": wsha,
                "full_pack_byte_identity": csha == wsha,
                "nonmetadata_sections_2_to_5_equal": all(v["equal"] for v in section_checks.values()),
                "sections": section_checks,
                "metadata_section_1_equal": csections.get(1) == wsections.get(1),
            }
    checks["mrv_geometry"] = mrv_reports

    # Hash all data used by this check a second time to detect concurrent mutation.
    before_hashes = dict(input_hashes)
    after_hashes = {path: sha(Path(path)) for path in before_hashes}
    unchanged = before_hashes == after_hashes
    require(unchanged, "one or more input files changed during verification")
    report = {
        "schema": "numi.human.accepted-force-paired-smoke-verification.v1",
        "status": "pass" if not failures else "failed",
        "scope": "512-step paired native observer smoke; exact trace/geometry comparison and bounded force-row schema/alignment check only.",
        "input_sha256": before_hashes,
        "input_sha256_after": after_hashes,
        "input_files_unchanged_during_verification": unchanged,
        "checks": checks,
        "limitations": [
            "The run is a short observer smoke, not a long-run physical or anatomy qualification.",
            "Muscle row sums, tendon transfer-correction sums, and post-consumer workspace values are separated observations; they are not a net-force reconstruction.",
            "CSV observer equality and MRVPACK sections 2-5 do not prove internal GPU buffer equivalence beyond the recorded outputs.",
            "DOF indices are pinned by the retained local-v index map; labels remain source-owned and are not inferred from unnamed rows.",
        ],
        "failures": failures,
    }
    with OUT.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "status": report["status"], "failures": failures, "output": str(OUT),
        "output_sha256": sha(OUT), "common_csv_exact": {
            k: v["raw_bytes_equal"] for k, v in csv_reports.items()},
        "force_rows": force_report.get("rows"), "mrv_sections_equal": {
            step: v["nonmetadata_sections_2_to_5_equal"] for step, v in mrv_reports.items()},
        "inputs_unchanged": unchanged,
    }, indent=2))
    return 0 if not failures else 1

if __name__ == "__main__":
    raise SystemExit(main())
