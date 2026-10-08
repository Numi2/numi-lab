#!/usr/bin/env python3
"""Record a direct-native, 20 s treatment-program identity probe.

This recorder does not use the public Human intervention builder because that
builder correctly bounds an intervention to the physical run duration. The
native parser admits a future [60,100) s interval in a 20 s run; this records
only that the compiled native program includes the treatment configuration.
It is not a dose-response or physiological treatment observation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = Path("/Users/n/numi-human-resting-evidence-20261005")
HUMAN_SRC = Path("/Users/n/numi-human-lung-triangulation-candidate-003/src")
SOURCE_BASE_HASHES = EVIDENCE / "native-delivery-provenance-865/source-hashes.json"
SOURCE_BASE_REVISIONS = EVIDENCE / "native-delivery-provenance-865/source-revisions.json"
PROGRAM_PROBE_NAME = "treatment-program-probe"
EXPECTED_STEPS = 10000
EXPECTED_DT = 0.002
INTERVENTION = (60.0, 100.0, 0.5)
PROBE_SCOPE = "direct native 20 s program-identity probe; configured future intervention is outside the simulated interval; no treatment dose or physiological response is observed"

sys.path.insert(0, str(HUMAN_SRC))
from numilab_human import resting_run


def load_readiness():
    path = ROOT / "prepare_final_plan.py"
    spec = importlib.util.spec_from_file_location("readiness_prepare_final_plan", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return resting_run._hash(Path(path))


def make_probe(control_invocation, scene_dir, output_dir, readiness):
    """Apply the registered readiness transform and fail closed on any drift."""
    scene_dir = Path(scene_dir).resolve()
    output_dir = Path(output_dir).resolve()
    require(output_dir.parent == scene_dir and output_dir.name == PROGRAM_PROBE_NAME,
            "probe output must be the prescribed child of the corrected control preflight")
    argv, environment = readiness.normalize_for_probe(
        control_invocation, treatment=True, output_dir=output_dir)
    require(len(argv) > 4 and Path(argv[4]).resolve() == output_dir,
            "normalized native argv does not target the exact probe directory")
    require(argv[0] == str(readiness.NATIVE_BINARY),
            "normalized native argv does not use the pinned terminal-capture-017 executable")
    require("--resting-drive-intervention" in argv,
            "normalized native argv lacks the treatment program")
    index = argv.index("--resting-drive-intervention")
    try:
        values = tuple(float(argv[index + offset]) for offset in (1, 2, 3))
    except (ValueError, IndexError):
        raise ValueError("treatment program must contain start, end, and scale")
    require(values == INTERVENTION and index + 4 == len(argv),
            "probe must append only [60,100,0.5] to the native argv")
    require("--resting-movie" in argv and
            Path(argv[argv.index("--resting-movie") + 1]).resolve() ==
            (output_dir / "native-viewer.mov").resolve(),
            "native viewer output is not retained at the analyzer's expected path")
    require(environment.get("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT") ==
            str(output_dir / "common-field-failure.json"),
            "common-field failure receipt does not target the probe directory")
    return argv, environment


def make_receipt(control_invocation, argv, environment, control_refs, identity_ids,
                 recorder_sha, parser_sha):
    receipt = {
        "argv": list(argv),
        "asset_sha256": dict(control_invocation["asset_sha256"]),
        "environment": dict(environment),
        "host": control_invocation["host"],
        "machine": control_invocation["machine"],
        "system": control_invocation["system"],
        "qualification": control_invocation["qualification"],
        "direct_native_probe": {
            "scope": PROBE_SCOPE,
            "configured_intervention": {
                "start_seconds": INTERVENTION[0],
                "end_seconds": INTERVENTION[1],
                "drive_scale": INTERVENTION[2],
            },
            "simulated_horizon_seconds": EXPECTED_STEPS * EXPECTED_DT,
            "intervention_overlaps_simulated_horizon": False,
            "dose_observed": False,
            "native_working_directory": str(control_refs["working_directory"]),
            "predicted_body_source_fingerprint": identity_ids[
                "treatment_body_source_fingerprint_predicted"],
            "predicted_coupled_program_fingerprint": identity_ids[
                "treatment_program_fingerprint_predicted"],
            "identity_algorithm_source": str(control_refs["identity_source"]),
            "identity_algorithm_source_sha256": parser_sha,
            "recorder_sha256": recorder_sha,
            "control_invocation_sha256": control_refs["invocation_sha256"],
            "control_run_metadata_sha256": control_refs["metadata_sha256"],
            "control_native_log_sha256": control_refs["log_sha256"],
            "tracked_environment_keys": sorted(environment),
        },
    }
    return receipt



def verify_future_interval_parser(readiness):
    """Confirm the pinned native parser validates ordering/range, not run duration."""
    source = readiness.PROBE_SOURCE.read_text(encoding="utf-8")
    start = source.find('if(argument=="--resting-drive-intervention")')
    require(start >= 0, "pinned native source has no drive-intervention parser")
    block = source[start:source.find("continue;", start) + len("continue;")]
    require("values[0]>=0&&values[1]>values[0]&&values[2]>=0&&values[2]<=2" in block,
            "native parser's admitted treatment start/end/scale predicate changed")
    require("muscleStepCount" not in block and "muscleStepSeconds" not in block and
            "args.seconds" not in block,
            "native parser now binds intervention end to physical run duration")
    require(INTERVENTION[0] >= 0 and INTERVENTION[1] > INTERVENTION[0] and
            0 <= INTERVENTION[2] <= 2,
            "configured treatment tuple violates the parser's start/end/scale predicate")
    return {
        "source": str(readiness.PROBE_SOURCE),
        "sha256": readiness.sha(readiness.PROBE_SOURCE),
        "source_predicate": "start >= 0; end > start; 0 <= scale <= 2",
        "no_run_duration_bound_in_parser": True,
    }

def tracked_source_snapshot(source_hashes, readiness):
    snapshot = {}
    for path, expected in sorted(source_hashes.items()):
        target = Path(path)
        require(target.is_file() and not target.is_symlink(),
                "pinned source/build input is missing or symlinked: " + path)
        actual = readiness.sha(target)
        require(actual == expected, "pinned source/build input changed before probe: " + path)
        snapshot[path] = actual
    return snapshot


def changed_hashes(before, after):
    return [path for path in sorted(set(before) | set(after))
            if before.get(path) != after.get(path)]


def validate_identity_result(log_path, control_summary, identity_ids, steps, dt,
                             readiness):
    log = Path(log_path).read_text(encoding="utf-8", errors="replace")
    summary = readiness.owner.native_scene_summary(log)
    terminal_line = next(
        (line for line in reversed(log.splitlines())
         if line.startswith("stand_terminal_state=")), None)
    require(terminal_line is not None,
            "native log lacks a terminal accepted-state record")
    terminal = json.loads(terminal_line.split("=", 1)[1])
    require(steps == EXPECTED_STEPS and dt == EXPECTED_DT and
            summary.get("accepted_steps") == EXPECTED_STEPS,
            "native probe did not accept exactly 10,000 roots at 2 ms")
    require(abs(summary.get("simulated_s", -1.0) - 20.0) <= 1e-5,
            "native probe terminal horizon is not 20 seconds")
    require(terminal.get("timestep_seconds") == EXPECTED_DT and
            terminal.get("root_assistance") is False,
            "native probe terminal timestep/assistance evidence is invalid")
    require(summary.get("world_fingerprint") == control_summary.get("world_fingerprint") and
            summary.get("device") == control_summary.get("device"),
            "native probe world/device differs from the matched control")
    require(summary.get("body_source_fingerprint") == identity_ids[
                "treatment_body_source_fingerprint_predicted"] and
            summary.get("coupled_program_fingerprint") == identity_ids[
                "treatment_program_fingerprint_predicted"],
            "native treatment identity differs from the independent readiness prediction")
    return {
        "accepted_steps": summary["accepted_steps"],
        "simulated_s": summary["simulated_s"],
        "device": summary["device"],
        "world_fingerprint": summary["world_fingerprint"],
        "body_source_fingerprint": summary["body_source_fingerprint"],
        "coupled_program_fingerprint": summary["coupled_program_fingerprint"],
        "terminal_timestep_seconds": terminal["timestep_seconds"],
        "root_assistance": terminal["root_assistance"],
        "dose_observed": False,
        "scope": PROBE_SCOPE,
    }


def make_native_environment(inherited, tracked_environment, owner_module=resting_run):
    """Use the study driver's prefix sanitation, then apply recorded native controls."""
    env=dict(inherited)
    for key in list(env):
        if key.startswith(("NUMI_", "DYLD_")) or key in owner_module.INVOCATION_ENVIRONMENT_KEYS:
            env.pop(key,None)
    env.update(tracked_environment)
    require(owner_module.invocation_environment(env)==tracked_environment,
            "effective tracked native environment differs from the exact control-derived plan")
    return env


def recorder_exit_status(native_exit_code, validation_error, changed_sources,
                         changed_assets, runtime_verified):
    """Keep a zero native exit distinct from a failed recorder acceptance gate."""
    if native_exit_code not in (0, None):
        return int(native_exit_code)
    if validation_error or changed_sources or changed_assets or not runtime_verified:
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-preflight-dir", required=True,
                        help="completed corrected control preflight directory")
    parser.add_argument("--anatomy-receipt", required=True,
                        help="exact corrected anatomy receipt already bound by control")
    parser.add_argument("--execute", action="store_true",
                        help="launch the exact direct-native 20 s identity probe; default is plan-only")
    args = parser.parse_args(argv)

    scene_arg = Path(args.scene_preflight_dir).expanduser()
    require(not scene_arg.is_symlink(), "control preflight directory must not be a symlink")
    scene_dir = scene_arg.resolve()
    require(scene_dir.is_dir(), "control preflight directory is missing")
    receipt_path = Path(args.anatomy_receipt).expanduser()
    require(not receipt_path.is_symlink() and receipt_path.is_file(),
            "corrected anatomy receipt must be an existing regular file")
    receipt_path = receipt_path.resolve()

    readiness = load_readiness()
    readiness.verify_build_and_pins()
    control_invocation, control_metadata, control_log, control_summary, steps, dt = (
        readiness.scene_summary(scene_dir / "invocation.json", scene_dir, receipt_path))
    identity_ids = readiness.program_ids(control_log, readiness.RESP_METALLIB)
    parser_evidence = verify_future_interval_parser(readiness)
    output_dir = scene_dir / PROGRAM_PROBE_NAME
    require(not output_dir.exists() and not output_dir.is_symlink(),
            "probe output already exists; retain it and choose a fresh corrected control preflight")
    native_argv, native_environment = make_probe(
        control_invocation, scene_dir, output_dir, readiness)

    # This is the full source/build evidence input inventory used by readiness,
    # checked before and after native execution; the separate asset table is
    # copied from the exact owner invocation and independently rehashed.
    source_hashes, source_revisions = readiness.source_pin_files(
        json.loads(readiness.BUILD_MANIFEST.read_text(encoding="utf-8")))
    source_before = tracked_source_snapshot(source_hashes, readiness)
    assets_before = {}
    for path, expected in sorted(control_invocation["asset_sha256"].items()):
        target = Path(path)
        require(target.is_file() and not target.is_symlink(),
                "bound native input is missing or symlinked: " + path)
        actual = resting_run._hash(target)
        require(actual == expected, "bound native input changed before probe: " + path)
        assets_before[path] = actual

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("direct native probe requires Apple silicon macOS")
    require(platform.machine() == control_invocation["machine"] and
            platform.node() == control_invocation["host"] and
            platform.platform() == control_invocation["system"],
            "current host identity differs from corrected control")

    control_refs = {
        "identity_source": readiness.PROBE_SOURCE,
        "invocation_sha256": readiness.sha(scene_dir / "invocation.json"),
        "metadata_sha256": readiness.sha(scene_dir / "run-metadata.json"),
        "log_sha256": readiness.sha(scene_dir / "native.log"),
        "working_directory": str(scene_dir),
    }
    recorder_sha = readiness.sha(Path(__file__))
    parser_sha = readiness.sha(readiness.PROBE_SOURCE)
    invocation = make_receipt(control_invocation, native_argv, native_environment,
                              control_refs, identity_ids, recorder_sha, parser_sha)
    plan = {
        "status": "planned_not_run",
        "probe_directory": str(output_dir),
        "working_directory": str(scene_dir),
        "native_argv": native_argv,
        "native_environment": native_environment,
        "bound_asset_count": len(assets_before),
        "pinned_source_build_input_count": len(source_before),
        "expected_steps": EXPECTED_STEPS,
        "expected_dt_s": EXPECTED_DT,
        "expected_treatment_source_fingerprint":
            identity_ids["treatment_body_source_fingerprint_predicted"],
        "expected_treatment_program_fingerprint":
            identity_ids["treatment_program_fingerprint_predicted"],
        "scope": PROBE_SCOPE,
        "native_parser_validation": parser_evidence,
        "expected_files": [
            str(output_dir / "invocation.json"),
            str(output_dir / "run-metadata.json"),
            str(output_dir / "native.log"),
            str(output_dir / "source-hashes-before.json"),
            str(output_dir / "source-hashes-after.json"),
        ],
        "direct_native_invocation": invocation,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("Plan only: no directory, receipt, or native process was created.")
        return 0

    require(steps == EXPECTED_STEPS and dt == EXPECTED_DT,
            "control preflight is not the exact 20 s identity-probe template")
    require(control_metadata.get("loaded_metal_runtime", {}).get("verified") is True,
            "control runtime is not verified")
    require(platform.machine() == "arm64", "native probe host is not arm64")
    output_dir.mkdir(parents=True)
    (output_dir / "invocation.json").write_text(
        json.dumps(invocation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "source-hashes-before.json").write_text(
        json.dumps(source_before, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "source-revisions.json").write_text(
        json.dumps(source_revisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = make_native_environment(os.environ, native_environment)

    start = time.monotonic()
    native_exit_code = None
    launch_error = None
    try:
        with (output_dir / "native.log").open("w", encoding="utf-8") as log:
            native = subprocess.run(native_argv, cwd=str(scene_dir), env=env,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
            native_exit_code = native.returncode
    except Exception as exc:
        launch_error = type(exc).__name__ + ": " + str(exc)
        (output_dir / "native.log").write_text(
            "direct_native_launch_error=" + launch_error + "\n", encoding="utf-8")
    wall_seconds = time.monotonic() - start

    source_after = {}
    for path in source_before:
        target = Path(path)
        source_after[path] = readiness.sha(target) if target.is_file() and not target.is_symlink() else None
    assets_after = {}
    for path in assets_before:
        target = Path(path)
        assets_after[path] = resting_run._hash(target) if target.is_file() and not target.is_symlink() else None
    source_changes = changed_hashes(source_before, source_after)
    asset_changes = changed_hashes(assets_before, assets_after)
    (output_dir / "source-hashes-after.json").write_text(
        json.dumps(source_after, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_path = readiness.LIBMETALROBO.resolve()
    runtime_hash = assets_before.get(str(runtime_path))
    runtime = resting_run.loaded_metal_runtime(
        output_dir / "native.log", runtime_path, runtime_hash or "")
    validation = None
    validation_error = launch_error
    if native_exit_code == 0 and validation_error is None:
        try:
            validation = validate_identity_result(
                output_dir / "native.log", control_summary, identity_ids, steps, dt, readiness)
        except Exception as exc:
            validation_error = type(exc).__name__ + ": " + str(exc)

    metadata = dict(invocation)
    metadata.update({
        "exit_code": native_exit_code,
        "wall_seconds": wall_seconds,
        "source_files_changed_during_run": source_changes,
        "asset_files_changed_during_run": asset_changes,
        "loaded_metal_runtime": runtime,
        "direct_native_probe_validation": {
            "passed": validation_error is None and native_exit_code == 0 and
                      not source_changes and not asset_changes and runtime["verified"],
            "error": validation_error,
            "observed": validation,
            "scope": PROBE_SCOPE,
        },
        "source_hashes_before_path": str(output_dir / "source-hashes-before.json"),
        "source_hashes_before_sha256": readiness.sha(output_dir / "source-hashes-before.json"),
        "source_hashes_after_path": str(output_dir / "source-hashes-after.json"),
        "source_hashes_after_sha256": readiness.sha(output_dir / "source-hashes-after.json"),
        "source_revisions_path": str(output_dir / "source-revisions.json"),
        "source_revisions_sha256": readiness.sha(output_dir / "source-revisions.json"),
    })
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = recorder_exit_status(native_exit_code, validation_error, source_changes,
                                  asset_changes, runtime["verified"])
    print(json.dumps({
        "exit_code": native_exit_code,
        "recorder_exit_code": status,
        "probe_directory": str(output_dir),
        "runtime_verified": runtime["verified"],
        "changed_sources": source_changes,
        "changed_assets": asset_changes,
        "identity_probe_passed": metadata["direct_native_probe_validation"]["passed"],
        "identity_probe_scope": PROBE_SCOPE,
        "validation_error": validation_error,
    }, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
