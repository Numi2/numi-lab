#!/usr/bin/env python3
"""Prepare a fail-closed native Human 310 s science-v2 plan; never run native code."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
from pathlib import Path

E = Path("/Users/n/numi-human-resting-evidence-20261005")
LAB = Path("/Users/n/numi-human-performance-source-014")
HUMAN = Path("/Users/n/numi-human-lung-triangulation-candidate-003")
HUMAN_RESTING_RUN = HUMAN / "src/numilab_human/resting_run.py"
HUMAN_RESTING_RUN_SHA = "f6bc12635fcf41227a79b4d67e058b36f392a4147d083ce24366f8c3ef41c412"
TERMINAL_931_NATIVE_LOG = E / "native-terminal-cycle-931/native.log"
TERMINAL_931_NATIVE_LOG_SHA = "17319decc6d30e29db54e79a43c99d254ce52598512883c983eaaa51ca26c999"
BRAIN = Path("/Users/n/numi-human-resting-integration-20261005/numi-brain")
OWNER = LAB / "matter/tools/resting_intervention_study.py"
SCIENCE = LAB / "tools/numi"
PY39 = Path("/Applications/Xcode-26.6.0.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9")
BASE_HASHES = E / "native-delivery-provenance-865/source-hashes.json"
BASE_REVISIONS = E / "native-delivery-provenance-865/source-revisions.json"
BUILD = Path("/Users/n/numi-human-terminal-capture-build-017")
BUILD_MANIFEST = BUILD / "evidence/build-pins.json"
BUILD_SOURCE_PINS = BUILD / "evidence/source-pins.json"
BUILD_FOCUSED_TESTS = BUILD / "evidence/focused-tests.json"
BUILD_MANIFEST_SHA = "ed576aba3b330574de7d9afa83474f8b33f51df48770532862e56e0c69e69984"
BUILD_SOURCE_PINS_SHA = "ccbfa7e27f51b8635d0e075945cb672084c4793b0153df33c9093747c9c919ba"
BUILD_FOCUSED_TESTS_SHA = "0ff18d138d5851d333944ed086d8ce0bd6e23a3fa6edd7f130da44bb63b326dc"
BUILD_SOURCE = Path("/Users/n/numi-human-terminal-accepted-state-017")
BUILD_COMPILED_HEAD = "b091d7dcead509a325194563ed38261319118a88"
BUILD_EVIDENCE_HEAD = "efde8e704e55a3a6eb1306b080ee34f198e3578c"
BUILD_PATCH = BUILD / "evidence/build-native-viewer.sh"
BUILD_PATCH_SHA = "2275aadd362378244042c4f9fd1b4faf31849bbd0f0a714c701d04bf837d9d62"
NATIVE_BINARY = BUILD / "bin/numi-human-native"
EXPECTED_BINARY_SHA = "11733b5d10f3354df54416d1941281ad4feeb73c2b3e5baee8ebd1be22518ba2"
RESP_METALLIB = BUILD / "matter/shaders/HumanRespiration.metallib"
RESP_METALLIB_SHA = "4b61361f513bf0996d687398498b85ba4e379edca91c36f134e1b0398f31c426"
BUILD_LIBMETALROBO = BUILD / "lib/libmetalrobo.dylib"
LIBMETALROBO = Path("/Users/n/numi-human-performance-build-014/lib/libmetalrobo.dylib")
LIBMETALROBO_SHA = "6bccfc4044d825423e66bc2f60ba3cf59eaa9a4936773ab08182f58a09927092"
RUNTIME_SOURCE = BUILD_SOURCE / "matter/tools/human_resting_runtime.hpp"
PROBE_SOURCE = BUILD_SOURCE / "apps/numilab_human_myosim_visual_probe.mm"
SCENE_MARKER = "__PENDING_CORRECTED_SCENE_PREFLIGHT_DIR__"
RECEIPT_MARKER = "__PENDING_CORRECTED_ANATOMY_RECEIPT_PATH__"
PROGRAM_PROBE_NAME = "treatment-program-probe"
CARDIAC_911 = E / "native-cardiac-interface-localization-911/final-localization-report.json"
CARDIAC_911_SHA = "9c468f7a1da58563387acc9712f845a794f07fa3121c7fa4a3a7ae905ecf03e4"
TERMINAL_Q0_COM8_932 = E / "native-terminal-production-review-932/verification.json"
TERMINAL_Q0_COM8_932_SHA = "453e8f69bdad311d1d273d5049d17b886b609780bfbe3614700e82c424e782f9"
READINESS_ROOT = Path(__file__).resolve().parent
READINESS_SCRIPT = READINESS_ROOT / "prepare_final_plan.py"
READINESS_ANALYZER = READINESS_ROOT / "analyze_final_pair.py"
READINESS_README = READINESS_ROOT / "README.md"
READINESS_REVISION = READINESS_ROOT / "revision-005.json"
CAPTURE_PLAN_TESTS = READINESS_ROOT / "test_terminal_capture_plan.py"
OWNER_LINEAGE_TESTS = READINESS_ROOT / "test_owner_invocation_lineage.py"
CAPTURE_PLAN_FILE = "accepted-geometry-capture-plan.json"
CAPTURE_TEMPLATE_SCHEMA = "numi.human.resting.accepted-geometry-launch-template.v1"
CAPTURE_PLAN_SCHEMA = "numi.human.resting.accepted-geometry-capture-plan.v1"
CAPTURE_ENV_KEY = "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"
FINAL_ACCEPTED_STEPS = 155000
FINAL_DT_S = 0.002
NATIVE_FLOAT_DT_S = struct.unpack("<f", struct.pack("<f", FINAL_DT_S))[0]
PRESENTATION_CADENCE_STEPS = 32
SHARED_CAPTURE_STEPS = (47519, 49151, 51903, 54047, 55647)
CONTROL_LATE_CAPTURE_STEPS = (152191, 154143)
TREATMENT_LATE_CAPTURE_STEPS = (152447, 154367)
TREATMENT_PROBE_RECORDER = READINESS_ROOT / "record_treatment_program_probe.py"
TREATMENT_PROBE_TESTS = READINESS_ROOT / "test_record_treatment_program_probe.py"
SEGMENT8 = E / "integrated-parallel-contact-batched-752/execution.json"
FULL_Q = E / "integrated-final-runtime-2ms-check-801/execution.json"
RUNTIME_REFERENCE = LAB / "docs/evidence/human-resting/2026-10-07-native-runtime.json"
PARSER_FIXTURE = E / "dense45-pair-004/fgmres-6s.csv"
STUDY = E / "native-integrated-resting-study-917"
FROZEN_LAB_REV = "d550d8ad88a76fdee5bb6e028286fd24e962571f"
FROZEN_HUMAN_REV = "b354949c258106d00f2bd3dd6ac91216d5a3d409"
FROZEN_BRAIN_REV = "a1cf7218fae5d26f9aed9d845047fc6c472ad596"
STEP_COUNT = 10000
DT = 0.002
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1

spec = importlib.util.spec_from_file_location("resting_intervention_study", str(OWNER))
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def accepted_geometry_capture_schedule():
    """Frozen, prior-867-informed plan for seven interior frames plus true N."""
    nominal_duration_s = FINAL_ACCEPTED_STEPS * FINAL_DT_S
    native_terminal_time_s = FINAL_ACCEPTED_STEPS * NATIVE_FLOAT_DT_S
    require(nominal_duration_s == 310.0,
            "final accepted-step horizon is not nominally 310 seconds")
    arms = {}
    for arm, late in (("control", CONTROL_LATE_CAPTURE_STEPS),
                      ("treatment", TREATMENT_LATE_CAPTURE_STEPS)):
        steps = tuple(sorted((*SHARED_CAPTURE_STEPS, *late, FINAL_ACCEPTED_STEPS)))
        require(len(steps) == 8 and len(set(steps)) == 8,
                arm + " capture plan must contain exactly eight unique frames")
        require(steps[-1] == FINAL_ACCEPTED_STEPS and FINAL_ACCEPTED_STEPS - 1 not in steps,
                arm + " capture plan must use true terminal N, never N-1 as terminal")
        for step in steps[:-1]:
            require(0 < step < FINAL_ACCEPTED_STEPS and
                    (step + 1) % PRESENTATION_CADENCE_STEPS == 0,
                    arm + " interior capture is not an ordinary submission endpoint: " + str(step))
        arms[arm] = {
            "step_ids": list(steps),
            "environment_value": ",".join(str(step) for step in steps),
            "events": [{"accepted_step_id": step,
                        "accepted_time_s": step * NATIVE_FLOAT_DT_S,
                        "nominal_time_s": step * FINAL_DT_S,
                        "terminal": step == FINAL_ACCEPTED_STEPS}
                       for step in steps],
        }
    return {
        "schema": CAPTURE_PLAN_SCHEMA,
        "accepted_horizon_steps": FINAL_ACCEPTED_STEPS,
        "physical_timestep_s": FINAL_DT_S,
        "accepted_duration_s": nominal_duration_s,
        "nominal_duration_s": nominal_duration_s,
        "native_float_timestep_s": NATIVE_FLOAT_DT_S,
        "native_float_terminal_time_s": native_terminal_time_s,
        "ordinary_timestamp_rule": "accepted_time_s = accepted_step_id * the native runtime Float32 representation of the requested 0.002 s timestep; nominal_time_s = accepted_step_id * 0.002 s",
        "terminal": {"accepted_step_id": FINAL_ACCEPTED_STEPS,
                     "accepted_time_s": native_terminal_time_s,
                     "nominal_time_s": nominal_duration_s,
                     "n_minus_one_is_not_terminal": True},
        "submission_cadence_steps": PRESENTATION_CADENCE_STEPS,
        "selection_basis": "Prior-867-derived exploratory geometry coverage: five shared phase IDs plus two arm-specific late-cycle IDs and the exact terminal. This is not fitted to the final pair.",
        "coverage_limits": [
            "No initial frame is requested; the corrected 20-second native preflight separately covers initialization and an early complete cycle.",
            "The eight-frame owner cap leaves the old-867 global rib-volume minimum near 65.168 seconds unsampled.",
            "These sparse captures are phase-selected geometry checks, not complete-state or whole-cycle anatomy proof."
        ],
        "arms": arms,
    }


def validate_capture_receipt(schedule, arm, receipt):
    """Validate one native accepted-geometry receipt against its declared ID/time."""
    require(arm in schedule["arms"], "capture receipt arm is not in the schedule")
    step = receipt.get("accepted_step")
    require(type(step) is int and step in schedule["arms"][arm]["step_ids"],
            "geometry receipt has an unrequested or noninteger accepted step")
    expected_time = step * NATIVE_FLOAT_DT_S
    try:
        observed_time = float(receipt.get("accepted_time_s"))
    except (TypeError, ValueError):
        observed_time = float("nan")
    require(math.isfinite(observed_time) and abs(observed_time - expected_time) <= 1e-9,
            "geometry receipt timestamp does not equal accepted_step_id * dt")
    if step == FINAL_ACCEPTED_STEPS:
        require(abs(observed_time - FINAL_ACCEPTED_STEPS * NATIVE_FLOAT_DT_S) <= 1e-9 and
                schedule["terminal"].get("nominal_time_s") == 310.0,
                "terminal geometry receipt must be exact accepted step 155000 at nominal 310 seconds")
    else:
        require(step != FINAL_ACCEPTED_STEPS - 1,
                "N-1 must not be mislabeled as the terminal accepted state")
    require(receipt.get("physical_endpoint") == "accepted" and
            receipt.get("surface_audit_endpoint") == "passed",
            "geometry receipt is not a passing accepted-state surface capture")
    for key in ("accepted_root_fingerprint_hex", "accepted_body_state_sha256",
                "accepted_respiration_state_sha256", "pack_file_sha256"):
        require(isinstance(receipt.get(key), str) and receipt[key],
                "geometry receipt is missing identity field " + key)
    return {"accepted_step": step, "accepted_time_s": observed_time,
            "nominal_time_s": step * FINAL_DT_S,
            "terminal": step == FINAL_ACCEPTED_STEPS}


def validate_parent_capture_selection(parent_invocation):
    """Validate a preflight capture list with the exact compiled 017 cadence rule."""
    environment = parent_invocation.get("environment")
    require(isinstance(environment, dict), "owner preflight environment is missing")
    if CAPTURE_ENV_KEY not in environment:
        return None
    raw = environment[CAPTURE_ENV_KEY]
    require(isinstance(raw, str), "owner preflight capture-step environment must be a string")
    if raw == "":
        return ""
    tokens = raw.split(",")
    require(0 < len(tokens) <= 8 and all(t and t.isascii() and t.isdecimal() for t in tokens),
            "owner preflight capture-step selection must contain at most eight unsigned IDs")
    steps = [int(t) for t in tokens]
    require(len(set(steps)) == len(steps), "owner preflight capture-step selection contains duplicates")
    argv = parent_invocation.get("argv", [])
    require(isinstance(argv, list) and argv.count("--muscle-step-count") == 1 and
            argv.count("--muscle-step-seconds") == 1,
            "owner preflight capture-step validation requires one native horizon and timestep")
    total = int(argv[argv.index("--muscle-step-count") + 1])
    dt = float(argv[argv.index("--muscle-step-seconds") + 1])
    require(total == STEP_COUNT and dt == DT,
            "parent capture IDs must be validated against the completed 10,000-root 20-second preflight")
    for step in steps:
        valid = (0 <= step <= total and
                 (step == 0 or step == total or (step + 1) % PRESENTATION_CADENCE_STEPS == 0))
        require(valid, "owner preflight capture ID is not initial, a 017 submission endpoint, or true terminal: " + str(step))
    return raw


def make_capture_launch_template(parent_invocation, parent_path, parent_sha,
                                 schedule, arm):
    require(arm in schedule["arms"], "unknown launch-template arm")
    environment = parent_invocation.get("environment")
    require(isinstance(environment, dict), "owner preflight environment is missing")
    prior_capture_value = validate_parent_capture_selection(parent_invocation)
    derived = dict(parent_invocation)
    derived["environment"] = dict(environment)
    value = schedule["arms"][arm]["environment_value"]
    derived["environment"][CAPTURE_ENV_KEY] = value
    derived["readiness_derived_capture_launch_template"] = {
        "schema": CAPTURE_TEMPLATE_SCHEMA,
        "scope": "Sets only the declared accepted-geometry export-step value on the verified 20-second owner launch template; the original selection is recorded below, all other argv/environment/assets remain unchanged, and the native run builder supplies the registered 155000-step horizon.",
        "arm": arm,
        "parent_invocation_path": str(Path(parent_path).resolve()),
        "parent_invocation_sha256": parent_sha,
        "capture_environment_key": CAPTURE_ENV_KEY,
        "parent_capture_environment_value": prior_capture_value,
        "capture_environment_value": value,
        "capture_step_ids": schedule["arms"][arm]["step_ids"],
        "accepted_horizon_steps": FINAL_ACCEPTED_STEPS,
        "physical_timestep_s": FINAL_DT_S,
        "terminal_step_id": FINAL_ACCEPTED_STEPS,
        "terminal_nominal_time_s": 310.0,
        "terminal_accepted_time_s": FINAL_ACCEPTED_STEPS * NATIVE_FLOAT_DT_S,
        "identity_claim": "This is a derived launch template, not an owner run receipt; parent owner run-metadata/native.log remain the authority for the completed 20-second preflight."
    }
    return derived


def verify_capture_launch_template(parent_invocation, derived, parent_path,
                                   parent_sha, schedule, arm):
    expected = make_capture_launch_template(parent_invocation, parent_path,
                                            parent_sha, schedule, arm)
    require(derived == expected,
            arm + " derived template changed argv, assets, source identity, or an unapproved environment field")
    return True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path, label):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), label + " must be a regular, non-symlink file: " + str(path))
    return path.resolve()


def read_json(path, label):
    return json.loads(regular(path, label).read_text(encoding="utf-8"))


def wrapped_payload_hash(path, label):
    doc = read_json(path, label)
    payload = doc.get("payload")
    require(isinstance(payload, dict), label + " has no payload object")
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode("utf-8")).hexdigest()
    require(doc.get("sha256") == expected, label + " payload hash mismatch")
    return doc, payload


def git(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True, check=False)
    require(p.returncode == 0, "git command failed in " + str(repo) + ": " + (p.stderr or p.stdout)[-1000:])
    return p.stdout.strip()


def fnv_bytes(seed, data):
    h = seed
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h or FNV_OFFSET


def payload_fingerprint(data):
    return fnv_bytes(FNV_OFFSET, data)


def append_payload_owner(source, domain, payload):
    owner_hash = payload_fingerprint(domain.encode("utf-8"))
    value = (((source ^ owner_hash) * FNV_PRIME) & MASK64) ^ payload
    value = (value * FNV_PRIME) & MASK64
    return value or FNV_OFFSET


def reverse_fnv_bytes(hash_after, data):
    inv_prime = pow(FNV_PRIME, -1, 1 << 64)
    h = hash_after
    for b in reversed(data):
        h = ((h * inv_prime) & MASK64) ^ b
    return h


def program_ids(control_log, shader_path):
    body_match = re.search(r"resting_body_source_fingerprint=(\d+) coupled_program_fingerprint=(\d+)",
                           control_log)
    require(body_match is not None, "control log lacks exact body/program identity")
    control_source, control_program = (int(body_match.group(1)), int(body_match.group(2)))
    shader = regular(shader_path, "loaded respiration metallib").read_bytes()
    # Compiled 017 source folds sourceIdentity and then the exact loaded shader
    # bytes into rootProgramIdentity. Reverse the observed control tail, append
    # the exact [60,100,0.5] source owner, then replay the same tail.
    prefix = reverse_fnv_bytes(control_program, shader)
    base = reverse_fnv_bytes(prefix, struct.pack("<Q", control_source))
    replay = fnv_bytes(fnv_bytes(base, struct.pack("<Q", control_source)), shader)
    require(replay == control_program, "offline FNV reconstruction does not reproduce native control identity")
    intervention_payload = struct.pack("<3d", 60.0, 100.0, 0.5)
    treatment_source = append_payload_owner(control_source, "resting_drive_intervention",
                                            payload_fingerprint(intervention_payload))
    treatment_program = fnv_bytes(fnv_bytes(base, struct.pack("<Q", treatment_source)), shader)
    require(treatment_program not in (0, control_program), "derived treatment identity is invalid")
    return {
        "control_body_source_fingerprint": control_source,
        "control_program_fingerprint": control_program,
        "treatment_body_source_fingerprint_predicted": treatment_source,
        "treatment_program_fingerprint_predicted": treatment_program,
        "derivation": "offline replay of compiled 017 source FNV identity chain; treated only as prediction until an independent native treatment identity probe matches",
        "source_code": {
            "compiled_head": BUILD_COMPILED_HEAD,
            "runtime_header": str(RUNTIME_SOURCE),
            "probe_source": str(PROBE_SOURCE),
            "loaded_metallib": str(shader_path),
            "intervention_bytes_hex": intervention_payload.hex()
        }
    }


def scene_summary(invocation_path, scene_dir, expected_receipt):
    scene_dir = Path(scene_dir).resolve()
    inv_path = regular(invocation_path, "control invocation")
    invocation = read_json(inv_path, "control invocation")
    md_path = regular(scene_dir / "run-metadata.json", "control owner run metadata")
    metadata = json.loads(md_path.read_text(encoding="utf-8"))
    log_path = regular(scene_dir / "native.log", "control native log")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    require(metadata.get("exit_code") == 0, "control owner run-metadata does not report exit 0")
    require(metadata.get("loaded_metal_runtime", {}).get("verified") is True,
            "control run-metadata does not verify its loaded Metal runtime")
    require(metadata.get("source_files_changed_during_run") == [],
            "control run reports changed source files")
    for key in ("argv", "asset_sha256", "environment"):
        require(metadata.get(key) == invocation.get(key),
                "control run-metadata " + key + " does not match invocation")
    summary = owner.native_scene_summary(log)
    argv = invocation.get("argv", [])
    try:
        requested_steps = int(argv[argv.index("--muscle-step-count") + 1])
        requested_dt = float(argv[argv.index("--muscle-step-seconds") + 1])
    except (ValueError, IndexError, TypeError):
        raise ValueError("control invocation lacks native step-count/timestep flags")
    terminal_line = next((x for x in reversed(log.splitlines()) if x.startswith("stand_terminal_state=")), None)
    require(terminal_line is not None, "native control log has no terminal accepted-state record")
    terminal = json.loads(terminal_line.split("=", 1)[1])
    require(summary["accepted_steps"] == requested_steps == STEP_COUNT,
            "control accepted count must exactly equal the predeclared 10,000-root 20 s preflight")
    require(requested_dt == DT and abs(summary["simulated_s"] - requested_steps * requested_dt) <= 1e-5,
            "control preflight does not match 20 s at 2 ms")
    require(terminal.get("timestep_seconds") == DT and terminal.get("root_assistance") is False,
            "control native terminal timestep/assistance evidence is invalid")
    require(summary["device"] == "Apple M4 Pro", "control run is not from the expected Apple M4 Pro")
    try:
        owner.validate_native_310s_invocation(invocation)
    except Exception as exc:
        raise ValueError("control invocation fails frozen 310 s admission: " + str(exc))
    argv_value = lambda flag: argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None
    receipt_path = Path(argv_value("--resting-anatomy-receipt") or "")
    nha_path = Path(argv_value("--torso-anatomy-payload") or "")
    require(receipt_path.resolve() == expected_receipt.resolve(),
            "control invocation receipt flag differs from supplied corrected receipt")
    require(nha_path.is_file() and not nha_path.is_symlink(),
            "control invocation torso-anatomy payload is missing or symlinked")
    bindings = invocation.get("asset_sha256", {})
    require(isinstance(bindings, dict) and bool(bindings), "control invocation lacks asset SHA bindings")
    for path, digest in bindings.items():
        bound_path = Path(path)
        require(bound_path.is_absolute() and bound_path.is_file() and not bound_path.is_symlink(),
                "control invocation contains a missing or symlinked bound asset: " + str(path))
        require(sha(bound_path) == digest, "control invocation asset SHA mismatch: " + str(path))
    require(str(receipt_path.resolve()) in bindings and sha(receipt_path) == bindings[str(receipt_path.resolve())],
            "corrected anatomy receipt is not byte-bound by the control invocation")
    receipt = read_json(receipt_path, "corrected anatomy receipt")
    require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1",
            "corrected anatomy receipt has an unexpected schema")
    payload = receipt.get("payload", {})
    require(payload.get("path") == str(nha_path.resolve()) and payload.get("sha256") == sha(nha_path),
            "receipt payload path/SHA does not identify the exact invocation NHANATOMY")
    require(str(nha_path.resolve()) in bindings and sha(nha_path) == bindings[str(nha_path.resolve())],
            "corrected NHANATOMY is not byte-bound by the control invocation")
    require(Path(argv[4]).resolve() == scene_dir, "control invocation output directory differs from preflight directory")
    movie_index = argv.index("--resting-movie") if "--resting-movie" in argv else -1
    require(movie_index >= 0 and Path(argv[movie_index + 1]).resolve() == (scene_dir / "native-viewer.mov").resolve(),
            "control invocation movie output differs from preflight directory")
    binary = Path(argv[0]).resolve()
    require(binary == NATIVE_BINARY.resolve() and sha(binary) == EXPECTED_BINARY_SHA,
            "preflight is not using the exact terminal-capture-017 native binary")
    require(str(RESP_METALLIB.resolve()) in bindings and sha(RESP_METALLIB) == bindings[str(RESP_METALLIB.resolve())],
            "the exact linked respiration metallib is not bound")
    require(str(LIBMETALROBO.resolve()) in bindings and sha(LIBMETALROBO) == bindings[str(LIBMETALROBO.resolve())],
            "the exact frozen014 physical MetalRobo library is not bound")
    return invocation, metadata, log, summary, requested_steps, requested_dt


def treatment_probe(scene_dir, control_invocation, control_summary, control_log):
    probe_dir = Path(scene_dir) / PROGRAM_PROBE_NAME
    require(probe_dir.is_dir() and not probe_dir.is_symlink(),
            "treatment-program-probe must be a real child directory of the corrected control preflight")
    inv_path = regular(probe_dir / "invocation.json", "direct-native treatment identity-probe invocation")
    md_path = regular(probe_dir / "run-metadata.json", "direct-native treatment identity-probe run metadata")
    log_path = regular(probe_dir / "native.log", "treatment identity-probe native log")
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    md = json.loads(md_path.read_text(encoding="utf-8"))
    log = log_path.read_text(encoding="utf-8", errors="replace")
    require(md.get("exit_code") == 0 and md.get("source_files_changed_during_run") == [],
            "treatment identity probe did not complete successfully with unchanged sources")
    control_md = read_json(Path(scene_dir) / "run-metadata.json", "control owner run metadata")
    for key in ("host", "machine", "system", "qualification"):
        require(md.get(key) == control_md.get(key) and
                inv.get(key) == control_invocation.get(key) == md.get(key),
                "treatment probe owner/invocation metadata differs from control for " + key)
    require(md.get("loaded_metal_runtime", {}).get("verified") is True,
            "treatment identity probe lacks loaded Metal runtime verification")
    require(md.get("direct_native_probe_validation", {}).get("passed") is True,
            "direct-native treatment identity recorder did not pass its post-run validation")
    for key in ("argv", "asset_sha256", "environment"):
        require(md.get(key) == inv.get(key), "treatment probe metadata " + key + " mismatch")
    ca = list(control_invocation["argv"])
    ta = list(inv.get("argv", []))
    require(len(ta) > 4 and Path(ta[4]).resolve() == probe_dir.resolve(),
            "treatment identity probe must be the exact child treatment-program-probe directory")
    movie_idx = ta.index("--resting-movie") if "--resting-movie" in ta else -1
    require(movie_idx >= 0 and movie_idx + 1 < len(ta) and
            Path(ta[movie_idx + 1]).resolve() == (probe_dir / "native-viewer.mov").resolve(),
            "treatment identity probe movie must be retained inside its required child directory")
    require("--resting-drive-intervention" not in ca,
            "control preflight must not contain an intervention")
    require("--resting-drive-intervention" in ta,
            "treatment identity probe is missing the prescribed intervention")
    k = ta.index("--resting-drive-intervention")
    try:
        vals = [float(ta[k + i]) for i in (1, 2, 3)]
    except (ValueError, IndexError):
        raise ValueError("treatment probe must carry exact start/end/scale values")
    require(vals == [60.0, 100.0, 0.5] and k + 4 == len(ta),
            "treatment probe must append only the exact [60,100,0.5] intervention tuple")
    del ta[k:k + 4]
    require(len(ca) > 4 and len(ta) > 4, "probe native argv is malformed")
    ca[4] = ta[4] = "<RUN_OUTPUT>"
    for arr in (ca, ta):
        require("--resting-movie" in arr, "identity probe does not preserve native viewer output")
        idx = arr.index("--resting-movie")
        arr[idx + 1] = "<MOVIE_OUTPUT>"
    require(ca == ta, "treatment probe differs from control beyond outputs and the registered intervention")
    require(inv.get("asset_sha256") == control_invocation.get("asset_sha256"),
            "treatment identity probe asset hashes differ from control")
    ce = dict(control_invocation.get("environment", {}))
    te = dict(inv.get("environment", {}))
    fail_key = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
    if fail_key in ce or fail_key in te:
        ce[fail_key] = te[fail_key] = "<RUN_FAILURE_RECEIPT>"
    require(ce == te, "treatment probe environment differs from control beyond its unique output receipt path")
    direct_probe = md.get("direct_native_probe", {})
    require(direct_probe.get("scope") ==
            "direct native 20 s program-identity probe; configured future intervention is outside the simulated interval; no treatment dose or physiological response is observed",
            "treatment probe metadata does not identify the direct-native identity-only scope")
    require(direct_probe.get("intervention_overlaps_simulated_horizon") is False and
            direct_probe.get("dose_observed") is False and
            direct_probe.get("recorder_sha256"),
            "treatment probe metadata lacks future-interval or recorder provenance")
    require(direct_probe.get("identity_algorithm_source_sha256") == sha(PROBE_SOURCE),
            "treatment probe parser/source binding differs from the pinned compiled source")
    summary = owner.native_scene_summary(log)
    try:
        steps = int(inv["argv"][inv["argv"].index("--muscle-step-count") + 1])
        dt = float(inv["argv"][inv["argv"].index("--muscle-step-seconds") + 1])
    except (ValueError, IndexError, TypeError):
        raise ValueError("treatment identity probe lacks step-count/timestep flags")
    terminal_line = next((x for x in reversed(log.splitlines()) if x.startswith("stand_terminal_state=")), None)
    require(terminal_line is not None, "treatment identity probe lacks native terminal state")
    terminal = json.loads(terminal_line.split("=", 1)[1])
    require(steps == STEP_COUNT and dt == DT and summary["accepted_steps"] == steps and
            terminal.get("timestep_seconds") == DT and terminal.get("root_assistance") is False,
            "treatment probe must match control's exact 20 s / 2 ms identity-only run")
    require(summary["world_fingerprint"] == control_summary["world_fingerprint"] and
            summary["device"] == control_summary["device"],
            "treatment probe world/device identity differs from control")
    ids = program_ids(control_log, RESP_METALLIB)
    require(summary["body_source_fingerprint"] == ids["treatment_body_source_fingerprint_predicted"] and
            summary["coupled_program_fingerprint"] == ids["treatment_program_fingerprint_predicted"],
            "native treatment identity probe disagrees with the independently derived source identity")
    return summary, ids, probe_dir


def verify_build_and_pins():
    require(sha(BUILD_MANIFEST) == BUILD_MANIFEST_SHA, "017 build pins changed; do not reuse these pins")
    require(sha(BUILD_SOURCE_PINS) == BUILD_SOURCE_PINS_SHA, "017 source pins changed")
    require(sha(BUILD_FOCUSED_TESTS) == BUILD_FOCUSED_TESTS_SHA, "017 focused-test report changed")
    pins = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    source_pins = json.loads(BUILD_SOURCE_PINS.read_text(encoding="utf-8"))
    expected_sources = {k: v for k, v in pins.items()
                        if isinstance(v, str) and k not in ("source_revision", "build_script_sha256")}
    require(source_pins == {**expected_sources,
                            "source_revision": pins.get("source_revision"),
                            "build_script_sha256": pins.get("build_script_sha256")},
            "017 source-pins file differs from the build-pins source inventory")
    require(pins.get("source_revision") == BUILD_COMPILED_HEAD and
            pins.get("build_script_sha256") == BUILD_PATCH_SHA,
            "017 build pins do not identify the reviewed source and build script")
    for rel, expected in expected_sources.items():
        path = BUILD_SOURCE / rel
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "017 source pin mismatch: " + str(path))
    require(git(BUILD_SOURCE, "rev-parse", "HEAD") == BUILD_COMPILED_HEAD,
            "017 source checkout is not the exact compiled commit")
    require(git(BUILD_SOURCE, "status", "--porcelain") == "",
            "017 source checkout has uncommitted changes")
    require(git(BUILD_SOURCE, "cat-file", "-e", BUILD_COMPILED_HEAD + "^{commit}") == "",
            "017 compiled source commit is missing")
    require(subprocess.run(["git", "-C", str(BUILD_SOURCE), "merge-base", "--is-ancestor",
                            BUILD_COMPILED_HEAD, BUILD_EVIDENCE_HEAD],
                           check=False).returncode == 0,
            "merged main commit does not contain the exact compiled 017 source")
    source_diff = subprocess.check_output(["git", "-C", str(BUILD_SOURCE), "diff",
                                           "HEAD^", "HEAD", "--binary"])
    require(hashlib.sha256(source_diff).hexdigest() ==
            "4a3b02e52f769d25e2dede362f81fd9447c72c4a956afc3b773a4fcd17ff47df",
            "017 source diff differs from the reviewed terminal-capture change")
    runtime_text = RUNTIME_SOURCE.read_text(encoding="utf-8")
    probe_text = PROBE_SOURCE.read_text(encoding="utf-8")
    visual_text = (BUILD_SOURCE / "apps/NumiHumanRestingVisual.hpp").read_text(encoding="utf-8")
    cadence_text = (BUILD_SOURCE / "apps/NumiHumanAcceptedGeometryCadence.hpp").read_text(encoding="utf-8")
    coupling_text = (BUILD_SOURCE / "apps/NumiHumanRestingCoupling.hpp").read_text(encoding="utf-8")
    terminal_publisher = coupling_text.split("void publishTerminalSnapshot", 1)[1]
    require("if(bodySourceIdentity)mix(&bodySourceIdentity,sizeof(bodySourceIdentity));" in runtime_text and
            "mix(libraryBytes.bytes,libraryBytes.length);" in runtime_text,
            "017 source no longer matches the pinned native identity-FNV algorithm")
    require('appendSource("resting_drive_intervention",std::as_bytes(std::span(*restingDriveIntervention)))' in probe_text,
            "017 source no longer appends the prescribed intervention identity payload")
    require("resting_terminal_capture_identity=accepted_step_" in probe_text and
            "terminal_physical_steps_advanced=0" in probe_text,
            "017 terminal snapshot does not declare accepted N and zero physical advancement")
    require("MetalArticulatedOperator_pointJacobiansOnly" in probe_text and
            "publishTerminalSnapshot(terminalStep" in probe_text and
            "[commitEncoder setBuffer:acceptedCommonCoordinates" in terminal_publisher and
            "[commitEncoder setBuffer:presentationFrameCommonCoordinates" in terminal_publisher,
            "017 terminal publication no longer uses query-only accepted poses and copies the GPU-owned accepted common-coordinate buffer")
    require('#include "NumiHumanAcceptedGeometryCadence.hpp"' in visual_text and
            "classifyCaptureStep(" in visual_text and
            "CaptureStepClass::terminal" in cadence_text and
            "terminalSnapshotReady(" in cadence_text,
            "017 viewer no longer uses the tested accepted-capture cadence policy")
    require(sha(BUILD_PATCH) == BUILD_PATCH_SHA, "017 native build script changed")
    artifact_pins = pins.get("artifacts", {})
    for path, expected in ((NATIVE_BINARY, EXPECTED_BINARY_SHA),
                           (RESP_METALLIB, RESP_METALLIB_SHA),
                           (BUILD_LIBMETALROBO, LIBMETALROBO_SHA)):
        resolved = path.resolve()
        require(artifact_pins.get(str(path)) == expected and
                path.is_file() and not path.is_symlink() and sha(path) == expected,
                "017 build artifact identity mismatch: " + str(path))
    focused = json.loads(BUILD_FOCUSED_TESTS.read_text(encoding="utf-8"))
    results = focused.get("results", [])
    require(len(results) >= 2 and all(x.get("compile_exit") == 0 and x.get("test_exit") == 0 for x in results),
            "017 focused native cadence/surface tests did not all pass")
    for path, digest, label in (
        (E / "native-terminal-capture-review-930/verification-v2.json",
         "b6d9e92fd6a27f81d58281923808ea45b8e256f2274bfb734b62d4299842886a", "930"),
        (E / "native-terminal-cycle-review-931/verification.json",
         "c9429ec3a10296ead8e6963899c7278736d513f2afeb08d5afb60f167e4fe849", "931"),
        (TERMINAL_Q0_COM8_932, TERMINAL_Q0_COM8_932_SHA, "932 q0/COM8")):
        regular(path, label + " terminal-capture regression")
        require(sha(path) == digest, label + " terminal-capture regression changed")
        report = json.loads(path.read_text(encoding="utf-8"))
        require(report.get("pass") is True, label + " terminal-capture regression did not pass")
    reduced = json.loads(TERMINAL_Q0_COM8_932.read_text(encoding="utf-8"))
    expected_aggregates = [
        "min_contact_gap_m", "peak_penetration_m",
        "pre_projection_contact_residual_m_s", "pre_projection_limit_residual_generalized_s",
        "pre_projection_equality_residual_generalized_s", "post_projection_contact_residual_m_s",
        "post_projection_limit_residual_generalized_s", "post_projection_equality_residual_generalized_s",
        "equality_position_projection_max_generalized", "equality_velocity_projection_max_generalized_s",
    ]
    expected_packs = {
        "0": "a88fcb852681e75d6453a5db2719f4201d454b68cda71ca4bf07ace3c5691a4c",
        "63": "8422cb46bcef9251d359e26dfb331a3e7dfa8a2960963428266a797fc15614a1",
        "64": "2bcd55d5b75c353e6c412bf7ded0457947810e6656d4b809574d23efabd6e358",
    }
    require(reduced.get("q_integration_audit") == 0 and
            reduced.get("com_segment_steps") == 8 and reduced.get("com_observations") == 8 and
            reduced.get("reference_trace_rows") == 64 and reduced.get("candidate_trace_rows") == 8 and
            reduced.get("identical_endpoint_columns") == 50 and
            reduced.get("exact_window_aggregated_columns") == expected_aggregates and
            reduced.get("window_aggregation") ==
            "minimum for min_contact_gap_m; maximum for the other nine diagnostic columns" and
            {step: value.get("pack_sha256") for step, value in reduced.get(
                "captures_byte_identical_to_q1_com1_run930", {}).items()} == expected_packs,
            "932 q0/COM8 terminal regression semantics or pack identities changed")
    require(LIBMETALROBO.is_file() and not LIBMETALROBO.is_symlink() and
            sha(LIBMETALROBO) == LIBMETALROBO_SHA,
            "the exact frozen014 physical MetalRobo library differs from the compiled library")
    regular(CARDIAC_911, "cardiac interface localization 911 report")
    require(sha(CARDIAC_911) == CARDIAC_911_SHA, "cardiac interface localization 911 report changed")
    regular(HUMAN_RESTING_RUN, "frozen Human loaded-runtime verifier")
    require(sha(HUMAN_RESTING_RUN) == HUMAN_RESTING_RUN_SHA,
            "frozen Human loaded-runtime verifier changed")
    regular(TERMINAL_931_NATIVE_LOG, "931 terminal-cycle native log fixture")
    require(sha(TERMINAL_931_NATIVE_LOG) == TERMINAL_931_NATIVE_LOG_SHA,
            "931 terminal-cycle native log fixture changed")
    for path, label in ((E / "native-source-state-cycle-914/invocation.json", "914 invocation"),
                        (E / "native-source-state-cycle-914/run-metadata.json", "914 run-metadata"),
                        (E / "native-source-state-cycle-review-914/verification.json", "914 verification")):
        regular(path, label)
    lab_head = git(LAB, "rev-parse", "HEAD")
    human_head = git(HUMAN, "rev-parse", "HEAD")
    brain_head = git(BRAIN, "rev-parse", "HEAD")
    require(lab_head == FROZEN_LAB_REV and human_head == FROZEN_HUMAN_REV and brain_head == FROZEN_BRAIN_REV,
            "frozen owner source revisions changed; refresh source pins rather than relabeling")
    return pins, lab_head, human_head, brain_head

def source_pin_files(manifest):
    old_hashes = read_json(BASE_HASHES, "865 source hash inventory")
    old_revisions = read_json(BASE_REVISIONS, "865 source revision inventory")
    for path, expected in old_hashes.items():
        require(Path(path).is_file() and not Path(path).is_symlink() and sha(path) == expected,
                "frozen865 source hash changed: " + path)
    for name, expected, repo in (
        ("numi-lab", FROZEN_LAB_REV, LAB),
        ("numilab-human", FROZEN_HUMAN_REV, HUMAN),
        ("numi-brain", FROZEN_BRAIN_REV, BRAIN)):
        require(old_revisions[name]["revision"] == expected and git(repo, "rev-parse", "HEAD") == expected,
                "865 source revision differs for " + name)
        require(git(repo, "status", "--porcelain") == "", "frozen source tree is dirty: " + str(repo))
    hashes = dict(old_hashes)
    additions = [
        BASE_HASHES, BASE_REVISIONS, BUILD_MANIFEST, BUILD_SOURCE_PINS, BUILD_FOCUSED_TESTS,
        BUILD_PATCH, NATIVE_BINARY, RESP_METALLIB, BUILD_LIBMETALROBO, LIBMETALROBO,
        BUILD_SOURCE / "CMakeLists.txt",
        BUILD_SOURCE / "apps/NumiHumanRestingVisual.hpp",
        BUILD_SOURCE / "apps/NumiHumanRestingCoupling.hpp",
        BUILD_SOURCE / "apps/NumiHumanAcceptedGeometryCadence.hpp",
        BUILD_SOURCE / "apps/numilab_human_myosim_visual_probe.mm",
        BUILD_SOURCE / "matter/src/human_respiration.metal",
        PROBE_SOURCE, RUNTIME_SOURCE,
        CARDIAC_911, HUMAN_RESTING_RUN, TERMINAL_931_NATIVE_LOG,
        E / "native-source-state-cycle-914/invocation.json",
        E / "native-source-state-cycle-914/run-metadata.json",
        E / "native-source-state-cycle-review-914/verification.json",
        E / "native-terminal-capture-review-930/verification-v2.json",
        E / "native-terminal-cycle-review-931/verification.json", TERMINAL_Q0_COM8_932,
        READINESS_SCRIPT, READINESS_ANALYZER, READINESS_README, READINESS_REVISION,
        TREATMENT_PROBE_RECORDER, TREATMENT_PROBE_TESTS, CAPTURE_PLAN_TESTS, OWNER_LINEAGE_TESTS
    ]
    for path in additions:
        q = regular(path, "compiled source/build provenance input")
        hashes[str(q)] = sha(q)
    revisions = dict(old_revisions)
    revisions["numi-human-terminal-capture-017"] = {
        "path": str(BUILD_SOURCE),
        "revision": BUILD_COMPILED_HEAD,
        "merged_main_revision": BUILD_EVIDENCE_HEAD,
        "diff_sha256": "4a3b02e52f769d25e2dede362f81fd9447c72c4a956afc3b773a4fcd17ff47df",
        "status_porcelain": "",
        "build_pins": str(BUILD_MANIFEST),
        "build_pins_sha256": BUILD_MANIFEST_SHA,
        "source_pins": str(BUILD_SOURCE_PINS),
        "source_pins_sha256": BUILD_SOURCE_PINS_SHA,
        "build_script": str(BUILD_PATCH),
        "build_script_sha256": BUILD_PATCH_SHA,
        "scope": "017 native viewer terminal accepted-state capture; frozen Lab014 physical runtime and guard015 respiratory source remain separately pinned"
    }
    return hashes, revisions

def normalize_for_probe(invocation, treatment=False, output_dir=None):
    argv = list(invocation["argv"])
    if treatment:
        require("--resting-drive-intervention" not in argv,
                "control invocation unexpectedly includes a drive intervention")
        argv.extend(["--resting-drive-intervention", "60.0", "100.0", "0.5"])
    argv[4] = str(output_dir)
    if "--resting-movie" in argv:
        argv[argv.index("--resting-movie") + 1] = str(Path(output_dir) / "native-viewer.mov")
    env = dict(invocation["environment"])
    env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"] = str(Path(output_dir) / "common-field-failure.json")
    return argv, env


def print_probe_command(invocation, scene_dir, ids):
    out = Path(scene_dir).resolve() / PROGRAM_PROBE_NAME
    argv, env = normalize_for_probe(invocation, treatment=True, output_dir=out)
    payload = {
        "status": "planned_not_run",
        "working_directory": str(Path(scene_dir).resolve()),
        "expected_program_fingerprint": ids["treatment_program_fingerprint_predicted"],
        "expected_body_source_fingerprint": ids["treatment_body_source_fingerprint_predicted"],
        "device": "Apple M4 Pro", "accepted_steps": STEP_COUNT, "timestep_s": DT,
        "output_directory": str(out), "native_argv": argv, "native_environment": env,
        "instructions": "Run the printed command through the readiness-owned direct-native recorder, preserving invocation.json, run-metadata.json, native.log, and before/after source hashes under output_directory. The intervention begins at 60 s, so this 20 s probe verifies program identity only; it does not exercise dose response. Do not register or launch the 310 s pair until the actual native treatment fingerprint matches this prediction."
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    recorder_command = [
        str(PY39), str(TREATMENT_PROBE_RECORDER),
        "--scene-preflight-dir", str(Path(scene_dir).resolve()),
        "--anatomy-receipt", str(Path(scene_dir).resolve().parent / "resting-anatomy-receipt.json"),
        "--execute",
    ]
    # The caller's corrected receipt path is supplied separately by main below;
    # this preview uses the exact flag path already present in the control argv.
    argv_receipt_index = argv.index("--resting-anatomy-receipt")
    recorder_command[recorder_command.index("--anatomy-receipt") + 1] = argv[argv_receipt_index + 1]
    print("Direct-native identity probe recorder command (not executed):")
    print(shlex.join(recorder_command))
    print("The recorder will retain invocation.json, run-metadata.json, native.log, and before/after source hashes; it will not use the duration-bounded Human intervention builder.")



def write_new_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bind_accepted_geometry_capture_plan(plan, out, scene_dir, control_invocation):
    """Add exact arm-specific export schedules without changing the owner preflight."""
    schedule = accepted_geometry_capture_schedule()
    schedule_path = Path(out) / CAPTURE_PLAN_FILE
    write_new_json(schedule_path, schedule)
    parent_path = (Path(scene_dir) / "invocation.json").resolve()
    parent_sha = sha(parent_path)
    template_paths = {}
    template_hashes = {}
    templates = {}
    for arm in ("control", "treatment"):
        template_path = Path(out) / ("native-launch-template-" + arm + ".json")
        template = make_capture_launch_template(control_invocation, parent_path, parent_sha,
                                               schedule, arm)
        write_new_json(template_path, template)
        template_paths[arm] = str(template_path.resolve())
        template_hashes[arm] = sha(template_path)
        templates[arm] = template
    for trial_id, arm in (("resting-baseline", "control"),
                          ("resting-drive-half", "treatment")):
        trial = next((x for x in plan.get("trials", []) if x.get("id") == trial_id), None)
        require(trial is not None, "owner plan omitted registered arm " + trial_id)
        argv = trial.get("argv", [])
        require(argv.count("--invocation") == 1 and argv.index("--invocation") + 1 < len(argv),
                trial_id + " argv does not have exactly one invocation path")
        argv[argv.index("--invocation") + 1] = template_paths[arm]
    plan["design"]["accepted_geometry_capture_schedule"] = schedule
    plan["design"]["accepted_geometry_capture_plan_path"] = str(schedule_path.resolve())
    plan["design"]["accepted_geometry_capture_plan_sha256"] = sha(schedule_path)
    plan["design"]["accepted_geometry_capture_template_paths"] = template_paths
    plan["design"]["accepted_geometry_capture_template_hashes"] = template_hashes
    identity_path = Path(out) / "native-build-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["accepted_geometry_capture_plan"] = {
        "path": str(schedule_path.resolve()), "sha256": sha(schedule_path),
        "schema": CAPTURE_PLAN_SCHEMA, "terminal_step_id": FINAL_ACCEPTED_STEPS,
        "terminal_nominal_time_s": FINAL_ACCEPTED_STEPS * FINAL_DT_S,
        "terminal_accepted_time_s": FINAL_ACCEPTED_STEPS * NATIVE_FLOAT_DT_S
    }
    identity["accepted_geometry_capture_launch_templates"] = {
        arm: {"path": template_paths[arm], "sha256": template_hashes[arm],
              "parent_invocation_path": str(parent_path), "parent_invocation_sha256": parent_sha,
              "scope": "Derived export-step environment only; not an owner run receipt"}
        for arm in ("control", "treatment")
    }
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
    calibration_path = Path(out) / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration.setdefault("bindings", {})
    for path in [schedule_path, *(Path(x) for x in template_paths.values())]:
        calibration["bindings"][str(path.resolve())] = sha(path)
    plan["artifacts"] = list(dict.fromkeys(
        [*plan.get("artifacts", []), str(schedule_path.resolve()),
         *(template_paths[a] for a in ("control", "treatment"))]))
    plan["instrument"]["artifacts"] = list(dict.fromkeys(
        [*plan["instrument"].get("artifacts", []), str(schedule_path.resolve()),
         *(template_paths[a] for a in ("control", "treatment"))]))
    for item in plan["instrument"]["artifacts"]:
        candidate = Path(item)
        if candidate.is_file() and not candidate.is_symlink():
            calibration["bindings"][str(candidate.resolve())] = sha(candidate)
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                encoding="utf-8")
    return schedule, template_paths, template_hashes, parent_path, parent_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-preflight-dir", default=SCENE_MARKER,
                        help="corrected control preflight directory (contains invocation.json, run-metadata.json, native.log)")
    parser.add_argument("--anatomy-receipt", default=RECEIPT_MARKER,
                        help="exact corrected resting-anatomy-receipt.json bound by the control invocation")
    parser.add_argument("--draft-dir", type=Path, default=READINESS_ROOT / "prepared-plan-001",
                        help="new science-v2 plan draft directory; this script never registers or runs it")
    parser.add_argument("--print-treatment-probe-command", action="store_true",
                        help="validate only the control preflight and print exact treatment identity-probe argv/env; do not run it")
    args = parser.parse_args()
    if args.scene_preflight_dir == SCENE_MARKER or args.anatomy_receipt == RECEIPT_MARKER:
        parser.error("replace both pending path placeholders or pass --scene-preflight-dir and --anatomy-receipt")
    require("__PENDING_" not in args.scene_preflight_dir and "__PENDING_" not in args.anatomy_receipt,
            "unresolved placeholder remains in a supplied path")
    require(sys.byteorder == "little", "offline source identity reconstruction requires Apple arm64 little-endian")
    require(not Path(args.draft_dir).exists(), "draft output exists; do not overwrite prior plan evidence")
    require(not STUDY.exists(), "final study directory already exists; select a new study path before registration")
    require(SCIENCE.is_file() and PY39.is_file() and OWNER.is_file(), "frozen Lab owner CLI/runtime is missing")
    manifest, lab_head, human_head, brain_head = verify_build_and_pins()
    scene_input = Path(args.scene_preflight_dir).expanduser()
    require(not scene_input.is_symlink(), "corrected scene preflight directory must not be a symlink")
    scene_dir = scene_input.resolve()
    require(scene_dir.is_dir(), "corrected scene preflight directory is missing")
    receipt = regular(args.anatomy_receipt, "corrected anatomy receipt")
    control_invocation, control_metadata, control_log, control_summary, steps, dt = scene_summary(
        scene_dir / "invocation.json", scene_dir, receipt)
    ids = program_ids(control_log, RESP_METALLIB)
    if args.print_treatment_probe_command:
        print_probe_command(control_invocation, scene_dir, ids)
        return 0
    treatment_summary, ids, probe_dir = treatment_probe(scene_dir, control_invocation, control_summary, control_log)
    require(treatment_summary["coupled_program_fingerprint"] == ids["treatment_program_fingerprint_predicted"],
            "treatment program identity probe mismatch")

    hashes, revisions = source_pin_files(manifest)
    out = Path(args.draft_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    hash_path = out.parent / "source-hashes-final.json"
    revision_path = out.parent / "source-revisions-final.json"
    require(not hash_path.exists() and not revision_path.exists(),
            "source pin output already exists; preserve it and select a fresh draft directory")
    hash_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    revision_path.write_text(json.dumps(revisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args_list = [
        str(PY39), str(OWNER), "prepare-native-310s",
        "--repository", str(HUMAN),
        "--directory", str(out),
        "--invocation", str((scene_dir / "invocation.json").resolve()),
        "--source-hashes", str(hash_path),
        "--source-revisions", str(revision_path),
        "--parser-fixture", str(PARSER_FIXTURE),
        "--world-fingerprint", str(control_summary["world_fingerprint"]),
        "--control-program-fingerprint", str(ids["control_program_fingerprint"]),
        "--treatment-program-fingerprint", str(ids["treatment_program_fingerprint_predicted"]),
        "--device", control_summary["device"],
        "--segment8-reference", str(SEGMENT8),
        "--full-q-reference", str(FULL_Q),
        "--runtime-correctness-reference", str(RUNTIME_REFERENCE)
    ]
    proc = subprocess.run(args_list, cwd=str(LAB), capture_output=True, text=True, check=False)
    if proc.returncode:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise ValueError("frozen owner failed to prepare the science-v2 plan; retained source pin files")
    plan_path = out / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    (capture_schedule, capture_template_paths, capture_template_hashes,
     parent_invocation_path, parent_invocation_sha) = bind_accepted_geometry_capture_plan(
        plan, out, scene_dir, control_invocation)
    plan["artifacts"] = list(dict.fromkeys([*plan.get("artifacts", []), str(CARDIAC_911)]))
    plan["instrument"]["artifacts"] = list(dict.fromkeys(
        [*plan["instrument"].get("artifacts", []), str(CARDIAC_911)]))
    plan["design"]["current_source_preflight"] = {
        "native_owner_run_directory": str(scene_dir),
        "control_invocation_sha256": sha(scene_dir / "invocation.json"),
        "control_run_metadata_sha256": sha(scene_dir / "run-metadata.json"),
        "treatment_identity_probe_directory": str(probe_dir),
        "treatment_identity_probe_invocation_sha256": sha(probe_dir / "invocation.json"),
        "treatment_identity_probe_run_metadata_sha256": sha(probe_dir / "run-metadata.json"),
        "treatment_identity_probe_scope": json.loads(
            (probe_dir / "run-metadata.json").read_text(encoding="utf-8")
        )["direct_native_probe"],
        "treatment_identity_probe_recorder": str(TREATMENT_PROBE_RECORDER),
        "treatment_identity_probe_recorder_sha256": sha(TREATMENT_PROBE_RECORDER),
        "corrected_anatomy_receipt": str(receipt),
        "corrected_anatomy_receipt_sha256": sha(receipt),
        "accepted_geometry_capture_plan_path": str((out / CAPTURE_PLAN_FILE).resolve()),
        "accepted_geometry_capture_plan_sha256": sha(out / CAPTURE_PLAN_FILE),
        "accepted_geometry_capture_template_paths": capture_template_paths,
        "accepted_geometry_capture_template_hashes": capture_template_hashes,
        "capture_template_parent_invocation_path": str(parent_invocation_path),
        "capture_template_parent_invocation_sha256": parent_invocation_sha,
        "capture_template_parent_run_metadata_sha256": sha(parent_invocation_path.with_name("run-metadata.json")),
        "capture_template_parent_native_log_sha256": sha(parent_invocation_path.with_name("native.log")),
        "preflight_accepted_steps": steps,
        "preflight_dt_s": dt,
        "preflight_duration_s": steps * dt,
        "preflight_device": control_summary["device"],
        "preflight_world_fingerprint": control_summary["world_fingerprint"],
        "preflight_body_source_fingerprints": {
            "control": ids["control_body_source_fingerprint"],
            "treatment": ids["treatment_body_source_fingerprint_predicted"]
        },
        "preflight_program_fingerprints": {
            "control": ids["control_program_fingerprint"],
            "treatment": ids["treatment_program_fingerprint_predicted"]
        },
        "treatment_program_fingerprint": {
            "predicted_offline": ids["treatment_program_fingerprint_predicted"],
            "native_observed": treatment_summary["coupled_program_fingerprint"],
            "matched": treatment_summary["coupled_program_fingerprint"] == ids["treatment_program_fingerprint_predicted"]
        },
        "compiled_native_provenance": {
            "build_pins": str(BUILD_MANIFEST),
            "build_pins_sha256": BUILD_MANIFEST_SHA,
            "source_pins": str(BUILD_SOURCE_PINS),
            "source_pins_sha256": BUILD_SOURCE_PINS_SHA,
            "focused_tests": str(BUILD_FOCUSED_TESTS),
            "focused_tests_sha256": BUILD_FOCUSED_TESTS_SHA,
            "source_revision": BUILD_COMPILED_HEAD,
            "merged_main_revision": BUILD_EVIDENCE_HEAD,
            "source_diff_sha256": "4a3b02e52f769d25e2dede362f81fd9447c72c4a956afc3b773a4fcd17ff47df",
            "build_script": str(BUILD_PATCH),
            "build_script_sha256": BUILD_PATCH_SHA,
            "native_binary": str(NATIVE_BINARY),
            "native_binary_sha256": EXPECTED_BINARY_SHA,
            "physical_library": str(LIBMETALROBO),
            "physical_library_sha256": LIBMETALROBO_SHA,
            "respiratory_metallib": str(RESP_METALLIB),
            "respiratory_metallib_sha256": RESP_METALLIB_SHA,
            "scope": "017 provides exact accepted-state terminal geometry publication. It links the frozen014 physical MetalRobo library and guard015 respiratory source; the 017 build-pins record identifies capture code and artifacts, not a new physics qualification."
        },
        "scope": "20 s accepted native asset/integration preflight only; not 310 s endurance, physiological qualification, or whole-body anatomy clearance."
    }
    plan["design"]["post_run_reference_reporting"] = {
        "physiology_ranges": "Generic adult reference ranges are descriptive context; retain and report every outlier. Do not tune thresholds after seeing outcomes.",
        "pulsatile_pressures": "Pulmonary artery and aortic samples are instantaneous pulse pressures; compare mean-pressure references only with explicitly labeled complete-cycle means/proxies.",
        "supine_cohort_summaries": "Supine cohort means and standard deviations are cohort context, not clinical intervals or subject cutoffs.",
        "clinical_reference_sources": [
            {"source": "MedlinePlus ABG", "url": "https://medlineplus.gov/lab-tests/arterial-blood-gas-abg-test/", "values": "PaO2 75-100 mmHg, PaCO2 35-45 mmHg, oxygen saturation 95-100%; generic ABG context, not an individual diagnosis."},
            {"source": "MedlinePlus Vital Signs", "url": "https://medlineplus.gov/ency/article/002341.htm", "values": "Average healthy resting adult breathing 12-18/min; variation with age, sex, weight, activity, and health."},
            {"source": "Kovacs et al. 2009", "url": "https://doi.org/10.1183/09031936.00145608", "values": "Review total 1,187 individuals across 47 studies; supine subset n=882, mean resting mPAP 14.0 +/- 3.3 mmHg."},
            {"source": "Mendes et al. 2020", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7253877/", "values": "Male supine quiet-breathing cohort means +/- SD: RR 16.15 +/- 4.72/min, VT 0.58 +/- 0.28 L, VE 8.32 +/- 2.78 L/min."}
        ],
        "causal_scope": "One deterministic paired simulation; no individual clinical prediction or population probability."
    }
    plan["limitations"] += (
        " The terminal-capture-017 executable is pinned separately from the frozen Lab science/owner CLI: "
        "its build/source pins identify exact accepted-state presentation code and the compiled artifacts, "
        "while it links the frozen014 physical MetalRobo library and guard015 respiratory source. This "
        "capture build identity does not independently qualify runtime physics. "
        "The 20 s owner preflight checks current corrected assets and program identities only; it is not "
        "310 s endurance, physiological validation, or whole-body anatomy acceptance. "
        "Cardiac interface localization 911 reports 11,568 current source-neutral RA/RV intersections near "
        "the common-map tricuspid leaflet projection with a localized source-to-current RV residual up to 0.75 mm. This "
        "supports a junction-localized segmentation-overlap interpretation but establishes neither a 3D "
        "leaflet surface nor valve-plane/orifice ownership. Reduced-order CVSim chambers and valves remain "
        "the sole functional and blood owner; this localization is a stated anatomy limitation, not a "
        "geometry-clearance pass.")
    readiness_docs = [str(READINESS_SCRIPT), str(READINESS_ANALYZER), str(READINESS_README), str(READINESS_REVISION)]
    for path in readiness_docs:
        require(Path(path).is_file() and not Path(path).is_symlink(),
                "readiness document is missing or symlinked: " + path)
    plan["artifacts"] = list(dict.fromkeys([*plan["artifacts"], str(BUILD_MANIFEST), str(BUILD_SOURCE_PINS),
        str(BUILD_FOCUSED_TESTS), str(BUILD_PATCH),
        str(E / "native-source-state-cycle-914/run-metadata.json"),
        str(E / "native-source-state-cycle-review-914/verification.json"),
        str(E / "native-terminal-capture-review-930/verification-v2.json"),
        str(E / "native-terminal-cycle-review-931/verification.json"),
        str(TERMINAL_Q0_COM8_932), *readiness_docs]))
    plan["instrument"]["artifacts"] = list(dict.fromkeys([*plan["instrument"]["artifacts"],
        str(BUILD_MANIFEST), str(BUILD_SOURCE_PINS), str(BUILD_FOCUSED_TESTS), str(BUILD_PATCH),
        str(E / "native-source-state-cycle-914/run-metadata.json"),
        str(E / "native-source-state-cycle-review-914/verification.json"),
        str(E / "native-terminal-capture-review-930/verification-v2.json"),
        str(E / "native-terminal-cycle-review-931/verification.json"),
        str(TERMINAL_Q0_COM8_932), *readiness_docs]))
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    registration_dir = STUDY
    supplemental_dir = READINESS_ROOT / "completed-pair-final"
    commands = [
        [str(SCIENCE), "science", "register", str(plan_path), str(registration_dir)],
        [str(SCIENCE), "science", "status", str(registration_dir)],
        [str(SCIENCE), "science", "run", str(registration_dir)],
        [str(SCIENCE), "science", "run", str(registration_dir)],
        [str(SCIENCE), "science", "analyze", str(registration_dir)],
        [str(SCIENCE), "science", "verify", str(registration_dir)],
        [str(PY39), str(READINESS_ANALYZER),
         "--study", str(registration_dir), "--output", str(supplemental_dir)]
    ]
    print(json.dumps({
        "status": "plan_prepared_not_registered_or_run",
        "plan": str(plan_path),
        "plan_sha256": sha(plan_path),
        "native_build_pins_sha256": BUILD_MANIFEST_SHA,
        "compiled_source_revision": BUILD_COMPILED_HEAD,
        "merged_main_revision": BUILD_EVIDENCE_HEAD,
        "accepted_geometry_capture_schedule_sha256": sha(out / CAPTURE_PLAN_FILE),
        "accepted_geometry_capture_template_hashes": capture_template_hashes,
        "control_preflight_steps": steps,
        "control_world_fingerprint": control_summary["world_fingerprint"],
        "control_program_fingerprint": ids["control_program_fingerprint"],
        "treatment_program_fingerprint": ids["treatment_program_fingerprint_predicted"],
        "native_treatment_program_identity_matched": treatment_summary["coupled_program_fingerprint"] == ids["treatment_program_fingerprint_predicted"],
        "asset_identity": json.loads((out / "native-build-identity.json").read_text())["common_asset_identity"],
        "source_revision_frozen_owner_cli": lab_head,
        "source_revision_human": human_head,
        "source_revision_brain": brain_head,
        "cardiac_interface_limitation_sha256": CARDIAC_911_SHA,
        "commands": [shlex.join(map(str, cmd)) for cmd in commands],
        "registration_is_not_automatic": True,
        "no_native_execution_performed": True
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write("REFUSED/FAILED: " + str(exc) + "\n")
        raise SystemExit(2)
