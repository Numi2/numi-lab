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
TERMINAL_931_INVOCATION = E / "native-terminal-cycle-931/invocation.json"
TERMINAL_931_INVOCATION_SHA = "1842eb1f8dba4f83b0ff4e86d491e1936a386eaba91ae06fb5acf84400123175"
TERMINAL_931_RESPIRATION_CONFIG = E / "integrated-anatomy-engineering-lung-edge-collapse-761/resting-reference-respiration.json"
TERMINAL_931_RESPIRATION_CONFIG_SHA = "c518926bf47fba945cef52bb952ed6c559d604508988082c5b641b720bda503d"
BRAIN = Path("/Users/n/numi-human-resting-integration-20261005/numi-brain")
PREPARATION_OWNER_ROOT = Path("/Users/n/numi-human-area-audit-fma-fix-001")
PREPARATION_OWNER = PREPARATION_OWNER_ROOT / "matter/tools/resting_intervention_study.py"
OWNER = PREPARATION_OWNER
PREPARATION_OWNER_REV = "d81e236663e0d3427956a6253ac44900409c145a"
PREPARATION_OWNER_SHA = "81b910a3ef745aa04bd47d1d9c291a1865cccb56f7234021fe10c6335252fe11"
PREPARATION_OWNER_TEST = PREPARATION_OWNER_ROOT / "matter/tools/test_resting_intervention_study.py"
PREPARATION_OWNER_TEST_SHA = "064269efc55671a1f89f240c51c708af872e98c0f6ed8da1e1e1469d6e3536a2"
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
PROGRAM_PROBE_NAME = "treatment-program-probe-v015"
CARDIAC_911 = E / "native-cardiac-interface-localization-911/final-localization-report.json"
CARDIAC_911_SHA = "9c468f7a1da58563387acc9712f845a794f07fa3121c7fa4a3a7ae905ecf03e4"
TERMINAL_Q0_COM8_932 = E / "native-terminal-production-review-932/verification.json"
TERMINAL_Q0_COM8_932_SHA = "453e8f69bdad311d1d273d5049d17b886b609780bfbe3614700e82c424e782f9"
READINESS_ROOT = Path(__file__).resolve().parent
READINESS_SCRIPT = Path(__file__).resolve()
READINESS_DEFAULT_SCRIPT = READINESS_ROOT / "prepare_final_plan.py"
READINESS_VERSIONED_SCRIPT = READINESS_ROOT / "prepare_final_plan-v016.py"
READINESS_ANALYZER = READINESS_ROOT / "analyze_final_pair.py"
READINESS_README = READINESS_ROOT / "README-v016.md"
READINESS_REVISION = READINESS_ROOT / "revision-016.json"
V014_PROBE_PREVIEW = READINESS_ROOT / "actual-v014-preflight-probe-preview-v009.txt"
CAPTURE_PLAN_TESTS = READINESS_ROOT / "test_terminal_capture_plan.py"
OWNER_LINEAGE_TESTS = READINESS_ROOT / "test_owner_invocation_lineage.py"
SCENE_LINEAGE_TESTS = READINESS_ROOT / "test_v015_dynamic_scene_readiness.py"
CALIBRATION_BINDING_TESTS = READINESS_ROOT / "test_final_instrument_calibration_bindings_v015.py"
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
TREATMENT_PROBE_TESTS = READINESS_ROOT / "test_record_treatment_program_probe_v015.py"
V015_ASSEMBLY_TESTS = READINESS_ROOT / "test_v015_dynamic_scene_readiness.py"
P12_IDENTITY_TESTS = READINESS_ROOT / "test_owner_summary_identity_normalization.py"
P12_REVISION_TESTS = READINESS_ROOT / "test_current_git_revision_identity_v013.py"
P12_TEST_LOG = READINESS_ROOT / "test-suite-v016.log"
PREPARATION_OWNER_TEST_LOG = READINESS_ROOT / "preparation-owner-tests-v015.log"
SEGMENT8 = E / "integrated-parallel-contact-batched-752/execution.json"
FULL_Q = E / "native-terminal-cycle-931/run-metadata.json"
FULL_Q_VERIFICATION = E / "native-terminal-cycle-review-931/verification.json"
RUNTIME_REFERENCE = LAB / "docs/evidence/human-resting/2026-10-07-native-runtime.json"
PARSER_FIXTURE = E / "dense45-pair-004/fgmres-6s.csv"
STUDY = E / "native-integrated-resting-study-917"
FROZEN_LAB_REV = "d550d8ad88a76fdee5bb6e028286fd24e962571f"
FROZEN_HUMAN_REV = "b354949c258106d00f2bd3dd6ac91216d5a3d409"
FROZEN_BRAIN_REV = "a1cf7218fae5d26f9aed9d845047fc6c472ad596"
FINAL_SCENE_936_ROOT = E / "final-native-scene-preflight-936"
V014_FINAL_SCENE_936_SCRIPT = FINAL_SCENE_936_ROOT / "v014/final_scene_936-v014.py"
V014_FINAL_SCENE_936_SCRIPT_SHA = "8972ddba951766714637fb58621e113797a6a8b3f934dd1bb526abd6c2aec496"
V014_FINAL_SCENE_936_TESTS = FINAL_SCENE_936_ROOT / "v014/test_final_scene_936_v014.py"
V014_FINAL_SCENE_936_TESTS_SHA = "1c42c78a05097370d78b4158155ead9e0b57fe274aeb7d84fb2d3003df58594c"
FINAL_SCENE_936_SCRIPT = FINAL_SCENE_936_ROOT / "v015/final_scene_936-v015.py"
FINAL_SCENE_936_SCRIPT_SHA = "9828b90ac3bca4a4e91068df6128bbde829fb0dc1f2d9e1d874ddc07af965eac"
FINAL_SCENE_936_TESTS = FINAL_SCENE_936_ROOT / "v015/test_final_scene_936_v015.py"
FINAL_SCENE_936_TESTS_SHA = "b0fad9475835f478db685c9c7c73ea11596931b2dcd9452aaa7ceba85e8130bf"
FINAL_SCENE_936_REVISION = FINAL_SCENE_936_ROOT / "v015/revision-015.json"
FINAL_SCENE_936_REVISION_SHA = "7b1960ddf5a5f0c652f954fb9844456750a74a9cc7007581809ce1df91bb666e"
FINAL_SCENE_936_README = FINAL_SCENE_936_ROOT / "v015/README-v015.md"
FINAL_SCENE_936_README_SHA = "f5b2396078a00bb355a2e4ead3b8f5b89e7e0ceefae4560841a60c3a34ec44e2"
FINAL_SCENE_936_VIEWER = Path("/Users/n/numi-human-retired-alias-visibility-018")
FINAL_SCENE_936_VIEWER_REV = "41b254605e66035a4e3966e5a1ae790197431605"
VIEWER018_BUILD = Path("/Users/n/numi-human-retired-alias-visibility-build-018-attempt2")
SCENE_NATIVE_BINARY = VIEWER018_BUILD / "bin/numi-human-native"
SCENE_NATIVE_BINARY_SHA = "486442abe2070168a902fb06669469e8b6cc612a9a6ef65ddb3d54957308d906"
SCENE_RESP_METALLIB = VIEWER018_BUILD / "matter/shaders/HumanRespiration.metallib"
SCENE_RESP_METALLIB_SHA = "4b61361f513bf0996d687398498b85ba4e379edca91c36f134e1b0398f31c426"
VIEWER018_BUILD_PINS = VIEWER018_BUILD / "evidence/build-pins.json"
VIEWER018_SOURCE_PINS = VIEWER018_BUILD / "evidence/source-pins.json"
VIEWER018_BUILD_PINS_SHA = "cb883675ac1fc6735c80c49769ea6e0065181b427181bb99ea947e35d6952fdd"
VIEWER018_SOURCE_PINS_SHA = "a2066150f3dedb1c40b64a74c8476bfa9b0bd76b5110a35f018a71859b008c2f"
VIEWER018_BUILD_SCRIPT = E / "native-retired-alias-visibility-018-attempt2/build-native-viewer-018-attempt2.sh"
VIEWER018_BUILD_SCRIPT_SHA = "f82d8646914ea94189cb767b87bbea30bdb3c0fbb9c921066de502a1bc253cee"
VIEWER018_SOURCE_DELTA = E / "native-retired-alias-visibility-018-attempt2/source-delta.patch"
VIEWER018_SOURCE_DELTA_SHA = "e7acace57f332c0ae202f47cd8f8afe59242f7ebbab97499f5cc5cfbc58f52fa"
VIEWER018_NATIVE_BINARY = VIEWER018_BUILD / "bin/numi-human-native"
VIEWER018_NATIVE_BINARY_SHA = "486442abe2070168a902fb06669469e8b6cc612a9a6ef65ddb3d54957308d906"
VIEWER018_RESP_METALLIB = VIEWER018_BUILD / "matter/shaders/HumanRespiration.metallib"
VIEWER018_RESP_METALLIB_SHA = "4b61361f513bf0996d687398498b85ba4e379edca91c36f134e1b0398f31c426"
VIEWER018_BUILD_LOG = E / "native-retired-alias-visibility-018-attempt2/build.log"
VIEWER018_BUILD_LOG_SHA = "b3f933d429df933bea37c56d8468d77539e9d968e8bcc4d3d916e7f450e3ac88"
VIEWER018_TEST_LOG = E / "native-retired-alias-visibility-018-attempt2/focused-test.log"
VIEWER018_TEST_LOG_SHA = "69e933d4e43664579c4adc2ff61a60df21dd53da5efc141028db91714d96024b"
VIEWER018_TEST_BINARY = E / "native-retired-alias-visibility-018-attempt2/numi_human_retired_inspection_layers_test"
VIEWER018_TEST_BINARY_SHA = "7b61202aff36c377e1e511cc94ebf4c191d3552404dab2f91d1f48b4b6ac2c6a"
V014_PREFLIGHT_ROOT = FINAL_SCENE_936_ROOT / "skin-927-lung-924-viewer-018-v014-final-attempt3/native-run"
V014_PREFLIGHT_INVOCATION = V014_PREFLIGHT_ROOT / "invocation.json"
V014_PREFLIGHT_INVOCATION_SHA = "98e4ec56d5fededb8794e515d8c3cd27a8a794a50a8a4778653bb1157ca7e2c9"
V014_PREFLIGHT_METADATA = V014_PREFLIGHT_ROOT / "run-metadata.json"
V014_PREFLIGHT_METADATA_SHA = "3fd01323c05051d093394bc91632134e282bc8b3943488fb551e697ead4db4fa"
V014_PREFLIGHT_LOG = V014_PREFLIGHT_ROOT / "native.log"
V014_PREFLIGHT_LOG_SHA = "842f13feba65008893f831cdabda2379295b693cf01b509487f4a0d9382a7929"
V014_COMPARISON = FINAL_SCENE_936_ROOT / "v014-native-comparison/comparison.json"
V014_COMPARISON_SHA = "64db87dbaefed917204694fd83a4b4c232a17ba440cecbcfaffe3e2e0cc3f933"
FINAL_SCENE_936_HUMAN = Path("/Users/n/numi-human-resting-final-integration-001")
FINAL_SCENE_936_HUMAN_REV = "935cc7d332b9118f41c719e2d2f9c7c568a3d41f"
FINAL_SCENE_936_LAB = Path("/Users/n/numi-human-performance-source-014")
FINAL_SCENE_936_LAB_REV = "d550d8ad88a76fdee5bb6e028286fd24e962571f"
LEGACY_907_SKIN_SHA = "898990d49a3f1dfa0cbe85b10765bf62fa0cc3c834a3371f5503c83facfacba2"
LEGACY_907_MANIFEST_SHA = "f3aab90ecbd0751e984143b9e7a270312d880e09bde803f12e2a0d4bd9073d9c"
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


UINT64_MAX = (1 << 64) - 1


def normalize_identity_uint64(value, label):
    """Normalize owner decimal-string and offline integer fingerprints exactly."""
    if type(value) is int:
        result = value
    elif type(value) is str:
        if not value or not value.isascii() or not value.isdecimal():
            raise ValueError(label + " must be a canonical decimal uint64")
        if value != "0" and value.startswith("0"):
            raise ValueError(label + " must not contain leading zeroes")
        if len(value) > 20:
            raise ValueError(label + " exceeds uint64")
        result = int(value, 10)
    else:
        raise ValueError(label + " must be an integer or decimal string")
    if result < 0 or result > UINT64_MAX:
        raise ValueError(label + " exceeds uint64")
    return result


def normalize_owner_summary_identities(summary):
    """Return owner summary with only its two decimal uint64 IDs normalized."""
    require(isinstance(summary, dict), "native owner summary must be an object")
    normalized = dict(summary)
    for key in ("body_source_fingerprint", "coupled_program_fingerprint"):
        require(key in normalized, "native owner summary omits " + key)
        normalized[key] = normalize_identity_uint64(normalized[key], key)
    return normalized


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


def current_git_identity(repo):
    """Return the canonical current-checkout identity, not a commit diff."""
    repo = Path(repo).resolve()
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--binary"])
    return {
        "repository": str(repo),
        "revision": git(repo, "rev-parse", "HEAD"),
        "status": git(repo, "status", "--porcelain=v1", "--untracked-files=normal"),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


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



def hash_path(path, label):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(),
            label + " must be an absolute non-symlink path: " + str(path))
    if path.is_file():
        return sha(path)
    if path.is_dir():
        files = sorted(q for q in path.rglob("*") if q.is_file() and not q.is_symlink())
        require(bool(files), label + " directory has no regular files: " + str(path))
        manifest = "".join(sha(q) + "  " + str(q) + "\n" for q in files).encode("utf-8")
        return hashlib.sha256(manifest).hexdigest()
    raise ValueError(label + " is missing or not a regular file/directory: " + str(path))


def merge_936_source_hashes(existing, source_files, source_directories):
    """Merge verified regular-file pins; keep directory digests in their own map."""
    require(isinstance(existing, dict) and isinstance(source_files, dict) and
            isinstance(source_directories, dict), "936 source hash inventories must be objects")
    merged = dict(existing)
    for source_path, expected in source_files.items():
        resolved = regular(source_path, "936 report-pinned source/input").resolve()
        observed = sha(resolved)
        require(observed == expected,
                "936 report-pinned source/input changed: " + str(resolved))
        key = str(resolved)
        previous = merged.get(key)
        require(previous in (None, observed),
                "936 source report conflicts with existing source pins: " + key)
        merged[key] = observed
    for source_path, expected in source_directories.items():
        path = Path(source_path)
        require(path.is_absolute() and not path.is_symlink() and path.is_dir(),
                "936 report-pinned source directory must be an absolute non-symlink directory: " + str(path))
        resolved = path.resolve()
        observed = hash_path(resolved, "936 report-pinned source directory")
        require(observed == expected,
                "936 report-pinned source directory changed: " + str(resolved))
        require(str(resolved) not in merged,
                "936 source directory must not enter the owner regular-file hash inventory: " + str(resolved))
    return merged


def verify_viewer018_binding(binding):
    require(isinstance(binding, dict), "936 assembly lacks its viewer018 build binding")
    require(binding.get("build_base_revision") == BUILD_COMPILED_HEAD and
            binding.get("source_tree_clean_at_build") is False and
            binding.get("dirty_source_delta_sha256") == VIEWER018_SOURCE_DELTA_SHA and
            binding.get("current_source_commit") == FINAL_SCENE_936_VIEWER_REV and
            binding.get("current_source_tree_clean") is True and
            binding.get("binary_sha256") == VIEWER018_NATIVE_BINARY_SHA,
            "936 viewer018 binding does not distinguish dirty build inputs from the clean committed source")
    require(binding.get("directory_hash_algorithm") ==
            "SHA-256 over sorted regular-file rows '<sha256>  <absolute path>\n'; symlinks excluded",
            "936 viewer018 directory hash rule changed")
    build_pins_path = regular(VIEWER018_BUILD_PINS, "viewer018 build pins")
    source_pins_path = regular(VIEWER018_SOURCE_PINS, "viewer018 source pins")
    require(sha(build_pins_path) == VIEWER018_BUILD_PINS_SHA and
            sha(source_pins_path) == VIEWER018_SOURCE_PINS_SHA,
            "viewer018 build/source pins changed")
    build_pins = json.loads(build_pins_path.read_text(encoding="utf-8"))
    source_pins = json.loads(source_pins_path.read_text(encoding="utf-8"))
    require(build_pins.get("attempt") == 2 and
            build_pins.get("source_revision") == BUILD_COMPILED_HEAD and
            build_pins.get("branch") == "codex/retired-alias-layer-visibility-018" and
            build_pins.get("parent_017_build_pins_sha256") == BUILD_MANIFEST_SHA and
            build_pins.get("parent_017_source_pins_sha256") == BUILD_SOURCE_PINS_SHA,
            "viewer018 build pin metadata does not identify the reviewed attempt2 parentage")
    require(source_pins.get("source_revision") == BUILD_COMPILED_HEAD and
            source_pins.get("source_tree_clean") is False and
            source_pins.get("source_delta_patch_sha256") == VIEWER018_SOURCE_DELTA_SHA and
            source_pins.get("build_script_sha256") == VIEWER018_BUILD_SCRIPT_SHA and
            source_pins.get("compiled_binary_sha256") == VIEWER018_NATIVE_BINARY_SHA and
            source_pins.get("human_respiration_metallib_sha256") == VIEWER018_RESP_METALLIB_SHA and
            source_pins.get("runtime_library_sha256") == LIBMETALROBO_SHA,
            "viewer018 source pins do not disclose the exact build-time source and linked artifacts")
    require(sha(VIEWER018_BUILD_SCRIPT) == VIEWER018_BUILD_SCRIPT_SHA and
            sha(VIEWER018_SOURCE_DELTA) == VIEWER018_SOURCE_DELTA_SHA and
            sha(VIEWER018_NATIVE_BINARY) == VIEWER018_NATIVE_BINARY_SHA and
            sha(VIEWER018_RESP_METALLIB) == VIEWER018_RESP_METALLIB_SHA,
            "viewer018 build script, patch, binary, or metallib changed")
    for path, digest, label in (
        (V014_PREFLIGHT_INVOCATION, V014_PREFLIGHT_INVOCATION_SHA, "v014 native invocation"),
        (V014_PREFLIGHT_METADATA, V014_PREFLIGHT_METADATA_SHA, "v014 native run metadata"),
        (V014_PREFLIGHT_LOG, V014_PREFLIGHT_LOG_SHA, "v014 native log"),
        (V014_COMPARISON, V014_COMPARISON_SHA, "v014-to-931 comparison"),
    ):
        require(sha(path) == digest, label + " changed")
    regression_meta = json.loads(V014_PREFLIGHT_METADATA.read_text(encoding="utf-8"))
    regression_comparison = json.loads(V014_COMPARISON.read_text(encoding="utf-8"))
    regression_log = V014_PREFLIGHT_LOG.read_text(encoding="utf-8", errors="replace")
    expected_retired_record = (
        "resting_retired_inspection_source semantic=51010 stable_id=22 "
        "layer_mask=0 payload_retained=true geometry_retained=true")
    policy = regression_comparison.get("retired_alias_inspection_policy", {})
    poses = regression_comparison.get("captured_86_body_poses", {})
    require(regression_meta.get("exit_code") == 0 and
            regression_meta.get("source_files_changed_during_run") == [] and
            regression_meta.get("loaded_metal_runtime", {}).get("verified") is True and
            regression_comparison.get("status") == "measured_no_parity_assumed" and
            regression_comparison.get("qualification") == "20 s regression only" and
            regression_comparison.get("candidate_observations") == 1250 and
            regression_comparison.get("invocation_contract", {}).get("loaded_runtime_verified") is True and
            regression_comparison.get("terminal_log_proof", {}).get("accepted_steps") == 10000 and
            regression_comparison.get("trace_cadence_comparison", {}).get("all_values_exact") is True and
            policy.get("retired_semantic") == 51010 and
            policy.get("retired_organ_stable_ids") == [22] and
            policy.get("zero_inspection_mask") is True and
            policy.get("payload_retained") is True and policy.get("geometry_retained") is True and
            policy.get("record") == expected_retired_record and
            regression_log.splitlines().count(expected_retired_record) == 1 and
            len(poses) == 8 and all(
                item.get("body_count") == 86 and item.get("body_state_hash_equal") is True and
                item.get("max_position_delta_m") == 0.0 and
                item.get("max_quaternion_component_delta") == 0.0
                for item in poses.values()),
            "bounded v014 native regression no longer proves its recorded viewer/runtime/capture scope")
    require(build_pins.get("artifacts", {}).get(str(VIEWER018_NATIVE_BINARY)) == VIEWER018_NATIVE_BINARY_SHA and
            build_pins.get("artifacts", {}).get(str(VIEWER018_RESP_METALLIB)) == VIEWER018_RESP_METALLIB_SHA and
            build_pins.get("artifacts", {}).get(str(LIBMETALROBO)) == LIBMETALROBO_SHA,
            "viewer018 compiled artifact map differs from the exact binary/runtime")
    source_files = build_pins.get("source_files")
    require(isinstance(source_files, dict) and len(source_files) == 22,
            "viewer018 build pin must cover the 22 file/directory inputs")
    for path, digest in source_files.items():
        require(hash_path(Path(path), "viewer018 compiled source/input") == digest,
                "viewer018 compiled source/input changed: " + str(path))
    expected_checks = {}
    relative_sources = {}
    for key, digest in source_pins.items():
        if not isinstance(key, str):
            continue
        if Path(key).is_absolute():
            expected_checks[key] = hash_path(Path(key), "viewer018 absolute source pin")
        elif "/" in key or key == "CMakeLists.txt":
            path = FINAL_SCENE_936_VIEWER / key
            observed = hash_path(path, "viewer018 relative source pin")
            expected_checks[key] = observed
            relative_sources[key] = observed
            require(source_files.get(str(path.resolve())) == digest,
                    "viewer018 committed source is not byte-equivalent to its compiled pin: " + key)
    require(binding.get("source_pin_path_checks") == expected_checks,
            "936 viewer018 source path checks differ from the actual source-pin manifest")
    require(binding.get("compiled_source_byte_equivalence") == relative_sources,
            "936 viewer018 source-byte equivalence map differs from the committed source checkout")
    require(binding.get("build_source_file_and_directory_pins_verified") == len(source_files) == 22,
            "936 viewer018 did not verify every compiled source/file/directory pin")
    require(git(FINAL_SCENE_936_VIEWER, "rev-parse", "HEAD") == FINAL_SCENE_936_VIEWER_REV and
            git(FINAL_SCENE_936_VIEWER, "status", "--porcelain", "--untracked-files=all") == "",
            "viewer018 source checkout changed after native scene assembly")
    return {
        "source_revision": FINAL_SCENE_936_VIEWER_REV,
        "source_tree_clean": True,
        "build_base_revision": BUILD_COMPILED_HEAD,
        "source_tree_clean_at_build": False,
        "dirty_source_delta_sha256": VIEWER018_SOURCE_DELTA_SHA,
        "build_pins_path": str(build_pins_path.resolve()),
        "build_pins_sha256": VIEWER018_BUILD_PINS_SHA,
        "source_pins_path": str(source_pins_path.resolve()),
        "source_pins_sha256": VIEWER018_SOURCE_PINS_SHA,
        "build_script_path": str(VIEWER018_BUILD_SCRIPT.resolve()),
        "build_script_sha256": VIEWER018_BUILD_SCRIPT_SHA,
        "source_delta_path": str(VIEWER018_SOURCE_DELTA.resolve()),
        "source_delta_sha256": VIEWER018_SOURCE_DELTA_SHA,
        "native_binary_path": str(VIEWER018_NATIVE_BINARY.resolve()),
        "native_binary_sha256": VIEWER018_NATIVE_BINARY_SHA,
        "respiratory_metallib_path": str(VIEWER018_RESP_METALLIB.resolve()),
        "respiratory_metallib_sha256": VIEWER018_RESP_METALLIB_SHA,
        "source_file_and_directory_pin_count": len(source_files),
    }


def validate_selected_scene_inputs(nha_path, nha_sha256, base_receipt_path, base_receipt_sha256,
                                   native_receipt_path, native_receipt_sha256,
                                   respiration_config_path, respiration_config_sha256):
    """Resolve and verify exact caller-selected final anatomy/config inputs."""
    nha = regular(nha_path, "selected NHA payload")
    base_receipt = regular(base_receipt_path, "selected source anatomy receipt")
    native_receipt = regular(native_receipt_path, "selected native anatomy receipt")
    respiration = regular(respiration_config_path, "selected respiration configuration")
    for path, expected, label in (
        (nha, nha_sha256, "selected NHA"),
        (base_receipt, base_receipt_sha256, "selected source anatomy receipt"),
        (native_receipt, native_receipt_sha256, "selected native anatomy receipt"),
        (respiration, respiration_config_sha256, "selected respiration configuration"),
    ):
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) and sha(path) == expected,
                label + " path/hash mismatch")
    for path, label in ((base_receipt, "selected source anatomy receipt"),
                        (native_receipt, "selected native anatomy receipt")):
        doc = read_json(path, label)
        require(doc.get("schema") == "numi.human.resting-anatomy-receipt.v1",
                label + " has an unexpected schema")
        payload = doc.get("payload", {})
        require(Path(payload.get("path", "")).resolve() == nha.resolve() and
                payload.get("sha256") == nha_sha256,
                label + " does not bind the exact selected NHA payload")
    return {
        "nha_path": str(nha), "nha_sha256": nha_sha256,
        "base_receipt_path": str(base_receipt), "base_receipt_sha256": base_receipt_sha256,
        "native_receipt_path": str(native_receipt), "native_receipt_sha256": native_receipt_sha256,
        "respiration_config_path": str(respiration),
        "respiration_config_sha256": respiration_config_sha256,
    }


def respiration_asset_from_assembly(assembly, invocation, selected):
    """Resolve the actual v015 --resting-scene config pair and bind it to the caller selection."""
    owner_cli = assembly.get("owner_cli", {})
    argv = owner_cli.get("resolved_native_argv", [])
    indices = [i for i, item in enumerate(argv) if item == "--resting-scene"]
    require(len(indices) == 1 and indices[0] + 2 < len(argv),
            "v015 assembled native argv lacks the circulation/respiration pair")
    config_path = Path(argv[indices[0] + 2]).resolve()
    config_sha = selected["respiration_config_sha256"]
    reported = assembly.get("respiration_config")
    require(isinstance(reported, dict) and
            Path(reported.get("path", "")).resolve() == config_path and
            reported.get("sha256") == config_sha and
            reported.get("source") == "caller-pinned path/hash override",
            "v015 assembly did not report the exact caller-selected respiration configuration")
    require(str(config_path) == selected["respiration_config_path"] and
            assembly.get("source_sha256", {}).get(str(config_path)) == config_sha and
            owner_cli.get("asset_sha256", {}).get(str(config_path)) == config_sha,
            "selected respiration configuration is absent or inconsistent in the assembly source/asset maps")
    requested = owner_cli.get("requested", {}).get("respiration_config", {})
    require(requested.get("path") == str(config_path) and requested.get("sha256") == config_sha,
            "v015 requested settings do not bind the selected respiration configuration")
    actual_argv = invocation.get("argv", [])
    actual_indices = [i for i, item in enumerate(actual_argv) if item == "--resting-scene"]
    require(len(actual_indices) == 1 and actual_indices[0] + 2 < len(actual_argv) and
            Path(actual_argv[actual_indices[0] + 2]).resolve() == config_path,
            "native invocation respiration path differs from the v015 assembled path")
    require(invocation.get("asset_sha256", {}).get(str(config_path)) == config_sha and
            sha(config_path) == config_sha,
            "native invocation does not bind the exact selected respiration configuration")
    return {str(config_path): config_sha}


def _invocation_asset(invocation, flag, offset=0):
    argv = invocation.get("argv", [])
    indices = [i for i, value in enumerate(argv) if value == flag]
    require(len(indices) == 1 and indices[0] + 1 + offset < len(argv),
            "native invocation lacks unique asset flag " + flag)
    raw_path = argv[indices[0] + 1 + offset]
    path = Path(raw_path).resolve()
    asset_map = invocation.get("asset_sha256", {})
    digest = asset_map.get(raw_path, asset_map.get(str(path)))
    require(path.is_file() and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) and
            sha(path) == digest,
            "native invocation asset is missing or its recorded hash is wrong: " + flag)
    return {"path": str(path), "sha256": digest}


def reference_931_input_context(candidate_invocation, selected):
    """Compare the selected configuration/contact geometry with actual 931 input bytes."""
    reference_path = regular(TERMINAL_931_INVOCATION, "931 native invocation").resolve()
    require(sha(reference_path) == TERMINAL_931_INVOCATION_SHA,
            "retained 931 invocation changed")
    reference = read_json(reference_path, "931 native invocation")
    reference_resp = _invocation_asset(reference, "--resting-scene", 1)
    require(Path(reference_resp["path"]) == TERMINAL_931_RESPIRATION_CONFIG.resolve() and
            reference_resp["sha256"] == TERMINAL_931_RESPIRATION_CONFIG_SHA and
            sha(TERMINAL_931_RESPIRATION_CONFIG) == TERMINAL_931_RESPIRATION_CONFIG_SHA,
            "931 respiration configuration is not the pinned input used by its invocation")
    candidate_resp = _invocation_asset(candidate_invocation, "--resting-scene", 1)
    require(candidate_resp["path"] == selected["respiration_config_path"] and
            candidate_resp["sha256"] == selected["respiration_config_sha256"],
            "candidate invocation does not use its caller-pinned respiration configuration")

    role_flags = {
        "circulation_configuration": ("--resting-scene", 0),
        "respiration_configuration": ("--resting-scene", 1),
        "skin_contact_geometry": ("--skin-payload", 0),
        "bed_support_geometry": ("--support-contact-payload", 0),
        "soft_tissue_geometry": ("--soft-tissue-payload", 0),
        "thorax_anatomy_geometry": ("--torso-anatomy-payload", 0),
        "anatomy_receipt": ("--resting-anatomy-receipt", 0),
    }
    comparisons = {}
    for role, (flag, offset) in role_flags.items():
        ref_asset = _invocation_asset(reference, flag, offset)
        candidate_asset = _invocation_asset(candidate_invocation, flag, offset)
        comparisons[role] = {
            "reference": ref_asset,
            "candidate": candidate_asset,
            "byte_identical": ref_asset["sha256"] == candidate_asset["sha256"],
        }
    contact_roles = ("skin_contact_geometry", "bed_support_geometry",
                     "soft_tissue_geometry", "thorax_anatomy_geometry")
    config_roles = ("circulation_configuration", "respiration_configuration")
    contact_identical = all(comparisons[key]["byte_identical"] for key in contact_roles)
    config_identical = all(comparisons[key]["byte_identical"] for key in config_roles)
    respiration_identical = comparisons["respiration_configuration"]["byte_identical"]
    scalar_options = {
        "--muscle-step-seconds": 1,
        "--muscle-step-count": 1,
        "--root-pose": 7,
        "--resting-reference-mass-kg": 1,
        "--muscle-activation": 1,
        "--stand-contact-iterations": 1,
        "--persistent-metal-stand": 0,
        "--vascular-dense45": 0,
        "--resting-release-initialization": 0,
        "--resting-rigid-hands": 0,
    }
    scalar_comparisons = {}
    for flag, width in scalar_options.items():
        def option_values(invocation):
            argv = invocation.get("argv", [])
            positions = [i for i, value in enumerate(argv) if value == flag]
            if width == 0:
                return {"present": len(positions) == 1}
            require(len(positions) == 1 and positions[0] + width < len(argv),
                    "native invocation lacks unique setting " + flag)
            return {"values": argv[positions[0] + 1:positions[0] + 1 + width]}
        ref_value = option_values(reference)
        candidate_value = option_values(candidate_invocation)
        scalar_comparisons[flag] = {
            "reference": ref_value,
            "candidate": candidate_value,
            "identical": ref_value == candidate_value,
        }
    scalar_identical = all(item["identical"] for item in scalar_comparisons.values())
    reference_env = reference.get("environment", {})
    candidate_env = candidate_invocation.get("environment", {})
    environment_keys = sorted(
        key for key in set(reference_env) | set(candidate_env)
        if key.startswith(("NUMI_", "DYLD_")) and
        key not in ("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT",
                    "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS", "DYLD_PRINT_LIBRARIES")
    )
    environment_comparisons = {
        key: {"reference": reference_env.get(key),
              "candidate": candidate_env.get(key),
              "identical": reference_env.get(key) == candidate_env.get(key)}
        for key in environment_keys
    }
    environment_identical = all(item["identical"] for item in environment_comparisons.values())
    return {
        "reference_invocation_path": str(reference_path),
        "reference_invocation_sha256": TERMINAL_931_INVOCATION_SHA,
        "reference_run_path": str(E / "native-terminal-cycle-931"),
        "reference_accepted_steps": 10000,
        "reference_inputs": comparisons,
        "runtime_scalar_settings": scalar_comparisons,
        "runtime_environment_settings": environment_comparisons,
        "runtime_scalar_settings_identical_to_931": scalar_identical,
        "runtime_environment_settings_identical_to_931": environment_identical,
        "respiration_configuration_byte_identical_to_931": respiration_identical,
        "all_mechanical_configuration_bytes_identical_to_931": config_identical,
        "all_contact_driving_geometry_bytes_identical_to_931": contact_identical,
        "all_compared_physical_inputs_identical_to_931": (
            config_identical and contact_identical and scalar_identical and environment_identical),
        "interpretation": (
            "If the selected mechanical configurations, runtime scalar settings, relevant runtime environment, "
            "and contact-driving geometry are identical to 931, any changed trace or pose requires regression "
            "investigation. If a selected configuration or "
            "geometry asset differs, retain all measured deltas and assess them against that declared input; "
            "finite differences alone do not establish physiological acceptance."
        ),
        "scope": "Read-only comparison of actual invocation-bound input bytes; no physiology or anatomy pass is implied.",
    }


def validate_v015_preflight_comparison(comparison_path, comparison_sha256, scene_dir):
    """Require measurable, finite 20 s differences without requiring 931 physiology parity."""
    path = regular(comparison_path, "v015 measured preflight comparison")
    require(path.is_relative_to(E) and sha(path) == comparison_sha256,
            "v015 comparison must be hash-bound inside evidence")
    report = read_json(path, "v015 measured preflight comparison")
    candidate_trace = regular(Path(scene_dir) / "resting-coupled.csv", "v015 accepted physiology trace")
    require(report.get("schema") == "numi.human.final-20s-vs-931-comparison.v1" and
            report.get("status") == "measured_no_parity_assumed" and
            report.get("qualification") == "20 s regression only" and
            Path(report.get("reference", "")).resolve() == (E / "native-terminal-cycle-931").resolve() and
            Path(report.get("candidate", "")).resolve() == Path(scene_dir).resolve() and
            report.get("candidate_csv_sha256") == sha(candidate_trace) and
            report.get("candidate_observations") == 1250 and report.get("reference_roots") == 10000,
            "v015 comparison is not bound to this accepted 20 s run and the declared 931 reference")
    cadence = report.get("trace_cadence_comparison", {})
    require(cadence.get("reference_root_count") == 10000 and
            cadence.get("reference_rows") == 10000 and cadence.get("candidate_rows") == 1250 and
            cadence.get("segment_steps") == 8 and
            cadence.get("candidate_accepted_steps") == list(range(8, 10001, 8)) and
            isinstance(cadence.get("all_values_exact"), bool),
            "v015 trace comparison does not preserve the accepted q0/COM8 cadence contract")
    columns = cadence.get("columns", {})
    require(len(columns) == 60 and set(columns) and
            all(item.get("samples") == 1250 and
                all(math.isfinite(float(item.get(key, float("nan"))))
                    for key in ("max_abs_delta", "mean_signed_delta"))
                for item in columns.values()),
            "v015 comparison omits or contains invalid numerical field differences")
    terminal = report.get("terminal_log_proof", {})
    require(terminal.get("accepted_steps") == 10000 and
            terminal.get("terminal_presentations") == 1 and
            terminal.get("terminal_state_advanced_physics") is False and
            math.isclose(float(terminal.get("float32_dt_s", float("nan"))),
                         NATIVE_FLOAT_DT_S, rel_tol=0.0, abs_tol=1e-15) and
            math.isclose(float(terminal.get("actual_simulated_s", float("nan"))),
                         10000 * NATIVE_FLOAT_DT_S, rel_tol=0.0, abs_tol=1e-12),
            "v015 comparison lacks the exact accepted terminal/no-advance clock proof")
    poses = report.get("captured_86_body_poses", {})
    expected_pose_steps = {"0", "4991", "5375", "5759", "6111", "6495", "7743", "10000"}
    require(set(poses) == expected_pose_steps and all(
        item.get("body_count") == 86 and
        math.isfinite(float(item.get("max_position_delta_m", float("nan")))) and
        math.isfinite(float(item.get("max_quaternion_component_delta", float("nan"))))
        for item in poses.values()),
        "v015 comparison lacks finite accepted-state body pose comparisons")
    require(report.get("invocation_contract", {}).get("loaded_runtime_verified") is True,
            "v015 comparison does not prove the loaded native runtime")
    # all_values_exact is reported as an observation; changed-configuration differences are expected inputs to interpretation.
    return {
        "path": str(path), "sha256": sha(path),
        "reference": report["reference"], "candidate": report["candidate"],
        "cadence_all_values_exact": cadence["all_values_exact"],
        "column_differences": columns,
        "captured_body_pose_differences": poses,
        "terminal_log_proof": terminal,
        "loaded_runtime_verified": True,
        "scope": "Measured 20 s regression only; parameter-dependent physiology/pose differences are reported, not gated to 931 equality."
    }


def verify_final_scene_assembly(scene_dir, invocation, selected):
    """Bind the native preflight to its exact non-fixture 936 scene assembly."""
    scene_dir = Path(scene_dir).resolve()
    expected_receipt = Path(selected["native_receipt_path"]).resolve()
    assembly_path = regular(scene_dir.parent / "assembly-preflight.json",
                            "adjacent 936 assembly preflight")
    require(sha(FINAL_SCENE_936_SCRIPT) == FINAL_SCENE_936_SCRIPT_SHA and
            sha(FINAL_SCENE_936_TESTS) == FINAL_SCENE_936_TESTS_SHA and
            sha(FINAL_SCENE_936_REVISION) == FINAL_SCENE_936_REVISION_SHA and
            sha(FINAL_SCENE_936_README) == FINAL_SCENE_936_README_SHA,
            "936 v015 scene-preparation source/test/revision changed")
    report = read_json(assembly_path, "936 assembly preflight")
    require(report.get("preflight_script_sha256") == FINAL_SCENE_936_SCRIPT_SHA,
            "936 report does not bind the exact v015 scene-preparation source")
    require(report.get("schema") == "numi.human.final-native-scene-preflight.v1" and
            report.get("status") == "assembled_owner_cli_validated_native_not_run" and
            report.get("native_run") is False and
            report.get("native_command_executable") is True and
            report.get("asset_role") ==
            "caller-pinned NHSKIN and exact receipt-bound NHA; assembly is not geometry qualification",
            "preflight is not a non-fixture corrected 936 scene assembly")

    raw_source_hashes = report.get("source_sha256")
    require(isinstance(raw_source_hashes, dict) and raw_source_hashes,
            "936 assembly report lacks source and input hash pins")
    source_hashes = {}
    source_directory_hashes = {}
    for source_path, source_digest in raw_source_hashes.items():
        source_file = Path(source_path)
        require(source_file.is_absolute(), "936 pinned source/input path is not absolute")
        observed = hash_path(source_file, "936 pinned source/input")
        require(observed == source_digest,
                "936 source/input hash changed: " + str(source_path))
        if source_file.is_dir():
            source_directory_hashes[str(source_file.resolve())] = observed
            members = sorted(q for q in source_file.rglob("*") if q.is_file() and not q.is_symlink())
            for member in members:
                source_hashes[str(member.resolve())] = sha(member)
        else:
            source_hashes[str(source_file.resolve())] = observed

    def bound_file(path_value, expected_hash, label, source_input=True):
        path = Path(path_value)
        require(path.is_absolute(), label + " path is not absolute")
        path = regular(path, label)
        observed = sha(path)
        require(isinstance(expected_hash, str) and observed == expected_hash,
                label + " hash does not match the 936 report")
        if source_input:
            require(source_hashes.get(str(path.resolve())) == observed,
                    label + " is not included in the 936 source hash map")
        return path, observed

    owners = report.get("owner_checkouts")
    require(isinstance(owners, list), "936 report lacks checkout provenance")
    owner_by_path = {}
    for item in owners:
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            owner_by_path[str(Path(item["path"]).resolve())] = item
    expected_owners = {
        str(FINAL_SCENE_936_HUMAN.resolve()): FINAL_SCENE_936_HUMAN_REV,
        str(FINAL_SCENE_936_LAB.resolve()): FINAL_SCENE_936_LAB_REV,
        str(FINAL_SCENE_936_VIEWER.resolve()): FINAL_SCENE_936_VIEWER_REV,
    }
    require(set(owner_by_path) == set(expected_owners),
            "936 scene source checkouts differ from the reviewed source pair")
    for owner_path, expected_revision in expected_owners.items():
        item = owner_by_path[owner_path]
        require(item.get("head") == expected_revision and item.get("clean") is True,
                "936 checkout report has a changed or dirty source revision: " + owner_path)
        require(git(Path(owner_path), "rev-parse", "HEAD") == expected_revision and
                git(Path(owner_path), "status", "--porcelain", "--untracked-files=all") == "",
                "936 source checkout changed after scene preparation: " + owner_path)
    viewer_binding = verify_viewer018_binding(report.get("viewer_build_binding"))

    composition = report.get("skin_composition", {})
    require(composition.get("used_by_native_cli") is True,
            "936 composed skin receipt was not the receipt used by the native CLI")
    owner_result = composition.get("owner_result", {})
    require(owner_result.get("anatomy_payload_unchanged") is True and
            owner_result.get("functional_bindings_unchanged") is True,
            "936 skin composition does not preserve the base anatomy and functional bindings")
    skin_path, skin_hash = bound_file(
        owner_result.get("candidate_skin_path", ""),
        owner_result.get("candidate_skin_sha256"), "936 candidate skin")
    manifest_path, manifest_hash = bound_file(
        owner_result.get("candidate_manifest_path", ""),
        owner_result.get("candidate_manifest_sha256"), "936 skin registration manifest")
    require(skin_hash != LEGACY_907_SKIN_SHA and
            "common-atlas-skin-composition-907" not in str(skin_path),
            "legacy 907 skin cannot be used as the corrected final candidate")
    require(manifest_hash != LEGACY_907_MANIFEST_SHA and
            "common-atlas-skin-composition-907" not in str(manifest_path),
            "legacy 907 registration manifest cannot be used as the corrected final candidate")
    manifest = read_json(manifest_path, "936 candidate skin registration manifest")
    output_payload = manifest.get("output_payload", {})
    require(manifest.get("schema") == "numi.human.common-atlas-skin-geometry-registration.v1" and
            Path(output_payload.get("path", "")).resolve() == skin_path.resolve() and
            output_payload.get("sha256") == skin_hash,
            "936 skin manifest does not identify the exact candidate payload")
    require(composition.get("composed_skin_sha256") == skin_hash and
            owner_result.get("candidate_skin_sha256") == skin_hash,
            "936 composed skin identity differs from the candidate manifest")

    receipt_path, receipt_hash = bound_file(
        report.get("native_receipt", {}).get("path", ""),
        report.get("native_receipt", {}).get("sha256"), "936 native anatomy receipt")
    require(receipt_path.resolve() == expected_receipt and
            receipt_hash == selected["native_receipt_sha256"],
            "supplied native anatomy receipt differs from the exact v015 scene receipt")
    require(composition.get("composed_receipt_path") == str(receipt_path) and
            composition.get("composed_receipt_sha256") == receipt_hash,
            "936 composition receipt identity differs from the native receipt")
    require(owner_result.get("receipt_path") == str(receipt_path) and
            owner_result.get("receipt_sha256") == receipt_hash,
            "936 owner composition result does not bind its native receipt")
    receipt = read_json(receipt_path, "936 composed anatomy receipt")
    payload = receipt.get("payload", {})
    nha_path, nha_hash = bound_file(
        payload.get("path", ""),
        payload.get("sha256"), "936 anatomy payload")
    require(nha_path.resolve() == Path(selected["nha_path"]).resolve() and
            nha_hash == selected["nha_sha256"] and
            report["native_receipt"].get("NHA_sha256") == nha_hash and
            report.get("base_nha_sha256") == nha_hash and
            composition.get("composition_base_nha_sha256") == nha_hash,
            "v015 composition changed or misreported the exact caller-selected NHA payload")
    base_receipt_path, base_receipt_hash = bound_file(
        composition.get("base_receipt_path", ""),
        composition.get("base_receipt_sha256"), "936 base anatomy receipt")
    require(base_receipt_path.resolve() == Path(selected["base_receipt_path"]).resolve() and
            base_receipt_hash == selected["base_receipt_sha256"],
            "v015 skin composition used a different source anatomy receipt")
    base_receipt = read_json(base_receipt_path, "936 base anatomy receipt")
    require(base_receipt.get("payload", {}).get("sha256") == nha_hash and
            owner_result.get("base_receipt_path") == str(base_receipt_path) and
            owner_result.get("base_receipt_sha256") == base_receipt_hash,
            "936 skin composition base receipt does not match the selected anatomy payload")

    owner_cli = report.get("owner_cli", {})
    require(owner_cli.get("resolved_native_argv") == invocation.get("argv") and
            owner_cli.get("asset_sha256") == invocation.get("asset_sha256"),
            "native invocation differs from the exact 936 assembled argv or assets")
    respiration_assets = respiration_asset_from_assembly(report, invocation, selected)
    requested = owner_cli.get("requested", {})
    require(requested.get("nominal_seconds") == 20 and requested.get("dt") == DT and
            requested.get("steps") == STEP_COUNT and requested.get("q_audit") == 0 and
            requested.get("COM_segment_steps") == 8 and
            requested.get("inspection_tour") is True and
            requested.get("inspection_period_s") == 2.5,
            "936 preflight assembly does not request the admitted 20 s q0/COM8 tour")
    env = invocation.get("environment", {})
    try:
        captures = [int(x) for x in env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"].split(",")]
    except (KeyError, ValueError, AttributeError):
        raise ValueError("936 native preflight does not bind its accepted geometry capture list")
    require(requested.get("captures") == captures and 0 in captures and STEP_COUNT in captures,
            "936 assembly capture settings differ from the native preflight or omit endpoints")

    argv = invocation.get("argv", [])
    def argv_value(flag):
        require(flag in argv and argv.index(flag) + 1 < len(argv),
                "936 native argv is missing " + flag)
        return Path(argv[argv.index(flag) + 1]).resolve()
    require(argv_value("--resting-anatomy-receipt") == receipt_path and
            argv_value("--torso-anatomy-payload") == nha_path,
            "936 native argv does not use the report's exact receipt and anatomy payload")
    require(owner_cli.get("asset_sha256", {}).get(str(skin_path)) == skin_hash and
            owner_cli.get("asset_sha256", {}).get(str(receipt_path)) == receipt_hash and
            owner_cli.get("asset_sha256", {}).get(str(nha_path)) == nha_hash,
            "936 native assets omit the exact candidate skin, receipt, or anatomy payload")

    scene = report.get("scene", {})
    # These are assembled runtime assets. The 936 source_sha256 map records
    # preparation inputs; final scene/support outputs are independently bound
    # by the report scene fields and native owner_cli.asset_sha256 map.
    scene_manifest, scene_manifest_hash = bound_file(
        scene.get("manifest", ""), scene.get("manifest_sha256"),
        "936 resting-scene manifest", source_input=False)
    support_path, support_hash = bound_file(
        scene.get("support_payload", ""), scene.get("support_sha256"),
        "936 initial support-contact payload", source_input=False)
    require(owner_cli["asset_sha256"].get(str(scene_manifest)) == scene_manifest_hash,
            "936 native asset map omits its exact resting-scene manifest")
    require(argv_value("--support-contact-payload") == support_path and
            owner_cli["asset_sha256"].get(str(support_path)) == support_hash,
            "936 native argv/assets do not use the exact assembled support-contact payload")
    try:
        root_pose_index = argv.index("--root-pose")
        actual_pose = [float(x) for x in argv[root_pose_index + 1:root_pose_index + 8]]
        pose = scene.get("root_pose", {})
        expected_pose = (pose.get("root_translation_xyz_m", []) +
                         pose.get("root_delta_quaternion_xyzw", []))
    except (ValueError, IndexError, TypeError):
        raise ValueError("936 native argv lacks its assembled root pose")
    require(len(actual_pose) == len(expected_pose) == 7 and
            all(math.isclose(a, float(b), rel_tol=1e-6, abs_tol=1e-7)
                for a, b in zip(actual_pose, expected_pose)),
            "936 native argv root pose differs from its assembled scene")

    return {
        "path": str(assembly_path.resolve()), "sha256": sha(assembly_path),
        "builder_path": str(FINAL_SCENE_936_SCRIPT),
        "builder_sha256": FINAL_SCENE_936_SCRIPT_SHA,
        "status": report["status"],
        "human_source_revision": FINAL_SCENE_936_HUMAN_REV,
        "lab_source_revision": FINAL_SCENE_936_LAB_REV,
        "viewer_source_revision": FINAL_SCENE_936_VIEWER_REV,
        "viewer_binding": viewer_binding,
        "skin_path": str(skin_path), "skin_sha256": skin_hash,
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
        "base_receipt_path": str(base_receipt_path),
        "base_receipt_sha256": base_receipt_hash,
        "native_receipt_path": str(receipt_path),
        "native_receipt_sha256": receipt_hash,
        "nha_path": str(nha_path), "nha_sha256": nha_hash,
        "base_receipt_path": str(base_receipt_path),
        "base_receipt_sha256": base_receipt_hash,
        "respiration_config_path": selected["respiration_config_path"],
        "respiration_config_sha256": selected["respiration_config_sha256"],
        "respiration_assets": respiration_assets,
        "scene_manifest_path": str(scene_manifest),
        "scene_manifest_sha256": scene_manifest_hash,
        "support_path": str(support_path), "support_sha256": support_hash,
        "source_hashes": dict(source_hashes),
        "source_directory_hashes": dict(source_directory_hashes),
    }


def scene_summary(invocation_path, scene_dir, selected):
    """Validate the accepted 20 s native control against caller-selected scene inputs."""
    scene_dir = Path(scene_dir).resolve()
    inv_path = regular(invocation_path, "control invocation")
    invocation = read_json(inv_path, "control invocation")
    md_path = regular(scene_dir / "run-metadata.json", "control owner run metadata")
    metadata = read_json(md_path, "control owner run metadata")
    log_path = regular(scene_dir / "native.log", "control native log")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    require(metadata.get("exit_code") == 0 and metadata.get("source_files_changed_during_run") == [],
            "control owner run failed or changed source files")
    require(metadata.get("loaded_metal_runtime", {}).get("verified") is True,
            "control run-metadata does not verify its loaded Metal runtime")
    for key in ("argv", "asset_sha256", "environment"):
        require(metadata.get(key) == invocation.get(key),
                "control run-metadata " + key + " does not match invocation")
    summary = normalize_owner_summary_identities(owner.native_scene_summary(log))
    argv = invocation.get("argv", [])
    try:
        requested_steps = int(argv[argv.index("--muscle-step-count") + 1])
        requested_dt = float(argv[argv.index("--muscle-step-seconds") + 1])
    except (ValueError, IndexError, TypeError):
        raise ValueError("control invocation lacks native step-count/timestep flags")
    terminal_line = next((x for x in reversed(log.splitlines())
                          if x.startswith("stand_terminal_state=")), None)
    require(terminal_line is not None, "native control log has no terminal accepted-state record")
    terminal = json.loads(terminal_line.split("=", 1)[1])
    require(summary["accepted_steps"] == requested_steps == STEP_COUNT,
            "control accepted count must exactly equal the predeclared 10,000-root preflight")
    require(requested_dt == DT and
            abs(summary["simulated_s"] - requested_steps * NATIVE_FLOAT_DT_S) <= 1e-5,
            "control accepted horizon does not match 10,000 roots at native Float32 2 ms")
    require(terminal.get("timestep_seconds") in (DT, NATIVE_FLOAT_DT_S) and
            terminal.get("root_assistance") is False,
            "control terminal timestep/assistance evidence is invalid")
    require(summary["device"] == "Apple M4 Pro", "control run is not from the expected Apple M4 Pro")
    try:
        owner.validate_native_310s_invocation(invocation)
    except Exception as exc:
        raise ValueError("control invocation fails frozen 310 s admission: " + str(exc))

    argv_value = lambda flag: argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None
    receipt_path = Path(argv_value("--resting-anatomy-receipt") or "").resolve()
    nha_path = Path(argv_value("--torso-anatomy-payload") or "").resolve()
    config_index = argv.index("--resting-scene") if "--resting-scene" in argv else -1
    require(config_index >= 0 and config_index + 2 < len(argv),
            "control invocation lacks its circulation/respiration scene pair")
    config_path = Path(argv[config_index + 2]).resolve()
    require(receipt_path == Path(selected["native_receipt_path"]).resolve() and
            nha_path == Path(selected["nha_path"]).resolve() and
            config_path == Path(selected["respiration_config_path"]).resolve(),
            "control invocation receipt, NHA, or respiration config differs from caller selection")
    for path, digest, label in (
        (receipt_path, selected["native_receipt_sha256"], "selected native anatomy receipt"),
        (nha_path, selected["nha_sha256"], "selected NHA"),
        (config_path, selected["respiration_config_sha256"], "selected respiration config"),
    ):
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                label + " is missing, symlinked, or changed")
    bindings = invocation.get("asset_sha256", {})
    require(isinstance(bindings, dict) and bool(bindings), "control invocation lacks asset SHA bindings")
    for path, digest in bindings.items():
        bound_path = Path(path)
        require(bound_path.is_absolute() and bound_path.is_file() and not bound_path.is_symlink(),
                "control invocation contains a missing or symlinked bound asset: " + str(path))
        require(sha(bound_path) == digest, "control invocation asset SHA mismatch: " + str(path))
    for path, digest, label in (
        (receipt_path, selected["native_receipt_sha256"], "native receipt"),
        (nha_path, selected["nha_sha256"], "NHA"),
        (config_path, selected["respiration_config_sha256"], "respiration config"),
    ):
        require(str(path) in bindings and bindings[str(path)] == digest,
                "selected " + label + " is not byte-bound by the native invocation")
    require(Path(argv[4]).resolve() == scene_dir,
            "control invocation output directory differs from its preflight directory")
    movie_index = argv.index("--resting-movie") if "--resting-movie" in argv else -1
    require(movie_index >= 0 and movie_index + 1 < len(argv) and
            Path(argv[movie_index + 1]).resolve() == (scene_dir / "native-viewer.mov").resolve(),
            "control invocation movie output differs from its preflight directory")
    require(not (scene_dir / "common-field-failure.json").exists(),
            "control preflight contains a common-field failure receipt")

    binary = Path(argv[0]).resolve()
    require(binary == SCENE_NATIVE_BINARY.resolve() and sha(binary) == SCENE_NATIVE_BINARY_SHA,
            "preflight is not using the exact viewer018 native binary")
    require(str(SCENE_RESP_METALLIB.resolve()) in bindings and
            bindings[str(SCENE_RESP_METALLIB.resolve())] == SCENE_RESP_METALLIB_SHA and
            sha(SCENE_RESP_METALLIB) == SCENE_RESP_METALLIB_SHA,
            "the exact viewer018-linked respiration metallib is not bound")
    require(str(LIBMETALROBO.resolve()) in bindings and
            bindings[str(LIBMETALROBO.resolve())] == LIBMETALROBO_SHA and
            sha(LIBMETALROBO) == LIBMETALROBO_SHA,
            "the exact frozen014 physical MetalRobo library is not bound")

    sys.path.insert(0, str(HUMAN / "src"))
    from numilab_human import resting_run as human_resting_run
    loaded_runtime = human_resting_run.loaded_metal_runtime(log_path, LIBMETALROBO, LIBMETALROBO_SHA)
    require(loaded_runtime.get("verified") is True and
            metadata.get("loaded_metal_runtime") == loaded_runtime,
            "control native log does not independently verify the exact frozen014 runtime image")
    assembly = verify_final_scene_assembly(scene_dir, invocation, selected)
    require(assembly["respiration_config_path"] == str(config_path) and
            assembly["respiration_config_sha256"] == selected["respiration_config_sha256"],
            "scene assembly config differs from accepted control invocation")
    return invocation, metadata, log, summary, requested_steps, requested_dt, assembly


def treatment_probe(scene_dir, probe_dir, control_invocation, control_summary,
                    control_log, control_metadata, assembly, selected):
    scene_dir = Path(scene_dir).resolve()
    probe_dir = Path(probe_dir).resolve()
    require(probe_dir.parent == scene_dir and probe_dir.name == PROGRAM_PROBE_NAME and
            probe_dir.is_dir() and not probe_dir.is_symlink(),
            "treatment identity probe must be the prescribed v015 child of the matched control preflight")
    inv_path = regular(probe_dir / "invocation.json", "direct-native treatment identity-probe invocation")
    md_path = regular(probe_dir / "run-metadata.json", "direct-native treatment identity-probe run metadata")
    log_path = regular(probe_dir / "native.log", "treatment identity-probe native log")
    inv = read_json(inv_path, "direct-native treatment identity-probe invocation")
    md = read_json(md_path, "direct-native treatment identity-probe run metadata")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    recorder_sha = sha(TREATMENT_PROBE_RECORDER)
    require(md.get("exit_code") == 0 and md.get("source_files_changed_during_run") == [] and
            md.get("source_directories_changed_during_run") == [] and
            md.get("asset_files_changed_during_run") == [],
            "treatment identity probe failed or changed pinned source/assets")
    for key in ("argv", "asset_sha256", "environment"):
        require(md.get(key) == inv.get(key), "treatment probe metadata " + key + " mismatch")
    control_metadata = control_metadata or read_json(scene_dir / "run-metadata.json", "control run metadata")
    for key in ("host", "machine", "system", "qualification"):
        require(md.get(key) == control_metadata.get(key) and
                inv.get(key) == control_invocation.get(key) == md.get(key),
                "treatment probe owner/invocation metadata differs from control for " + key)
    require(md.get("loaded_metal_runtime", {}).get("verified") is True and
            md.get("direct_native_probe_validation", {}).get("passed") is True,
            "treatment identity probe lacks successful runtime and recorder acceptance")
    ca = list(control_invocation.get("argv", []))
    ta = list(inv.get("argv", []))
    require(len(ca) > 4 and len(ta) > 4 and Path(ta[4]).resolve() == probe_dir,
            "treatment probe native output is not the exact v015 child")
    movie_idx = ta.index("--resting-movie") if "--resting-movie" in ta else -1
    require(movie_idx >= 0 and movie_idx + 1 < len(ta) and
            Path(ta[movie_idx + 1]).resolve() == (probe_dir / "native-viewer.mov").resolve(),
            "treatment identity probe movie is not retained in its child directory")
    require("--resting-drive-intervention" not in ca and "--resting-drive-intervention" in ta,
            "control/treatment probe intervention placement is invalid")
    k = ta.index("--resting-drive-intervention")
    try:
        values = [float(ta[k + i]) for i in (1, 2, 3)]
    except (ValueError, IndexError):
        raise ValueError("treatment probe must carry start, end, and scale")
    require(values == [60.0, 100.0, 0.5] and k + 4 == len(ta),
            "treatment identity probe must append only the prescribed [60,100,0.5] tuple")
    del ta[k:k + 4]
    ca[4] = ta[4] = "<RUN_OUTPUT>"
    for argv in (ca, ta):
        require("--resting-movie" in argv, "probe native argv lacks movie output")
        index = argv.index("--resting-movie")
        argv[index + 1] = "<MOVIE_OUTPUT>"
    require(ca == ta, "treatment probe differs from its matched control beyond outputs and the intervention")
    require(inv.get("asset_sha256") == control_invocation.get("asset_sha256"),
            "treatment probe asset hashes differ from selected control scene")
    ce = dict(control_invocation.get("environment", {}))
    te = dict(inv.get("environment", {}))
    fail_key = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
    if fail_key in ce or fail_key in te:
        ce[fail_key] = te[fail_key] = "<RUN_FAILURE_RECEIPT>"
    require(ce == te, "treatment probe environment differs beyond its output receipt path")

    direct = md.get("direct_native_probe", {})
    require(direct.get("scope") == (
        "direct native 20 s program-identity probe; configured future intervention is outside the simulated interval; "
        "no treatment dose or physiological response is observed") and
        direct.get("intervention_overlaps_simulated_horizon") is False and
        direct.get("dose_observed") is False and
        direct.get("recorder_sha256") == recorder_sha and
        direct.get("identity_algorithm_source_sha256") == sha(PROBE_SOURCE),
        "probe metadata does not carry the exact v015 identity-only recorder/source scope")
    require(direct.get("selected_scene_inputs") == selected,
            "treatment probe does not bind the caller-selected NHA/receipt/respiration inputs")
    require(not (probe_dir / "common-field-failure.json").exists(),
            "treatment identity probe contains a common-field failure receipt")
    require(direct.get("control_invocation_sha256") == sha(scene_dir / "invocation.json") and
            direct.get("control_assembly_sha256") == assembly["sha256"] and
            direct.get("control_run_metadata_sha256") == sha(scene_dir / "run-metadata.json") and
            direct.get("control_native_log_sha256") == sha(scene_dir / "native.log") and
            direct.get("source_directory_hashes") == assembly["source_directory_hashes"],
            "treatment identity probe is not bound to this exact control assembly/run and source directories")

    control_runtime = control_metadata.get("loaded_metal_runtime", {})
    probe_runtime = md.get("loaded_metal_runtime", {})
    require(probe_runtime.get("verified") is True and
            probe_runtime.get("expected_path") == control_runtime.get("expected_path") == str(LIBMETALROBO.resolve()) and
            probe_runtime.get("expected_sha256") == control_runtime.get("expected_sha256") == LIBMETALROBO_SHA,
            "treatment identity probe loaded a different physical runtime")

    summary = normalize_owner_summary_identities(owner.native_scene_summary(log))
    try:
        steps = int(inv["argv"][inv["argv"].index("--muscle-step-count") + 1])
        dt = float(inv["argv"][inv["argv"].index("--muscle-step-seconds") + 1])
    except (ValueError, IndexError, TypeError):
        raise ValueError("treatment identity probe lacks step-count/timestep flags")
    terminal_line = next((line for line in reversed(log.splitlines())
                          if line.startswith("stand_terminal_state=")), None)
    require(terminal_line is not None, "treatment identity probe lacks terminal accepted-state record")
    terminal = json.loads(terminal_line.split("=", 1)[1])
    require(steps == STEP_COUNT and dt == DT and summary["accepted_steps"] == steps and
            abs(summary["simulated_s"] - steps * NATIVE_FLOAT_DT_S) <= 1e-5 and
            terminal.get("timestep_seconds") in (DT, NATIVE_FLOAT_DT_S) and
            terminal.get("root_assistance") is False,
            "treatment identity probe must match exact 20 s / 2 ms accepted-state horizon")
    require(summary["world_fingerprint"] == control_summary["world_fingerprint"] and
            summary["device"] == control_summary["device"],
            "treatment probe world/device identity differs from control")
    ids = program_ids(control_log, SCENE_RESP_METALLIB)
    require(summary["body_source_fingerprint"] ==
            ids["treatment_body_source_fingerprint_predicted"] and
            summary["coupled_program_fingerprint"] ==
            ids["treatment_program_fingerprint_predicted"],
            "native treatment identity probe disagrees with the independent uint64 prediction")
    return summary, ids, probe_dir, {
        "invocation_path": str(inv_path), "invocation_sha256": sha(inv_path),
        "metadata_path": str(md_path), "metadata_sha256": sha(md_path),
        "native_log_path": str(log_path), "native_log_sha256": sha(log_path),
        "recorder_path": str(TREATMENT_PROBE_RECORDER),
        "recorder_sha256": recorder_sha,
        "tests_path": str(TREATMENT_PROBE_TESTS), "tests_sha256": sha(TREATMENT_PROBE_TESTS),
        "direct_native_probe": direct,
    }

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
        FINAL_SCENE_936_SCRIPT, FINAL_SCENE_936_TESTS, FINAL_SCENE_936_REVISION,
        FINAL_SCENE_936_README,
        VIEWER018_BUILD_PINS, VIEWER018_SOURCE_PINS, VIEWER018_BUILD_SCRIPT,
        VIEWER018_SOURCE_DELTA, VIEWER018_NATIVE_BINARY, VIEWER018_RESP_METALLIB,
        VIEWER018_BUILD_LOG, VIEWER018_TEST_LOG, VIEWER018_TEST_BINARY,
        V014_PREFLIGHT_INVOCATION, V014_PREFLIGHT_METADATA, V014_PREFLIGHT_LOG,
        V014_COMPARISON,
        BUILD_SOURCE / "CMakeLists.txt",
        BUILD_SOURCE / "apps/NumiHumanRestingVisual.hpp",
        BUILD_SOURCE / "apps/NumiHumanRestingCoupling.hpp",
        BUILD_SOURCE / "apps/NumiHumanAcceptedGeometryCadence.hpp",
        BUILD_SOURCE / "apps/numilab_human_myosim_visual_probe.mm",
        BUILD_SOURCE / "matter/src/human_respiration.metal",
        PROBE_SOURCE, RUNTIME_SOURCE,
        CARDIAC_911, HUMAN_RESTING_RUN, TERMINAL_931_NATIVE_LOG,
        TERMINAL_931_INVOCATION, TERMINAL_931_RESPIRATION_CONFIG,
        E / "native-source-state-cycle-914/invocation.json",
        E / "native-source-state-cycle-914/run-metadata.json",
        E / "native-source-state-cycle-review-914/verification.json",
        E / "native-terminal-capture-review-930/verification-v2.json",
        E / "native-terminal-cycle-review-931/verification.json", TERMINAL_Q0_COM8_932,
        READINESS_SCRIPT, READINESS_DEFAULT_SCRIPT, READINESS_VERSIONED_SCRIPT,
        READINESS_ANALYZER, READINESS_README, READINESS_REVISION, V014_PROBE_PREVIEW,
        SCENE_LINEAGE_TESTS, READINESS_ROOT / "test_final_pair_viewer_binding_v016.py",
        TREATMENT_PROBE_RECORDER, TREATMENT_PROBE_TESTS, V015_ASSEMBLY_TESTS,
        P12_IDENTITY_TESTS, P12_REVISION_TESTS, P12_TEST_LOG,
        PREPARATION_OWNER_TEST_LOG,
        CAPTURE_PLAN_TESTS, OWNER_LINEAGE_TESTS
    ]
    for path in additions:
        q = regular(path, "compiled source/build provenance input")
        hashes[str(q)] = sha(q)
    build_source_identity = current_git_identity(BUILD_SOURCE)
    viewer_source_identity = current_git_identity(FINAL_SCENE_936_VIEWER)
    require(build_source_identity["revision"] == BUILD_COMPILED_HEAD and
            build_source_identity["status"] == "",
            "017 current source checkout identity is not clean and exact")
    require(viewer_source_identity["revision"] == FINAL_SCENE_936_VIEWER_REV and
            viewer_source_identity["status"] == "",
            "018 current source checkout identity is not clean and exact")
    revisions = dict(old_revisions)
    revisions["numi-human-terminal-capture-017"] = {
        "path": str(BUILD_SOURCE),
        "revision": BUILD_COMPILED_HEAD,
        "merged_main_revision": BUILD_EVIDENCE_HEAD,
        "diff_sha256": build_source_identity["diff_sha256"],
        "status_porcelain": build_source_identity["status"],
        "compiled_source_commit_diff_sha256": "4a3b02e52f769d25e2dede362f81fd9447c72c4a956afc3b773a4fcd17ff47df",
        "build_pins": str(BUILD_MANIFEST),
        "build_pins_sha256": BUILD_MANIFEST_SHA,
        "source_pins": str(BUILD_SOURCE_PINS),
        "source_pins_sha256": BUILD_SOURCE_PINS_SHA,
        "build_script": str(BUILD_PATCH),
        "build_script_sha256": BUILD_PATCH_SHA,
        "scope": "017 native viewer terminal accepted-state capture; frozen Lab014 physical runtime and guard015 respiratory source remain separately pinned"
    }
    revisions["numi-human-retired-alias-visibility-018"] = {
        "path": str(FINAL_SCENE_936_VIEWER),
        "revision": FINAL_SCENE_936_VIEWER_REV,
        "diff_sha256": viewer_source_identity["diff_sha256"],
        "status_porcelain": viewer_source_identity["status"],
        "build_base_revision": BUILD_COMPILED_HEAD,
        "source_tree_clean_at_build": False,
        "dirty_source_delta_sha256": VIEWER018_SOURCE_DELTA_SHA,
        "build_pins": str(VIEWER018_BUILD_PINS),
        "build_pins_sha256": VIEWER018_BUILD_PINS_SHA,
        "source_pins": str(VIEWER018_SOURCE_PINS),
        "source_pins_sha256": VIEWER018_SOURCE_PINS_SHA,
        "build_script": str(VIEWER018_BUILD_SCRIPT),
        "build_script_sha256": VIEWER018_BUILD_SCRIPT_SHA,
        "binary": str(VIEWER018_NATIVE_BINARY),
        "binary_sha256": VIEWER018_NATIVE_BINARY_SHA,
        "scope": "018 adds receipt-validated retired-organ-alias inspection invisibility; semantic ID collision protection is included; it retains 017 terminal capture and the frozen014 physical/respiratory runtime. Build-time tree was dirty and is disclosed by the exact source-delta patch; current checkout is clean at the committed source revision."
    }
    return hashes, revisions, build_source_identity, viewer_source_identity

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



def json_document(value):
    """Serialize JSON using the one strict formatting rule shared by plan outputs."""
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_new_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json_document(value))


def rewrite_json_document(path, value):
    Path(path).write_text(json_document(value), encoding="utf-8")


def complete_instrument_calibration_bindings(plan, calibration):
    """Bind the final exact instrument artifact set after all lineage additions."""
    require(isinstance(plan, dict) and isinstance(plan.get("instrument"), dict),
            "prepared plan has no instrument object")
    artifacts = plan["instrument"].get("artifacts")
    require(isinstance(artifacts, list) and artifacts and
            all(isinstance(item, str) and item for item in artifacts),
            "instrument artifact list is missing or malformed")
    require(len(artifacts) == len(set(artifacts)),
            "instrument artifact list contains duplicate paths")
    require(isinstance(calibration, dict), "calibration report is not an object")
    bindings = calibration.get("bindings")
    require(isinstance(bindings, dict), "calibration bindings are missing")
    artifact_set = set(artifacts)
    extras = set(bindings) - artifact_set
    require(not extras, "calibration has bindings outside the final instrument: " +
            ", ".join(sorted(extras)))
    for item in artifacts:
        candidate = regular(item, "final instrument artifact")
        digest = sha(candidate)
        prior = bindings.get(item)
        require(prior in (None, digest),
                "calibration binding conflicts with the final instrument artifact: " + item)
        bindings[item] = digest
    require(set(bindings) == artifact_set,
            "calibration bindings do not exactly cover the final instrument artifacts")
    return calibration


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
    trial_by_arm = {}
    for trial_id, arm in (("resting-baseline", "control"),
                          ("resting-drive-half", "treatment")):
        trial = next((x for x in plan.get("trials", []) if x.get("id") == trial_id), None)
        require(trial is not None, "owner plan omitted registered arm " + trial_id)
        argv = trial.get("argv", [])
        require(argv.count("--invocation") == 1 and argv.index("--invocation") + 1 < len(argv),
                trial_id + " argv does not have exactly one invocation path")
        argv[argv.index("--invocation") + 1] = template_paths[arm]
        trial_by_arm[arm] = trial
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
    arm_identity_paths = {}
    arm_identity_hashes = {}
    arm_runtime_bindings = {}
    for arm in ("control", "treatment"):
        derived_invocation = templates[arm]
        derived_resolution = owner.native_scene_runtime_dependency_bindings(derived_invocation)
        require(len(derived_resolution) == 7,
                arm + " launch template did not resolve all seven runtime dependencies")
        arm_identity = dict(identity)
        arm_identity["native_invocation"] = {
            "path": template_paths[arm], "sha256": template_hashes[arm],
            "asset_sha256": derived_invocation.get("asset_sha256")}
        arm_identity["runtime_dependency_resolution"] = derived_resolution
        arm_identity["derived_launch_template_binding"] = {
            "arm": arm, "template_path": template_paths[arm],
            "template_sha256": template_hashes[arm],
            "parent_invocation_path": str(parent_path),
            "parent_invocation_sha256": parent_sha,
            "scope": "Exact arm-specific invocation bytes and resolved runtime dependencies used by run-native."
        }
        arm_identity_path = Path(out) / ("native-build-identity-" + arm + ".json")
        write_new_json(arm_identity_path, arm_identity)
        arm_identity_paths[arm] = str(arm_identity_path.resolve())
        arm_identity_hashes[arm] = sha(arm_identity_path)
        arm_runtime_bindings[arm] = derived_resolution
        argv = trial_by_arm[arm]["argv"]
        for flag, value in (("--native-build-identity", arm_identity_paths[arm]),
                            ("--native-build-identity-sha256", arm_identity_hashes[arm])):
            require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv),
                    arm + " trial must have one prepared native identity flag: " + flag)
            argv[argv.index(flag) + 1] = value
    plan["design"]["accepted_geometry_capture_runtime_binding_paths"] = arm_identity_paths
    plan["design"]["accepted_geometry_capture_runtime_binding_hashes"] = arm_identity_hashes
    identity["accepted_geometry_capture_runtime_bindings"] = {
        arm: {"identity_path": arm_identity_paths[arm], "identity_sha256": arm_identity_hashes[arm],
              "runtime_dependency_resolution": arm_runtime_bindings[arm]}
        for arm in ("control", "treatment")}
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
    calibration_path = Path(out) / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration.setdefault("bindings", {})
    for path in [schedule_path, *(Path(x) for x in template_paths.values()),
                 *(Path(x) for x in arm_identity_paths.values())]:
        calibration["bindings"][str(path.resolve())] = sha(path)
    plan["artifacts"] = list(dict.fromkeys(
        [*plan.get("artifacts", []), str(schedule_path.resolve()),
         *(template_paths[a] for a in ("control", "treatment")),
         *(arm_identity_paths[a] for a in ("control", "treatment"))]))
    plan["instrument"]["artifacts"] = list(dict.fromkeys(
        [*plan["instrument"].get("artifacts", []), str(schedule_path.resolve()),
         *(template_paths[a] for a in ("control", "treatment")),
         *(arm_identity_paths[a] for a in ("control", "treatment"))]))
    for item in plan["instrument"]["artifacts"]:
        candidate = Path(item)
        if candidate.is_file() and not candidate.is_symlink():
            calibration["bindings"][str(candidate.resolve())] = sha(candidate)
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                encoding="utf-8")
    return schedule, template_paths, template_hashes, parent_path, parent_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-preflight-dir", required=True,
                        help="completed v015 native control preflight directory")
    parser.add_argument("--nha", required=True, help="caller-selected final NHANATOMY payload")
    parser.add_argument("--nha-sha", required=True, help="exact SHA-256 of the selected NHANATOMY")
    parser.add_argument("--base-anatomy-receipt", required=True,
                        help="source receipt that binds the selected NHA before skin composition")
    parser.add_argument("--base-anatomy-receipt-sha", required=True,
                        help="exact SHA-256 of the selected source receipt")
    parser.add_argument("--native-anatomy-receipt", required=True,
                        help="skin-composed receipt used by the native control invocation")
    parser.add_argument("--native-anatomy-receipt-sha", required=True,
                        help="exact SHA-256 of the selected composed native receipt")
    parser.add_argument("--respiration-config", required=True,
                        help="caller-selected respiration configuration used by v015 assembly")
    parser.add_argument("--respiration-config-sha", required=True,
                        help="exact SHA-256 of the selected respiration configuration")
    parser.add_argument("--preflight-comparison", required=True,
                        help="measured v015 20 s comparison against the retained 931 reference")
    parser.add_argument("--preflight-comparison-sha", required=True,
                        help="exact SHA-256 of the measured comparison report")
    parser.add_argument("--treatment-probe-dir",
                        help="successful v015 direct-native identity probe; defaults to the prescribed child")
    parser.add_argument("--draft-dir", type=Path, required=True,
                        help="new science-v2 plan draft directory; this script never registers or runs it")
    args = parser.parse_args()
    require(sys.byteorder == "little", "offline source identity reconstruction requires Apple arm64 little-endian")
    require(not Path(args.draft_dir).exists(), "draft output exists; do not overwrite prior plan evidence")
    require(not STUDY.exists(), "final study directory already exists; select a new study path before registration")
    require(SCIENCE.is_file() and PY39.is_file() and OWNER.is_file(),
            "frozen Lab CLI/runtime or pinned preparation owner is missing")
    preparation_owner_head = git(PREPARATION_OWNER_ROOT, "rev-parse", "HEAD")
    preparation_owner_status = git(PREPARATION_OWNER_ROOT, "status", "--porcelain=v1")
    require(preparation_owner_head == PREPARATION_OWNER_REV and not preparation_owner_status,
            "modern full-q preparation owner is not the exact clean committed Lab revision")
    require(sha(OWNER) == PREPARATION_OWNER_SHA and sha(PREPARATION_OWNER_TEST) == PREPARATION_OWNER_TEST_SHA,
            "modern full-q preparation owner source/test bytes differ from the pinned commit")
    preparation_owner_identity = {
        "repository": str(PREPARATION_OWNER_ROOT), "revision": preparation_owner_head,
        "source_path": str(OWNER.resolve()), "source_sha256": sha(OWNER),
        "test_path": str(PREPARATION_OWNER_TEST.resolve()), "test_sha256": sha(PREPARATION_OWNER_TEST),
        "working_tree_status": preparation_owner_status,
        "scope": "Science-plan preparation/reference admission only; frozen L14 physical runtime and 017 viewer binary remain separately pinned.",
    }
    manifest, lab_head, human_head, brain_head = verify_build_and_pins()
    scene_input = Path(args.scene_preflight_dir).expanduser()
    require(not scene_input.is_symlink(), "corrected scene preflight directory must not be a symlink")
    scene_dir = scene_input.resolve()
    require(scene_dir.is_dir(), "corrected scene preflight directory is missing")
    selected = validate_selected_scene_inputs(
        args.nha, args.nha_sha,
        args.base_anatomy_receipt, args.base_anatomy_receipt_sha,
        args.native_anatomy_receipt, args.native_anatomy_receipt_sha,
        args.respiration_config, args.respiration_config_sha)
    receipt = Path(selected["native_receipt_path"])
    comparison = validate_v015_preflight_comparison(
        args.preflight_comparison, args.preflight_comparison_sha,
        scene_dir)
    control_invocation, control_metadata, control_log, control_summary, steps, dt, assembly = scene_summary(
        scene_dir / "invocation.json", scene_dir, selected)
    input_context = reference_931_input_context(control_invocation, selected)
    comparison["input_byte_comparison_to_931"] = input_context
    ids = program_ids(control_log, SCENE_RESP_METALLIB)
    probe_input = Path(args.treatment_probe_dir).expanduser() if args.treatment_probe_dir else scene_dir / PROGRAM_PROBE_NAME
    require(not probe_input.is_symlink(), "treatment probe directory must not be a symlink")
    probe_dir = probe_input.resolve()
    treatment_summary, ids, probe_dir, probe_evidence = treatment_probe(
        scene_dir, probe_dir, control_invocation, control_summary, control_log,
        control_metadata, assembly, selected)
    require(treatment_summary["coupled_program_fingerprint"] == ids["treatment_program_fingerprint_predicted"],
            "treatment program identity probe mismatch")

    hashes, revisions, build_source_identity, viewer_source_identity = source_pin_files(manifest)
    hashes = merge_936_source_hashes(
        hashes, assembly["source_hashes"], assembly["source_directory_hashes"])
    dynamic_pinned_paths = [
        Path(probe_evidence["invocation_path"]), Path(probe_evidence["metadata_path"]),
        Path(probe_evidence["native_log_path"]), Path(comparison["path"]),
        Path(selected["nha_path"]), Path(selected["base_receipt_path"]),
        Path(selected["native_receipt_path"]), Path(selected["respiration_config_path"]),
    ]
    for pinned_path in (
        *dynamic_pinned_paths, TREATMENT_PROBE_RECORDER, TREATMENT_PROBE_TESTS,
        V015_ASSEMBLY_TESTS, P12_IDENTITY_TESTS, P12_REVISION_TESTS, P12_TEST_LOG,
        PREPARATION_OWNER_TEST_LOG, PREPARATION_OWNER, PREPARATION_OWNER_TEST,
        Path(assembly["path"]), FINAL_SCENE_936_SCRIPT, FINAL_SCENE_936_REVISION,
        FINAL_SCENE_936_README, FINAL_SCENE_936_TESTS,
        Path(assembly["skin_path"]), Path(assembly["manifest_path"]),
        Path(assembly["base_receipt_path"]), Path(assembly["native_receipt_path"]),
        Path(assembly["nha_path"]), Path(assembly["scene_manifest_path"]),
        Path(assembly["support_path"]), FINAL_SCENE_936_TESTS,
        VIEWER018_BUILD_PINS, VIEWER018_SOURCE_PINS, VIEWER018_BUILD_SCRIPT,
        VIEWER018_SOURCE_DELTA, VIEWER018_NATIVE_BINARY, VIEWER018_RESP_METALLIB,
    ):
        resolved = regular(pinned_path, "corrected 936 scene lineage input").resolve()
        digest = sha(resolved)
        previous = hashes.get(str(resolved))
        require(previous in (None, digest),
                "936 corrected scene input conflicts with an existing source hash: " + str(resolved))
        hashes[str(resolved)] = digest
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
        "--full-q-verification", str(FULL_Q_VERIFICATION),
        "--runtime-correctness-reference", str(RUNTIME_REFERENCE)
    ]
    proc = subprocess.run(args_list, cwd=str(LAB), capture_output=True, text=True, check=False)
    if proc.returncode:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise ValueError("frozen owner failed to prepare the science-v2 plan; retained source pin files")
    plan_path = out / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["design"]["preparation_owner"] = preparation_owner_identity
    plan["artifacts"] = list(dict.fromkeys([
        *plan.get("artifacts", []), str(OWNER.resolve()), str(PREPARATION_OWNER_TEST.resolve())
    ]))
    if isinstance(plan.get("instrument"), dict):
        plan["instrument"]["artifacts"] = list(dict.fromkeys([
            *plan["instrument"].get("artifacts", []), str(OWNER.resolve()),
            str(PREPARATION_OWNER_TEST.resolve())
        ]))
    native_identity_path = out / "native-build-identity.json"
    native_identity = json.loads(native_identity_path.read_text(encoding="utf-8"))
    native_identity["preparation_owner"] = preparation_owner_identity
    rewrite_json_document(native_identity_path, native_identity)
    (capture_schedule, capture_template_paths, capture_template_hashes,
     parent_invocation_path, parent_invocation_sha) = bind_accepted_geometry_capture_plan(
        plan, out, scene_dir, control_invocation)
    scene_lineage_artifacts = [
        probe_evidence["recorder_path"], probe_evidence["tests_path"],
        probe_evidence["invocation_path"], probe_evidence["metadata_path"], probe_evidence["native_log_path"],
        comparison["path"], *(str(x) for x in dynamic_pinned_paths),
        str(PREPARATION_OWNER_TEST_LOG.resolve()),
        str(FULL_Q.resolve()), str(FULL_Q_VERIFICATION.resolve()), str(RUNTIME_REFERENCE.resolve()),
        assembly["path"], assembly["builder_path"], str(FINAL_SCENE_936_TESTS.resolve()),
        str(FINAL_SCENE_936_REVISION.resolve()), str(FINAL_SCENE_936_README.resolve()),
        assembly["skin_path"], assembly["manifest_path"], assembly["base_receipt_path"],
        assembly["native_receipt_path"], assembly["nha_path"],
        assembly["scene_manifest_path"], assembly["support_path"],
        assembly["viewer_binding"]["build_pins_path"],
        assembly["viewer_binding"]["source_pins_path"],
        assembly["viewer_binding"]["build_script_path"],
        assembly["viewer_binding"]["source_delta_path"],
        assembly["viewer_binding"]["native_binary_path"],
        assembly["viewer_binding"]["respiratory_metallib_path"],
        str(V014_PREFLIGHT_INVOCATION), str(V014_PREFLIGHT_METADATA),
        str(V014_PREFLIGHT_LOG), str(V014_COMPARISON),
    ]
    plan["artifacts"] = list(dict.fromkeys([
        *plan.get("artifacts", []), str(CARDIAC_911), *scene_lineage_artifacts
    ]))
    plan["instrument"]["artifacts"] = list(dict.fromkeys([
        *plan["instrument"].get("artifacts", []), str(CARDIAC_911), *scene_lineage_artifacts
    ]))
    plan["design"]["current_source_preflight"] = {
        "scene_assembly_report": assembly["path"],
        "scene_assembly_report_sha256": assembly["sha256"],
        "scene_assembly_builder": assembly["builder_path"],
        "scene_assembly_builder_sha256": assembly["builder_sha256"],
        "scene_assembly_status": assembly["status"],
        "scene_source_revisions": {
            "human": assembly["human_source_revision"],
            "lab": assembly["lab_source_revision"],
            "viewer": assembly["viewer_source_revision"]
        },
        "viewer018_build_binding": assembly["viewer_binding"],
        "source_directory_hashes": assembly["source_directory_hashes"],
        "caller_selected_inputs": selected,
        "input_byte_comparison_to_931": input_context,
        "v015_preflight_comparison": comparison,
        "modern_full_q_reference": {
            "run_metadata_path": str(FULL_Q.resolve()),
            "run_metadata_sha256": sha(FULL_Q),
            "verification_path": str(FULL_Q_VERIFICATION.resolve()),
            "verification_sha256": sha(FULL_Q_VERIFICATION),
            "runtime_correctness_reference_path": str(RUNTIME_REFERENCE.resolve()),
            "scope": "10000-root 2 ms exact-q reference with terminal query verification; separate from the planned 310 s paired study.",
        },
        "v009_prior_v014_924_preflight_preview": {
            "path": str(V014_PROBE_PREVIEW.resolve()),
            "sha256": sha(V014_PROBE_PREVIEW),
            "scope": "CPU-only probe-command preview against the completed 927-skin/924-NHA scene; planned_not_run; no draft directory created"
        },
        "viewer018_bounded_native_regression": {
            "invocation_path": str(V014_PREFLIGHT_INVOCATION),
            "invocation_sha256": V014_PREFLIGHT_INVOCATION_SHA,
            "run_metadata_path": str(V014_PREFLIGHT_METADATA),
            "run_metadata_sha256": V014_PREFLIGHT_METADATA_SHA,
            "native_log_path": str(V014_PREFLIGHT_LOG),
            "native_log_sha256": V014_PREFLIGHT_LOG_SHA,
            "comparison_path": str(V014_COMPARISON),
            "comparison_sha256": V014_COMPARISON_SHA,
            "scope": "historical completed 20 s 927-skin/924-NHA viewer/capture/runtime regression; not the current selected anatomy/config and not final anatomy qualification"
        },
        "corrected_skin_path": assembly["skin_path"],
        "corrected_skin_sha256": assembly["skin_sha256"],
        "skin_registration_manifest_path": assembly["manifest_path"],
        "skin_registration_manifest_sha256": assembly["manifest_sha256"],
        "composition_base_receipt_path": assembly["base_receipt_path"],
        "composition_base_receipt_sha256": assembly["base_receipt_sha256"],
        "native_receipt_path": assembly["native_receipt_path"],
        "native_receipt_sha256": assembly["native_receipt_sha256"],
        "current_nha_path": assembly["nha_path"],
        "current_nha_sha256": assembly["nha_sha256"],
        "resting_scene_manifest_path": assembly["scene_manifest_path"],
        "resting_scene_manifest_sha256": assembly["scene_manifest_sha256"],
        "support_contact_path": assembly["support_path"],
        "support_contact_sha256": assembly["support_sha256"],
        "native_owner_run_directory": str(scene_dir),
        "control_invocation_sha256": sha(scene_dir / "invocation.json"),
        "control_run_metadata_sha256": sha(scene_dir / "run-metadata.json"),
        "treatment_identity_probe_directory": str(probe_dir),
        "treatment_identity_probe_invocation_sha256": sha(probe_dir / "invocation.json"),
        "treatment_identity_probe_run_metadata_sha256": sha(probe_dir / "run-metadata.json"),
        "treatment_identity_probe_scope": json.loads(
            (probe_dir / "run-metadata.json").read_text(encoding="utf-8")
        )["direct_native_probe"],
        "treatment_identity_probe_execution_lineage": {
            "execution_package": "package-v016",
            "preparation_package": "package-v016",
            "recorder_path": probe_evidence["recorder_path"],
            "recorder_sha256": probe_evidence["recorder_sha256"],
            "invocation_path": probe_evidence["invocation_path"],
            "invocation_sha256": probe_evidence["invocation_sha256"],
            "run_metadata_path": probe_evidence["metadata_path"],
            "run_metadata_sha256": probe_evidence["metadata_sha256"],
            "native_log_path": probe_evidence["native_log_path"],
            "native_log_sha256": probe_evidence["native_log_sha256"],
            "scope": "completed v015 direct-native identity-only probe bound to this selected anatomy/config and its exact control preflight; no treatment dose or response is observed"
        },
        "treatment_identity_probe_recorder": str(TREATMENT_PROBE_RECORDER),
        "treatment_identity_probe_recorder_sha256": sha(TREATMENT_PROBE_RECORDER),
        "corrected_anatomy_receipt": selected["native_receipt_path"],
        "corrected_anatomy_receipt_sha256": selected["native_receipt_sha256"],
        "corrected_scene_lineage_policy": "Adjacent non-fixture 936 assembly report binds candidate NHSKIN/manifest, owner-composed receipt, dynamic NHA, native argv/assets, and source revisions; no 907/924 payload hash is assumed.",
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
            "source_diff_sha256": build_source_identity["diff_sha256"],
            "compiled_source_commit_diff_sha256": "4a3b02e52f769d25e2dede362f81fd9447c72c4a956afc3b773a4fcd17ff47df",
            "build_script": str(BUILD_PATCH),
            "build_script_sha256": BUILD_PATCH_SHA,
            "native_binary": str(NATIVE_BINARY),
            "native_binary_sha256": EXPECTED_BINARY_SHA,
            "physical_library": str(LIBMETALROBO),
            "physical_library_sha256": LIBMETALROBO_SHA,
            "respiratory_metallib": str(RESP_METALLIB),
            "respiratory_metallib_sha256": RESP_METALLIB_SHA,
            "scope": "017 provides exact accepted-state terminal geometry publication. It links the frozen014 physical MetalRobo library and guard015 respiratory source; the 017 build-pins record identifies capture code and artifacts, not a new physics qualification.",
            "viewer_018_overlay": {
                "source_revision": FINAL_SCENE_936_VIEWER_REV,
                "source_tree_clean": True,
                "source_current_worktree_diff_sha256": viewer_source_identity["diff_sha256"],
                "build_base_revision": BUILD_COMPILED_HEAD,
                "source_tree_clean_at_build": False,
                "dirty_source_delta_sha256": VIEWER018_SOURCE_DELTA_SHA,
                "build_pins": str(VIEWER018_BUILD_PINS),
                "build_pins_sha256": VIEWER018_BUILD_PINS_SHA,
                "source_pins": str(VIEWER018_SOURCE_PINS),
                "source_pins_sha256": VIEWER018_SOURCE_PINS_SHA,
                "native_binary": str(VIEWER018_NATIVE_BINARY),
                "native_binary_sha256": VIEWER018_NATIVE_BINARY_SHA,
                "scope": "receipt-validated retired-organ-alias inspection invisibility; exact 017 terminal capture behavior is retained, and no new physics qualification is claimed"
            }
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
        "The 20 s owner preflight checks the caller-selected assets/configuration and program identities only; it is not "
        "310 s endurance, physiological validation, or whole-body anatomy acceptance. "
        "Cardiac interface localization 911 reports 11,568 current source-neutral RA/RV intersections near "
        "the common-map tricuspid leaflet projection with a localized source-to-current RV residual up to 0.75 mm. This "
        "supports a junction-localized segmentation-overlap interpretation but establishes neither a 3D "
        "leaflet surface nor valve-plane/orifice ownership. Reduced-order CVSim chambers and valves remain "
        "the sole functional and blood owner; this localization is a stated anatomy limitation, not a "
        "geometry-clearance pass.")
    readiness_docs = [str(READINESS_SCRIPT), str(READINESS_ANALYZER), str(READINESS_README),
                      str(READINESS_REVISION), str(SCENE_LINEAGE_TESTS),
                      str(P12_IDENTITY_TESTS), str(P12_REVISION_TESTS), str(P12_TEST_LOG),
                      str(PREPARATION_OWNER_TEST_LOG), str(CAPTURE_PLAN_TESTS),
                      str(OWNER_LINEAGE_TESTS), str(TREATMENT_PROBE_TESTS),
                      str(CALIBRATION_BINDING_TESTS), str(V015_ASSEMBLY_TESTS),
                      str(FINAL_SCENE_936_REVISION), str(FINAL_SCENE_936_README)]
    for path in readiness_docs:
        require(Path(path).is_file() and not Path(path).is_symlink(),
                "readiness document is missing or symlinked: " + path)
    plan["artifacts"] = list(dict.fromkeys([*plan["artifacts"], str(BUILD_MANIFEST), str(BUILD_SOURCE_PINS),
        str(BUILD_FOCUSED_TESTS), str(BUILD_PATCH),
        str(E / "native-source-state-cycle-914/run-metadata.json"),
        str(E / "native-source-state-cycle-review-914/verification.json"),
        str(E / "native-terminal-capture-review-930/verification-v2.json"),
        str(E / "native-terminal-cycle-review-931/verification.json"),
        str(TERMINAL_931_INVOCATION), str(TERMINAL_931_RESPIRATION_CONFIG),
        str(TERMINAL_Q0_COM8_932), *readiness_docs]))
    plan["instrument"]["artifacts"] = list(dict.fromkeys([*plan["instrument"]["artifacts"],
        str(BUILD_MANIFEST), str(BUILD_SOURCE_PINS), str(BUILD_FOCUSED_TESTS), str(BUILD_PATCH),
        str(E / "native-source-state-cycle-914/run-metadata.json"),
        str(E / "native-source-state-cycle-review-914/verification.json"),
        str(E / "native-terminal-capture-review-930/verification-v2.json"),
        str(E / "native-terminal-cycle-review-931/verification.json"),
        str(TERMINAL_931_INVOCATION), str(TERMINAL_931_RESPIRATION_CONFIG),
        str(TERMINAL_Q0_COM8_932), *readiness_docs]))
    calibration_path = Path(plan["instrument"]["calibration"])
    calibration = read_json(calibration_path, "final instrument calibration")
    calibration = complete_instrument_calibration_bindings(plan, calibration)
    rewrite_json_document(calibration_path, calibration)
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
