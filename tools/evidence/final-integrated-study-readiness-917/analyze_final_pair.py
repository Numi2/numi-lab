#!/usr/bin/env python3
"""Read-only supplemental analysis for a completed, registered final integrated pair.

Runs the frozen Lab notebook verifier and reuses completion primitives from
the 855 analyzer, with a strict lineage gate for readiness-derived capture
templates. It writes only to a new output directory outside the study. It does
not simulate, modify the study, or redefine the registered estimand.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

EVIDENCE = Path("/Users/n/numi-human-resting-evidence-20261005")
STUDY_DEFAULT = EVIDENCE / "native-integrated-resting-study-917"
LAB = Path("/Users/n/numi-human-performance-source-014")
SCIENCE_CLI = LAB / "tools/numi"
ANALYZER_PATH = EVIDENCE / "native-paired-physiology-audit-855/analyze_completed_trial.py"
PREPARE_PATH = Path(__file__).resolve().with_name("prepare_final_plan.py")
EXPECTED_REVISION = "d550d8ad88a76fdee5bb6e028286fd24e962571f"
CARDIAC_911 = EVIDENCE / "native-cardiac-interface-localization-911/final-localization-report.json"
CARDIAC_911_SHA256 = "9c468f7a1da58563387acc9712f845a794f07fa3121c7fa4a3a7ae905ecf03e4"

spec = importlib.util.spec_from_file_location("completed_trial_audit", str(ANALYZER_PATH))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
prepare_spec = importlib.util.spec_from_file_location("final_plan_readiness", str(PREPARE_PATH))
prepare = importlib.util.module_from_spec(prepare_spec)
prepare_spec.loader.exec_module(prepare)
HUMAN_SOURCE_PATH = str(prepare.HUMAN / "src")
if HUMAN_SOURCE_PATH not in sys.path:
    sys.path.insert(0, HUMAN_SOURCE_PATH)
from numilab_human.resting_run import loaded_metal_runtime as human_loaded_metal_runtime

OWNER_OBSERVATION_SCHEMA = "numi.human-resting.intervention-observation.v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sealed(path, label):
    doc = audit.read_json(path)
    payload = doc.get("payload")
    require(isinstance(payload, dict), label + " has no payload")
    require(doc.get("sha256") == audit.canonical_payload_hash(payload), label + " payload hash mismatch")
    return doc, payload


def argv_value(argv, flag):
    value = audit.getopt(argv, flag)
    if value is None:
        raise ValueError("registered argv is missing " + flag)
    return value


def verify_owner_scene_invocation_binding(scene_invocation, observation, invocation_path):
    """Match frozen owner serialization: both outputs hash the --invocation file itself."""
    expected_sha = audit.sha256(Path(invocation_path).resolve())
    require(scene_invocation.get("reference_invocation_sha256") == expected_sha,
            "native scene invocation does not bind the derived launch-template file")
    if observation is not None:
        require(observation.get("reference_invocation_sha256") == expected_sha,
                "owner observation does not bind the derived launch-template file")
    return expected_sha


def verify_registered_scene_loaded_runtime(native_log, parent_metadata, parent_invocation):
    """Recheck each arm log with frozen Human helper and the parent-verified library identity."""
    parent_runtime = parent_metadata.get("loaded_metal_runtime", {})
    expected_path = prepare.LIBMETALROBO.resolve()
    expected_sha = prepare.LIBMETALROBO_SHA
    require(parent_runtime.get("verified") is True and
            Path(parent_runtime.get("expected_path", "")).resolve() == expected_path and
            parent_runtime.get("expected_sha256") == expected_sha,
            "verified parent runtime does not bind the pinned physical library")
    bound_assets = parent_invocation.get("asset_sha256", {})
    require(bound_assets.get(str(expected_path)) == expected_sha,
            "parent invocation does not bind the pinned loaded MetalRobo library")
    proof = human_loaded_metal_runtime(Path(native_log), expected_path, expected_sha)
    require(proof.get("verified") is True,
            "registered native scene did not load exactly the pinned MetalRobo image")
    require(proof.get("expected_path") == str(expected_path) and
            proof.get("expected_sha256") == expected_sha and
            len(proof.get("observed_images", [])) == 1 and
            proof["observed_images"][0].get("path") == str(expected_path),
            "registered scene loaded-runtime proof differs from the pinned path/hash")
    return proof


def finite(r, key):
    x = audit.n(r, key)
    if x is None:
        raise ValueError("missing/nonfinite trace field " + key)
    return x


def window_mean(rows, key, lo, hi):
    xs = [finite(r, key) for r in rows if lo <= finite(r, "time_s") < hi]
    require(xs, "no " + key + " samples in [%.3f, %.3f)" % (lo, hi))
    return statistics.fmean(xs), len(xs)


def finite_stats(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    return {"n": len(xs), "mean": statistics.fmean(xs), "min": min(xs), "max": max(xs)}


def derived_capture_trial(trial, schedule, arm):
    """Completion gate for arm-specific launch templates derived from a verified short preflight."""
    trial = trial.resolve()
    scene = trial / "output" / "scene"
    csv_path = scene / "resting-coupled.csv"
    require(csv_path.is_file(), "missing registered physiology trace: " + str(csv_path))
    study = trial.parent.parent
    reg_doc, registration = audit.read_wrapped(study / "registration.json", "study registration")
    started_doc, started = audit.read_wrapped(trial / "started.json", "started receipt")
    process_doc, process = audit.read_wrapped(trial / "process.json", "process receipt")
    receipt_doc, receipt = audit.read_wrapped(trial / "receipt.json", "registered exit receipt")
    require(started.get("registration_sha256") == reg_doc.get("sha256"),
            "started receipt is not bound to this registration")
    require(started_doc.get("sha256") == process.get("binding") == receipt.get("started_sha256"),
            "started/process/exit receipts do not bind the same trial")
    require(process_doc.get("sha256") == receipt.get("process_sha256"),
            "exit receipt does not bind the recorded process")
    trial_obj = started.get("trial", {})
    plan = registration.get("plan", {})
    require(registration.get("schema") == "numi.science.registration.v2" and
            plan.get("schema") == "numi.science.plan.v2",
            "unexpected registered science schema")
    registered = next((x for x in plan.get("trials", []) if x.get("id") == trial_obj.get("id")), None)
    require(registered is not None and trial_obj == registered,
            "started trial differs from the registered plan")
    argv = started.get("argv", [])
    expected_argv = [str(x).replace("{run}", str(started.get("cwd", "")))
                     for x in registered.get("argv", [])]
    require(argv == expected_argv and receipt.get("argv") == argv and
            receipt.get("cwd") == started.get("cwd"),
            "started/exit argv differs from the registered arm")
    require(receipt.get("environment") == started.get("environment"),
            "registered exit environment differs from the started process")
    require(receipt.get("returncode") == 0 and receipt.get("failure") is None and
            receipt.get("ended_at"),
            "registered exit receipt is failed, killed, or missing completion time")
    for key, identity_key in (("pid", "identity"), ("runner_pid", "runner_identity")):
        if audit.live_same(process.get(key), process.get(identity_key)):
            raise RuntimeError("recorded native/study process still active: " + str(process.get(key)))
    steps = int(argv_value(argv, "--steps"))
    dt = float(argv_value(argv, "--dt"))
    require(steps == prepare.FINAL_ACCEPTED_STEPS and abs(dt - prepare.FINAL_DT_S) <= 1e-12,
            "registered arm is not exactly 155000 accepted roots at 2 ms")
    receipt_files = receipt.get("files", {})
    required = ("scene/resting-coupled.csv", "scene/native.log", "scene/invocation.json",
                "scene/intervention-observation.json", "stdout.json", "stderr.log")
    for rel in required:
        require(rel in receipt_files, "registered exit receipt omits " + rel)
        path = trial / "output" / rel
        require(path.is_file() and not path.is_symlink() and audit.sha256(path) == receipt_files[rel],
                "registered exit receipt file hash mismatch: " + rel)

    launch_path = Path(argv_value(argv, "--invocation")).resolve()
    launch = audit.read_json(launch_path)
    lineage = launch.get("readiness_derived_capture_launch_template")
    require(isinstance(lineage, dict) and lineage.get("schema") == prepare.CAPTURE_TEMPLATE_SCHEMA,
            "registered native launch is not a readiness-derived capture template")
    require(lineage.get("arm") == arm, "capture launch template arm differs from registered trial")
    parent_path = Path(lineage.get("parent_invocation_path", "")).resolve()
    require(parent_path.is_file() and not parent_path.is_symlink(),
            "derived launch template parent invocation is missing or symlinked")
    parent_sha = audit.sha256(parent_path)
    require(parent_sha == lineage.get("parent_invocation_sha256"),
            "derived launch template parent invocation hash differs")
    parent = audit.read_json(parent_path)
    expected_launch = prepare.make_capture_launch_template(
        parent, parent_path, parent_sha, schedule, arm)
    require(launch == expected_launch,
            "derived launch template includes an unapproved argv/environment/source change")
    require(lineage.get("capture_environment_value") ==
            schedule["arms"][arm]["environment_value"],
            "launch template capture list differs from the registered schedule")

    owner_meta_path = parent_path.with_name("run-metadata.json")
    owner_log_path = parent_path.with_name("native.log")
    require(owner_meta_path.is_file() and not owner_meta_path.is_symlink() and
            owner_log_path.is_file() and not owner_log_path.is_symlink(),
            "derived launch template parent lacks completed owner metadata/native log")
    owner_meta = audit.read_json(owner_meta_path)
    require(owner_meta.get("exit_code") == 0 and
            owner_meta.get("loaded_metal_runtime", {}).get("verified") is True and
            owner_meta.get("source_files_changed_during_run") == [] and
            owner_meta.get("argv") == parent.get("argv") and
            owner_meta.get("asset_sha256") == parent.get("asset_sha256") and
            owner_meta.get("environment") == parent.get("environment"),
            "parent owner run-metadata does not verify its exact short preflight invocation")
    pre_steps = int(audit.getopt(parent.get("argv", []), "--muscle-step-count") or 0)
    pre_dt = float(audit.getopt(parent.get("argv", []), "--muscle-step-seconds") or 0)
    preterminal, prebody = audit.terminal_from_log(owner_log_path)
    audit.completion_gate(owner_meta, preterminal, None, pre_steps, pre_dt, require_registered=False)
    require(pre_steps == 10000 and abs(pre_dt - 0.002) <= 1e-12 and prebody and
            "physiology_body_clock=matched" in prebody and "root_assistance=false" in prebody,
            "parent owner preflight is not the completed 10k-root unassisted control")

    scene_inv_path = scene / "invocation.json"
    scene_inv = audit.read_json(scene_inv_path)
    require(scene_inv.get("asset_sha256") == parent.get("asset_sha256"),
            "native scene assets differ from the parent preflight")
    expected_intervention = arm == "treatment"
    expected_scene_argv = prepare.owner.native_scene_command(
        launch, scene, argparse.Namespace(steps=steps, dt=dt, arm=arm,
                                          start_s=60.0, end_s=100.0, scale=0.5))
    require(scene_inv.get("argv") == expected_scene_argv,
            "completed native scene argv differs from the exact parent/template plus registered arm settings")
    scene_argv = scene_inv.get("argv", [])
    require(Path(scene_argv[0]).resolve() == prepare.NATIVE_BINARY.resolve() and
            int(audit.getopt(scene_argv, "--muscle-step-count")) == steps and
            abs(float(audit.getopt(scene_argv, "--muscle-step-seconds")) - dt) <= 1e-12,
            "completed native scene does not use pinned binary and exact accepted horizon")
    if expected_intervention:
        require(audit.getopt(scene_argv, "--resting-drive-intervention") == "60.0" and
                scene_argv[scene_argv.index("--resting-drive-intervention") + 2:] == ["100.0", "0.5"],
                "treatment native scene does not use the registered [60,100) half-drive intervention")
    else:
        require("--resting-drive-intervention" not in scene_argv,
                "control native scene unexpectedly includes a drive intervention")
    expected_scene_env = dict(launch["environment"])
    failure_key = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
    if expected_scene_env.get(failure_key):
        expected_scene_env[failure_key] = str(scene / "common-field-failure.json")
    require(scene_inv.get("environment") == expected_scene_env,
            "native arm environment differs from the authorized derived capture environment")
    terminal, body = audit.terminal_from_log(scene / "native.log")
    audit.completion_gate(owner_meta, terminal, receipt, steps, dt)
    loaded_runtime = verify_registered_scene_loaded_runtime(
        scene / "native.log", owner_meta, parent)
    require(body and "physiology_body_clock=matched" in body and
            "root_assistance=false" in body,
            "final native log lacks a completed matched-clock unassisted summary")
    observation = audit.read_json(scene / "intervention-observation.json")
    template_sha = verify_owner_scene_invocation_binding(scene_inv, observation, launch_path)
    stdout = audit.read_json(trial / "output" / "stdout.json")
    require(observation == stdout and
            observation.get("schema") == OWNER_OBSERVATION_SCHEMA and
            observation.get("accepted_steps") == steps and observation.get("duration_valid") is True,
            "registered stdout differs from exact completed owner observation")
    trial_id = trial_obj.get("id")
    arm_expected = "control" if trial_id == "resting-baseline" else "treatment"
    require(arm == arm_expected, "registered trial ID has the wrong capture arm")

    analysis_inv = dict(scene_inv)
    analysis_inv["source"] = registration.get("source", {})
    return {"csv": csv_path, "inv_path": scene_inv_path, "inv": analysis_inv,
            "trial": trial, "started": started_doc, "steps": steps, "dt": dt,
            "owner_metadata": owner_meta, "terminal": terminal, "receipt": receipt,
            "owner_result": observation, "registered": True,
            "scene_loaded_metal_runtime": loaded_runtime, "launch_template": launch,
            "launch_template_path": launch_path, "owner_preflight_path": parent_path}


def accepted_geometry_summary(item, schedule, arm):
    scene = item["trial"] / "output" / "scene"
    geometry_dir = scene / "accepted-geometry"
    require(geometry_dir.is_dir() and not geometry_dir.is_symlink(),
            arm + " accepted-geometry directory is missing or symlinked")
    expected_steps = schedule["arms"][arm]["step_ids"]
    found_receipts = sorted(geometry_dir.glob("step-*.receipt.json"))
    found_steps = []
    for path in found_receipts:
        try:
            found_steps.append(int(path.name[len("step-"):-len(".receipt.json")]))
        except ValueError:
            raise ValueError(arm + " accepted-geometry directory has a malformed receipt name")
    require(sorted(found_steps) == sorted(expected_steps),
            arm + " accepted-geometry receipt set differs from the exact eight-frame plan")
    files = item["receipt"].get("files", {})
    captures = []
    for step in expected_steps:
        stem = "step-" + str(step)
        receipt_path = geometry_dir / (stem + ".receipt.json")
        pack_path = geometry_dir / (stem + ".mrvpack")
        rel_receipt = "scene/accepted-geometry/" + receipt_path.name
        rel_pack = "scene/accepted-geometry/" + pack_path.name
        require(rel_receipt in files and rel_pack in files,
                arm + " registered exit receipt omits accepted geometry files for step " + str(step))
        require(receipt_path.is_file() and not receipt_path.is_symlink() and
                audit.sha256(receipt_path) == files[rel_receipt],
                arm + " accepted geometry receipt does not match registered exit receipt")
        require(pack_path.is_file() and not pack_path.is_symlink() and
                audit.sha256(pack_path) == files[rel_pack],
                arm + " accepted geometry pack does not match registered exit receipt")
        geometry_receipt = audit.read_json(receipt_path)
        require(geometry_receipt.get("schema") == "numi.human.accepted-render-geometry.v1",
                arm + " geometry receipt has an unexpected schema")
        require(geometry_receipt.get("pack_file_sha256") == audit.sha256(pack_path),
                arm + " geometry receipt does not bind its accepted MRV pack")
        event = prepare.validate_capture_receipt(schedule, arm, geometry_receipt)
        require(geometry_receipt.get("accepted_timestamp_microseconds") ==
                round(event["accepted_time_s"] * 1_000_000),
                arm + " geometry receipt timestamp microseconds disagree with accepted step ID")
        captures.append({**event, "receipt_path": str(receipt_path),
                         "receipt_sha256": audit.sha256(receipt_path),
                         "pack_path": str(pack_path), "pack_sha256": audit.sha256(pack_path),
                         "root_fingerprint": geometry_receipt["accepted_root_fingerprint_hex"],
                         "body_state_sha256": geometry_receipt["accepted_body_state_sha256"],
                         "respiration_state_sha256": geometry_receipt["accepted_respiration_state_sha256"]})
    require(item["steps"] == prepare.FINAL_ACCEPTED_STEPS and
            any(x["accepted_step"] == prepare.FINAL_ACCEPTED_STEPS and
                x["nominal_time_s"] == 310.0 and
                abs(x["accepted_time_s"] - schedule["terminal"]["accepted_time_s"]) <= 1e-9 and
                x["terminal"] for x in captures),
            arm + " missing true terminal geometry capture at N=155000, nominal t=310 s")
    log = (scene / "native.log").read_text(encoding="utf-8", errors="replace")
    require("resting_terminal_capture_identity=accepted_step_155000" in log,
            arm + " native log lacks explicit true-terminal capture identity")
    require("terminal_physical_steps_advanced=0" in log,
            arm + " terminal capture log does not prove zero additional physical steps")
    return {"arm": arm, "capture_count": len(captures), "captures": captures,
            "terminal_step": prepare.FINAL_ACCEPTED_STEPS,
            "terminal_nominal_time_s": 310.0,
            "terminal_accepted_time_s": schedule["terminal"]["accepted_time_s"],
            "interpretation": "Seven prior-867-derived ordinary presentation endpoints plus exact accepted terminal N. These sparse samples do not prove all unobserved anatomical states."}


def support_summary(path, receipt_files, duration):
    rel = "scene/resting-com-momentum-diagnostic.csv"
    require(rel in receipt_files, "registered exit receipt omits " + rel)
    require(audit.sha256(path) == receipt_files[rel], "registered support diagnostic hash mismatch")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    post = []
    for row in rows:
        t = audit.n(row, "time_s")
        if t is not None and 10.0 <= t < duration:
            post.append(row)
    require(len(post) >= 100, "support/COM diagnostic lacks post-initialization coverage")
    post.sort(key=lambda r: finite(r, "time_s"))
    ts = [finite(r, "time_s") for r in post]
    xyz = [[finite(r, "com_" + axis + "_m") for r in post] for axis in ("x", "y", "z")]
    first = [axis[0] for axis in xyz]
    last = [axis[-1] for axis in xyz]
    delta = [last[i] - first[i] for i in range(3)]
    mean_t = statistics.fmean(ts)
    denom = math.fsum((t - mean_t) ** 2 for t in ts)
    slopes = []
    for vals in xyz:
        mean_x = statistics.fmean(vals)
        slopes.append(math.fsum((t - mean_t) * (x - mean_x) for t, x in zip(ts, vals)) / denom if denom else 0.0)
    max_radius = max(math.sqrt(math.fsum((xyz[j][i] - first[j]) ** 2 for j in range(3))) for i in range(len(ts)))
    force = finite_stats([audit.n(r, "normal_force_last_physical_step_n") for r in post])
    contacts = finite_stats([audit.n(r, "active_contact_count") for r in post])
    return {
        "samples": len(post), "window_s": [ts[0], ts[-1]],
        "com_endpoint_delta_mm_xyz": [1000.0 * x for x in delta],
        "com_endpoint_displacement_mm": 1000.0 * math.sqrt(math.fsum(x * x for x in delta)),
        "max_com_excursion_from_first_mm": 1000.0 * max_radius,
        "com_linear_trend_mm_per_min_xyz": [60000.0 * x for x in slopes],
        "sampled_aggregate_normal_support_force_last_physical_step_n": force,
        "sampled_active_contact_count": contacts,
        "interpretation": "sampled COM trajectory and the total summed contact-normal impulse divided by one physical-step duration at each sampled endpoint; not a static-equilibrium or postural-qualification test"
    }


def budget_highwater(rows, duration):
    fields = ("respiratory_volume_balance_ml", "oxygen_balance_error_stpd_ml",
              "co2_balance_error_stpd_ml", "blood_error_ml",
              "blood_continuity_residual_accum_ml", "blood_residual_minus_physical_ml",
              "blood_endpoint_minus_physical_ml")
    result = {}
    post = [r for r in rows if 10.0 <= finite(r, "time_s") < duration]
    for key in fields:
        xs = [abs(x) for x in (audit.n(r, key) for r in post) if x is not None]
        if xs:
            result[key] = max(xs)
    return result


def safe_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def json_write(path, obj):
    safe_write(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, default=STUDY_DEFAULT)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "completed-pair-final")
    args = ap.parse_args()
    study = args.study.resolve()
    out = args.output.resolve()
    require(out != study and study not in out.parents, "output must be outside the registered study")
    require(not out.exists(), "output already exists; choose a fresh evidence output directory")
    require(ANALYZER_PATH.is_file(), "855 analyzer is missing")
    require(SCIENCE_CLI.is_file(), "frozen Lab science CLI is missing")
    prepare.verify_build_and_pins()

    # The notebook verifier is read-only. analyze must first have sealed the
    # registered primary result; this wrapper never replaces that authority.
    proc = subprocess.run([str(SCIENCE_CLI), "science", "verify", str(study)],
                          cwd=str(LAB), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ValueError("registered science verify failed: " + (proc.stderr or proc.stdout)[-2000:])
    reg_doc, reg = sealed(study / "registration.json", "study registration")
    plan = reg.get("plan", {})
    require(reg.get("schema") == "numi.science.registration.v2" and plan.get("schema") == "numi.science.plan.v2",
            "unexpected registered study schema")
    require(reg.get("source", {}).get("revision") == EXPECTED_REVISION,
            "registered Lab revision differs from expected frozen revision")
    require(CARDIAC_911.is_file() and audit.sha256(CARDIAC_911) == CARDIAC_911_SHA256,
            "pinned cardiac localization 911 report is missing or changed")
    current_preflight = plan.get("design", {}).get("current_source_preflight", {})
    expected_capture_schedule = prepare.accepted_geometry_capture_schedule()
    capture_schedule = plan.get("design", {}).get("accepted_geometry_capture_schedule")
    require(capture_schedule == expected_capture_schedule,
            "registered geometry capture plan differs from the frozen seven-interior-plus-terminal schedule")
    require(capture_schedule.get("terminal") == expected_capture_schedule["terminal"],
            "registered geometry plan mislabeled its terminal accepted state/time")
    require(current_preflight.get("preflight_accepted_steps") == 10000 and
            current_preflight.get("preflight_dt_s") == 0.002 and
            current_preflight.get("preflight_device") == "Apple M4 Pro" and
            isinstance(current_preflight.get("preflight_world_fingerprint"), int) and
            current_preflight.get("treatment_program_fingerprint", {}).get("matched") is True,
            "registered plan lacks the matched 20 s corrected-asset native program-identity preflight")
    native_provenance = current_preflight.get("compiled_native_provenance", {})
    require(native_provenance.get("build_pins") == str(prepare.BUILD_MANIFEST) and
            native_provenance.get("build_pins_sha256") == prepare.BUILD_MANIFEST_SHA and
            native_provenance.get("source_pins") == str(prepare.BUILD_SOURCE_PINS) and
            native_provenance.get("source_pins_sha256") == prepare.BUILD_SOURCE_PINS_SHA and
            native_provenance.get("focused_tests") == str(prepare.BUILD_FOCUSED_TESTS) and
            native_provenance.get("focused_tests_sha256") == prepare.BUILD_FOCUSED_TESTS_SHA and
            native_provenance.get("source_revision") == prepare.BUILD_COMPILED_HEAD and
            native_provenance.get("merged_main_revision") == prepare.BUILD_EVIDENCE_HEAD and
            native_provenance.get("build_script_sha256") == prepare.BUILD_PATCH_SHA and
            native_provenance.get("native_binary_sha256") == prepare.EXPECTED_BINARY_SHA and
            native_provenance.get("respiratory_metallib_sha256") == prepare.RESP_METALLIB_SHA and
            native_provenance.get("physical_library_sha256") == prepare.LIBMETALROBO_SHA,
            "registered plan lacks the exact 017 terminal-capture build/source/shader identity")
    template_paths = current_preflight.get("accepted_geometry_capture_template_paths", {})
    template_hashes = plan.get("design", {}).get("accepted_geometry_capture_template_hashes", {})
    require(set(template_paths) == {"control", "treatment"} and
            set(template_hashes) == {"control", "treatment"},
            "registered plan lacks both arm-specific derived capture templates")
    require(plan.get("design", {}).get("accepted_geometry_capture_template_paths") == template_paths,
            "registered capture template path map differs from its preflight record")
    for arm, trial_id in (("control", "resting-baseline"), ("treatment", "resting-drive-half")):
        template_path = Path(template_paths[arm]).resolve()
        require(template_path.is_file() and not template_path.is_symlink() and
                audit.sha256(template_path) == template_hashes[arm],
                arm + " derived capture template is missing or changed")
        plan_trial = next((x for x in plan.get("trials", []) if x.get("id") == trial_id), None)
        require(plan_trial is not None and argv_value(plan_trial.get("argv", []), "--invocation") ==
                str(template_path),
                arm + " registered trial does not use its pinned capture template")
        require(str(template_path) in plan.get("artifacts", []) and
                str(template_path) in plan.get("instrument", {}).get("artifacts", []),
                arm + " capture template is not a registered artifact")
    terminal_regressions = (
        (EVIDENCE / "native-terminal-capture-review-930/verification-v2.json",
         "b6d9e92fd6a27f81d58281923808ea45b8e256f2274bfb734b62d4299842886a", "930"),
        (EVIDENCE / "native-terminal-cycle-review-931/verification.json",
         "c9429ec3a10296ead8e6963899c7278736d513f2afeb08d5afb60f167e4fe849", "931"),
        (EVIDENCE / "native-terminal-production-review-932/verification.json",
         "453e8f69bdad311d1d273d5049d17b886b609780bfbe3614700e82c424e782f9", "932 q0/COM8"))
    for terminal_regression, expected_sha, label in terminal_regressions:
        require(terminal_regression.is_file() and audit.sha256(terminal_regression) == expected_sha,
                "pinned " + label + " terminal-capture regression is missing or changed")
        regression = audit.read_json(terminal_regression)
        require(regression.get("pass") is True,
                label + " terminal-capture regression did not pass")
        if label == "932 q0/COM8":
            require(regression.get("q_integration_audit") == 0 and
                    regression.get("com_segment_steps") == 8 and
                    regression.get("candidate_trace_rows") == 8 and
                    regression.get("reference_trace_rows") == 64 and
                    regression.get("identical_endpoint_columns") == 50 and
                    len(regression.get("exact_window_aggregated_columns", [])) == 10,
                    "932 q0/COM8 report no longer proves the bounded reduced-cadence scope")
        require(str(terminal_regression) in plan.get("artifacts", []) and
                str(terminal_regression) in plan.get("instrument", {}).get("artifacts", []),
                "registered plan does not bind terminal-capture regression " + label)
    for pinned in (prepare.BUILD_MANIFEST, prepare.BUILD_SOURCE_PINS, prepare.BUILD_FOCUSED_TESTS,
                   prepare.BUILD_PATCH):
        require(str(pinned) in plan.get("artifacts", []) and
                str(pinned) in plan.get("instrument", {}).get("artifacts", []),
                "registered plan omits a 017 build/source/test pin: " + str(pinned))
        require(pinned.is_file() and audit.sha256(pinned) ==
                current_preflight.get("compiled_native_provenance", {}).get({
                    prepare.BUILD_MANIFEST: "build_pins_sha256",
                    prepare.BUILD_SOURCE_PINS: "source_pins_sha256",
                    prepare.BUILD_FOCUSED_TESTS: "focused_tests_sha256",
                    prepare.BUILD_PATCH: "build_script_sha256"}[pinned]),
                "017 build/source/test pin is missing or changed: " + str(pinned))
    require(str(CARDIAC_911) in plan.get("artifacts", []) and
            str(CARDIAC_911) in plan.get("instrument", {}).get("artifacts", []),
            "registered plan does not bind cardiac localization 911 as a stated limitation")
    require([t.get("id") for t in plan.get("trials", [])] == ["resting-baseline", "resting-drive-half"],
            "registered trial order/ids differ from frozen final pair")
    require(plan.get("prediction", {}).get("minimum") == 0.0 and plan.get("prediction", {}).get("maximum") == 40.0,
            "registered primary sensitivity envelope differs from frozen plan")
    secondary = plan.get("secondary_predictions", {})
    recovery_limits = secondary.get("late_recovery_equivalence", {})
    require(recovery_limits.get("PaCO2_absolute_difference_max_mmhg") == 1.0 and
            recovery_limits.get("PaO2_absolute_difference_max_mmhg") == 5.0 and
            recovery_limits.get("inspiratory_minute_ventilation_relative_difference_max_fraction") == 0.1,
            "registered recovery margins differ from frozen plan")
    official_doc, official = sealed(study / "analysis.json", "registered science analysis")
    require(official.get("registration_sha256") == reg_doc.get("sha256"), "official analysis is not bound to this registration")
    require(len(official.get("paired_differences", [])) == 1, "expected one registered pair")
    official_delta = float(official["paired_differences"][0]["treatment_minus_control"])

    loaded = {}
    rows_by_arm = {}
    cycles = {}
    windows = {}
    support = {}
    capture_reports = {}
    budgets = {}
    reports = {}
    for trial_id, arm in (("resting-baseline", "control"), ("resting-drive-half", "treatment")):
        trial = study / "trials" / trial_id
        item = derived_capture_trial(trial, capture_schedule, arm)
        require(item["inv"].get("source", {}).get("revision") == EXPECTED_REVISION,
                trial_id + " native invocation has wrong source revision")
        steps, dt = audit.horizon(item["inv"], item["started"])
        require(steps == 155000 and abs(dt - 0.002) < 1e-12, trial_id + " horizon differs from registered 310 s at 2 ms")
        obs = item["owner_result"]
        require(obs.get("arm") == arm and obs.get("accepted_steps") == steps and obs.get("duration_valid") is True,
                trial_id + " owner observation does not match completed registered arm/horizon")
        require(obs.get("device") == current_preflight.get("preflight_device") and
                obs.get("world_fingerprint") == current_preflight.get("preflight_world_fingerprint"),
                trial_id + " native observation device/world differs from the corrected preflight")
        native_ids = re.search(
            r"resting_body_source_fingerprint=(\d+) coupled_program_fingerprint=(\d+)",
            (item["trial"] / "output" / "scene" / "native.log").read_text(
                encoding="utf-8", errors="replace"))
        require(native_ids is not None and
                int(native_ids.group(1)) == current_preflight["preflight_body_source_fingerprints"][arm] and
                int(native_ids.group(2)) == current_preflight["preflight_program_fingerprints"][arm],
                trial_id + " final native source/program identity differs from the matched preflight")
        require(obs.get("root_assistance_observed") is False and obs.get("post_initialization_cycle_coverage", {}).get("passed") is True,
                trial_id + " failed unassisted/300 s cycle-coverage evidence")
        if arm == "control":
            require(obs.get("intervention_applied") is False, "control observation reports an applied dose")
        else:
            require(obs.get("intervention_applied") is True and abs(float(obs.get("delivered_drive_scale", -1)) - 0.5) < 1e-12,
                    "treatment observation does not report the registered half-drive dose")
        hdr, rows = audit.load_csv(item["csv"])
        horizon = audit.check_horizon(rows, steps, dt)
        ev = audit.breath_events(rows)
        breaths = audit.respiratory(rows, ev)
        hearts = audit.cardiac(rows)
        args_for_report = argparse.Namespace(expected_source_revision=EXPECTED_REVISION)
        text, detail = audit.report(args_for_report, item["csv"], item["inv_path"], item["inv"],
                                    item["trial"], item["started"], None, rows, horizon, breaths, hearts)
        loaded[arm] = item
        rows_by_arm[arm] = rows
        cycles[arm] = {"breaths": breaths, "hearts": hearts, "horizon": horizon, "detail": detail}
        budgets[arm] = budget_highwater(rows, horizon["duration"])
        receipt_files = item["receipt"].get("files", {})
        capture_reports[arm] = accepted_geometry_summary(item, capture_schedule, arm)
        support_path = trial / "output" / "scene" / "resting-com-momentum-diagnostic.csv"
        support[arm] = support_summary(support_path, receipt_files, horizon["duration"])
        windows[arm] = {}
        argv = item["started"].get("payload", {}).get("argv", [])
        window = float(argv_value(argv, "--window-s"))
        start = float(argv_value(argv, "--start-s"))
        end = float(argv_value(argv, "--end-s"))
        windows[arm] = {"pre": [start - window, start], "dose": [end - window, end],
                        "recovery": [steps * dt - window, steps * dt]}
        reports[arm] = text

    # Confirm the pair settings are identical apart from the intended treatment
    # program/arm, and derive the windows from their registered argv.
    require(windows["control"] == windows["treatment"], "registered arms use different analysis windows")
    require(windows["control"] == {"pre": [30.0, 60.0], "dose": [70.0, 100.0], "recovery": [280.0, 310.0]},
            "registered analysis windows differ from sealed final plan")
    plan_pair_equal = plan.get("paired_equal", [])
    require(plan_pair_equal, "registered pair lacks equality constraints")
    control_obs = loaded["control"]["owner_result"]
    treatment_obs = loaded["treatment"]["owner_result"]
    for path in plan_pair_equal:
        key = path[0] if path else None
        aliases = {"unit_id": "unit_id", "device": "device", "world_fingerprint": "world_fingerprint",
                   "timestep_s": "timestep_s", "accepted_steps": "accepted_steps", "dense45": "dense45",
                   "brain_control": "brain_control", "common_asset_identity": "common_asset_identity"}
        if key in aliases:
            require(control_obs.get(aliases[key]) == treatment_obs.get(aliases[key]), "paired owner observation differs for " + key)

    wins = windows["control"]
    per_arm = {}
    for arm in ("control", "treatment"):
        rows = rows_by_arm[arm]
        per_arm[arm] = {}
        for name, (lo, hi) in wins.items():
            per_arm[arm][name] = {}
            for key in ("PaCO2_mmhg", "PaO2_mmhg"):
                mean_value, count = window_mean(rows, key, lo, hi)
                per_arm[arm][name][key] = {"mean": mean_value, "samples": count}
        obs = loaded[arm]["owner_result"]
        per_arm[arm]["owner_observation"] = obs
    pre_key = "PaCO2_mmhg"
    trace_arm_delta = {}
    for arm in ("control", "treatment"):
        trace_arm_delta[arm] = per_arm[arm]["dose"][pre_key]["mean"] - per_arm[arm]["pre"][pre_key]["mean"]
        owner_delta = float(per_arm[arm]["owner_observation"]["primary_delta_PaCO2_mmhg"])
        require(math.isfinite(owner_delta) and math.isfinite(trace_arm_delta[arm]),
                arm + " primary delta cross-check is nonfinite")
    trace_delta = trace_arm_delta["treatment"] - trace_arm_delta["control"]
    require(math.isfinite(official_delta) and math.isfinite(trace_delta),
            "registered primary result or raw-trace cross-check is nonfinite")

    def observation_mean(arm, key):
        return float(loaded[arm]["owner_result"][key])

    dose_ve = {arm: observation_mean(arm, "inspiratory_minute_ventilation_dose_L_min") for arm in ("control", "treatment")}
    recovery_ve = {arm: observation_mean(arm, "inspiratory_minute_ventilation_recovery_L_min") for arm in ("control", "treatment")}
    recovery_deltas = {
        "PaCO2_treatment_minus_control_mmhg": per_arm["treatment"]["recovery"]["PaCO2_mmhg"]["mean"] - per_arm["control"]["recovery"]["PaCO2_mmhg"]["mean"],
        "PaO2_treatment_minus_control_mmhg": per_arm["treatment"]["recovery"]["PaO2_mmhg"]["mean"] - per_arm["control"]["recovery"]["PaO2_mmhg"]["mean"],
        "inspiratory_VE_treatment_minus_control_L_min": dose_ve["treatment"] - dose_ve["control"],
        "recovery_inspiratory_VE_treatment_minus_control_L_min": recovery_ve["treatment"] - recovery_ve["control"]
    }
    recovery_deltas["recovery_inspiratory_VE_absolute_relative_difference"] = abs(recovery_deltas["recovery_inspiratory_VE_treatment_minus_control_L_min"]) / max(abs(recovery_ve["control"]), 1e-12)

    out.mkdir(parents=True, exist_ok=False)
    for arm in ("control", "treatment"):
        safe_write(out / (arm + "-physiology.md"), reports[arm])
        audit.write_cycles(out / (arm + "-cycles.csv"), cycles[arm]["detail"])

    lines = [
        "# Supplemental paired physiology and support audit (final integrated study)", "",
        "The registered `numi science analyze` result remains the primary-result authority; this report cross-checks that value from raw PaCO2 windows and adds physical-cycle and support diagnostics. The study plan, margins, and inputs were not changed.", "",
        "## Bound inputs and completion", "",
        "- Registered science verification succeeded read-only.",
        "- Study registration SHA-256: `" + reg_doc["sha256"] + "`.",
        "- Registered science analysis SHA-256: `" + official_doc["sha256"] + "`; registered primary difference: **%.6f mmHg** (verdict `%s`)." % (official_delta, official.get("verdict", "unknown")),
        "- Each arm passed the completion gate: exact registered count 155000 at requested 2 ms, successful registered exit receipt/native terminal/owner observation checks, and sampled endpoint only after exact terminal count. Time labels preserve the runtime Float32 timestep (0.0020000000949949026 s); N=155000 is nominal 310.000 s and its native receipt time is about 310.000014724 s.",
        "- Each final scene native log independently verifies the actually loaded MetalRobo library with the frozen Human loaded_metal_runtime helper, against the parent-verified path and SHA-256.",
        "- Post-init interval analyzed: [10, 310) s (300 observed seconds); 0–10 s initialization excluded.",
        "- Plan prediction envelope: [%.1f, %.1f] mmHg; it is a sensitivity envelope, not a probability interval or clinical prediction." % (float(plan["prediction"]["minimum"]), float(plan["prediction"]["maximum"])), "",
        "## Registered primary result and response checks", "",
        "| Metric | Control | Half-drive | Paired result / interpretation |", "|---|---:|---:|---|",
        "| PaCO2 pre mean, [%.0f, %.0f) s | %.4f | %.4f | mmHg |" % (wins["pre"][0], wins["pre"][1], per_arm["control"]["pre"]["PaCO2_mmhg"]["mean"], per_arm["treatment"]["pre"]["PaCO2_mmhg"]["mean"]),
        "| PaCO2 dose mean, [%.0f, %.0f) s | %.4f | %.4f | mmHg |" % (wins["dose"][0], wins["dose"][1], per_arm["control"]["dose"]["PaCO2_mmhg"]["mean"], per_arm["treatment"]["dose"]["PaCO2_mmhg"]["mean"]),
        "| Arm dose-minus-pre PaCO2 change | %.6f | %.6f | Registered difference-in-differences **%.6f mmHg**; raw-trace cross-check %.6f mmHg (difference %.6f) |" % (trace_arm_delta["control"], trace_arm_delta["treatment"], official_delta, trace_delta, trace_delta - official_delta),
        "| Dose-window inspiratory VE (owner accepted-flow metric) | %.4f | %.4f | Treatment minus control %.4f L/min; required direction `< 0`: %s |" % (dose_ve["control"], dose_ve["treatment"], recovery_deltas["inspiratory_VE_treatment_minus_control_L_min"], "observed" if recovery_deltas["inspiratory_VE_treatment_minus_control_L_min"] < 0 else "not observed"),
        "| Dose complete-breath ledger VE | %.4f | %.4f | Additional event-ledger context; see cycle reports for intervals and cycle counts |" % (float(control_obs["complete_breath_windows"]["dose"]["inspiratory_minute_ventilation_L_min"]), float(treatment_obs["complete_breath_windows"]["dose"]["inspiratory_minute_ventilation_L_min"])), "",
        "## Late recovery comparison", "",
        "Final 30 s is [%.0f, %.0f) s. These are the predeclared numerical equivalence margins, not clinical standards." % tuple(wins["recovery"]),
        "- PaCO2 treatment-minus-control: %+.4f mmHg; margin ±%.1f: %s." % (recovery_deltas["PaCO2_treatment_minus_control_mmhg"], float(recovery_limits["PaCO2_absolute_difference_max_mmhg"]), "within" if abs(recovery_deltas["PaCO2_treatment_minus_control_mmhg"]) <= float(recovery_limits["PaCO2_absolute_difference_max_mmhg"]) else "outside"),
        "- PaO2 treatment-minus-control: %+.4f mmHg; margin ±%.1f: %s." % (recovery_deltas["PaO2_treatment_minus_control_mmhg"], float(recovery_limits["PaO2_absolute_difference_max_mmhg"]), "within" if abs(recovery_deltas["PaO2_treatment_minus_control_mmhg"]) <= float(recovery_limits["PaO2_absolute_difference_max_mmhg"]) else "outside"),
        "- Inspiratory VE treatment-minus-control: %+.4f L/min (relative difference %.2f%%); margin %.1f%%: %s." % (recovery_deltas["recovery_inspiratory_VE_treatment_minus_control_L_min"], 100 * recovery_deltas["recovery_inspiratory_VE_absolute_relative_difference"], 100 * float(recovery_limits["inspiratory_minute_ventilation_relative_difference_max_fraction"]), "within" if recovery_deltas["recovery_inspiratory_VE_absolute_relative_difference"] <= float(recovery_limits["inspiratory_minute_ventilation_relative_difference_max_fraction"]) else "outside"), "",
        "## Physical cycles and conservation", ""
    ]
    for arm in ("control", "treatment"):
        b = cycles[arm]["breaths"]
        h = cycles[arm]["hearts"]
        physical = sum(x["has_both_flow_signs"] and x["volume_excursion_nonzero"] for x in b)
        cardiac_good = sum(x["lv_stroke_positive"] and x["aortic_ejection_positive"] and x["pulmonary_ejection_positive"] for x in h)
        lines.append("- **%s:** %d complete ledger breath intervals after initialization; %d have both positive/negative measured airflow and a nonzero sampled lung-volume excursion. %d cardiac counter intervals; %d have positive LV stroke and positive integrated aortic and pulmonary ejection. Per-cycle evidence is in `%s` and `%s`." % (arm, len(b), physical, len(h), cardiac_good, arm + "-cycles.csv", arm + "-physiology.md"))
        lines.append("- **%s maximum absolute exported residuals:** `%s` (high-watermarks across samples; never sum these running cumulative error fields)." % (arm, json.dumps(budgets[arm], sort_keys=True)))
    lines += ["", "## Accepted geometry captures", "", "Each arm has five shared historical phase samples, two arm-specific late-cycle samples, and a separate terminal capture. Ordinary frame identity is its accepted step ID. Receipt time uses the runtime Float32 representation of the requested 2 ms timestep; nominal times use accepted step ID * 0.002 s. The terminal is exactly accepted N=155000 (nominal 310.000 s; native receipt time about 310.000014724 s); N-1 is not labeled terminal. The seven interior IDs were selected from prior-867 phase evidence, not fitted to these outcomes. Sparse captures do not establish whole-cycle anatomy clearance; the historical global rib-volume minimum near 65.168 s remains unsampled under the eight-frame limit.", ""]
    for arm in ("control", "treatment"):
        cp = capture_reports[arm]
        pairs = ", ".join("%d (%.3f s)%s" % (x["accepted_step"], x["accepted_time_s"], " terminal" if x["terminal"] else "") for x in cp["captures"])
        lines.append("- **%s:** %d registered frames: %s. Terminal capture receipt and MRV pack hashes are listed in manifest.json." % (arm, cp["capture_count"], pairs))
    lines += ["", "## Sampled COM/support drift diagnostics", "", "These are run-bound sampled diagnostics, not static-equilibrium or postural qualification. COM trend is a least-squares line over exported samples after 10 s. The force statistic is the exported aggregate normal impulse summed over contacts for each sampled final physical step divided by the physical timestep; it is not the largest force at one contact or a time-integrated support estimate."]
    for arm in ("control", "treatment"):
        s = support[arm]
        lines.append("- **%s:** %d samples over [%.3f, %.3f) s; endpoint COM displacement %.3f mm (x/y/z = %s mm); maximum excursion from first post-init sample %.3f mm; fitted trend x/y/z = %s mm/min; sampled aggregate total normal support force at last physical step mean/range %s N; active contacts mean/range %s." % (
            arm, s["samples"], s["window_s"][0], s["window_s"][1], s["com_endpoint_displacement_mm"],
            ", ".join("%.3f" % x for x in s["com_endpoint_delta_mm_xyz"]), s["max_com_excursion_from_first_mm"],
            ", ".join("%.5f" % x for x in s["com_linear_trend_mm_per_min_xyz"]),
            "n/a" if not s["sampled_aggregate_normal_support_force_last_physical_step_n"] else "%.2f [%.2f, %.2f]" % (s["sampled_aggregate_normal_support_force_last_physical_step_n"]["mean"], s["sampled_aggregate_normal_support_force_last_physical_step_n"]["min"], s["sampled_aggregate_normal_support_force_last_physical_step_n"]["max"]),
            "n/a" if not s["sampled_active_contact_count"] else "%.2f [%.0f, %.0f]" % (s["sampled_active_contact_count"]["mean"], s["sampled_active_contact_count"]["min"], s["sampled_active_contact_count"]["max"])))
    lines += ["", "## Per-arm generic adult physiology ranges and outliers", "", "The two linked per-arm reports retain post-init means/ranges and explicit deviations versus generic adult intervals and posture-specific cohort summaries. They deliberately preserve outliers. These comparisons do not diagnose a person, calibrate the model, or qualify physiology.", "", "- [Control post-init report](control-physiology.md)", "- [Half-drive post-init report](treatment-physiology.md)", "", "Pressure samples are instantaneous pulsatile waveforms; the per-arm report compares AACN MAP/mPAP reference intervals only with sampled complete-counter-cycle means and identifies those as proxies. Conservation residuals establish numerical accounting only.", ""]
    lines += ["", "## Cardiac interface limitation", "",
        "The retained 911 report localizes 11,568 source-neutral right-atrial/right-ventricular intersections near the common-map tricuspid leaflet projection with a localized source-to-current RV residual up to 0.75 mm. It does not establish a 3D leaflet surface or valve-plane/orifice ownership. The reduced-order CVSim chambers and valves remain the sole functional and blood owner; this is a bounded source-interface limitation, not a whole-heart or anatomy-clearance pass. See [the exact 911 localization report](../../native-cardiac-interface-localization-911/final-localization-report.json).",
        "", "Reference context: [MedlinePlus ABG](https://medlineplus.gov/lab-tests/arterial-blood-gas-abg-test/) lists PaO2 75-100 mmHg, PaCO2 35-45 mmHg and O2 saturation 95-100%; [MedlinePlus Vital Signs](https://medlineplus.gov/ency/article/002341.htm) gives 12-18 breaths/min for the average healthy resting adult and notes individual variation. The per-arm report also retains the AACN PaO2 80-100 mmHg band as a distinct generic reference; both bands and all outliers are shown without post-hoc range changes. [Kovacs et al.](https://doi.org/10.1183/09031936.00145608) review 1,187 individuals across 47 studies, of whom 882 supplied supine data (mPAP 14.0 +/- 3.3 mmHg); the 882 is a posture-specific subset, not the review total. [Mendes et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7253877/) reports male supine quiet-breathing cohort means +/- SD, not individual reference limits.", ""]
    safe_write(out / "paired-physiology.md", "\n".join(lines))

    manifest = {
        "study": str(study), "registration_sha256": reg_doc["sha256"], "registered_analysis_sha256": official_doc["sha256"],
        "registered_analysis_verifier": "numi science verify", "official_primary_difference_mmhg": official_delta,
        "raw_trace_crosscheck_difference_mmhg": trace_delta,
        "inputs": {}, "outputs": {}, "accepted_geometry_captures": capture_reports,
        "per_scene_loaded_metal_runtime": {
            arm: loaded[arm]["scene_loaded_metal_runtime"] for arm in ("control", "treatment")
        }
    }
    manifest["inputs"]["cardiac_interface_localization_911"] = {
        "path": str(CARDIAC_911), "sha256": audit.sha256(CARDIAC_911)}
    for arm in ("control", "treatment"):
        item = loaded[arm]
        for name, path in (("trace", item["csv"]), ("native_log", item["trial"] / "output/scene/native.log"),
                           ("owner_observation", item["trial"] / "output/scene/intervention-observation.json"),
                           ("com_diagnostic", item["trial"] / "output/scene/resting-com-momentum-diagnostic.csv")):
            manifest["inputs"][arm + "_" + name] = {"path": str(path), "sha256": audit.sha256(path)}
    for p in sorted(out.iterdir()):
        if p.is_file():
            manifest["outputs"][p.name] = audit.sha256(p)
    json_write(out / "manifest.json", manifest)
    print("Wrote supplemental analysis under " + str(out))
    print("Registered primary = %.6f mmHg; raw trace cross-check = %.6f mmHg" % (official_delta, trace_delta))
    print("Registered plan/analysis/receipt verification passed; no study inputs were changed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write("REFUSED/FAILED: %s\n" % exc)
        raise SystemExit(2)
