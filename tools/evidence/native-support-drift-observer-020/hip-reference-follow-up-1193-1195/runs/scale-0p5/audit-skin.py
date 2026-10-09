#!/usr/bin/env python3
"""Run-specific exact skin audit for the nominal 40 s hip-reference capture.

Without --scan this performs only source/invocation/receipt preflight. It reads
an existing native run; it never launches or modifies the native simulation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

R = Path("/Users/n/numi-human-retained-delivery-20261009/native-hip-reference-scale05-1194")
E = Path("/Users/n/numi-human-resting-evidence-20261005")
RUN_DEFAULT = R / "native-run"
OUT_DEFAULT = R / "skin-audit-1194-step20000-exact"
STEPS = (20000,)
DT = 0.002
ROOTS = 20000
HIP_CAPSULE_SCALE = 0.5
SKIN = E / "native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin"
SKIN_SHA = "b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b"
SKIN_REPORT = E / "native-skin-epl143-clearance-1187/attempt-006/report.json"
SKIN_REPORT_SHA = "9df248cf4cdcad92dc28a7ad5b0336c365e93ecf712e5811b8d392a11b708e69"
NHA = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy"
NHA_SHA = "1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
NHA_COMPOSITION = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/composition-report.json"
NHA_COMPOSITION_SHA = "ddbfae12a5acd5c35521cb913121f08e49fb8c336c89b7dd60df3272ec0a7fbc"
NHA_MANIFEST = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-anatomy-manifest.json"
NHA_MANIFEST_SHA = "de298b7cc413e0cf079b39cb5d4792ebb537642f23a52cb4747f2dc37fac917f"
TISS = E / "passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
TISS_SHA = "b3d0381f73e05b04ed7ea1ba23eb6559073ad20aa0d9e2ad0f97a18fab3359bd"
TISS_MANIFEST = E / "passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
TISS_MANIFEST_SHA = "82cdd937e0f2daf0a8704fb21246353c38148527f8602d674ac865f935264dde"
RUNNER = E / "native-lung-late-skin-audit-runner-1172/revision-003/audit_full_skin_cycle_1172.py"
RUNNER_SHA = "b1b9be5959610325067f703c32bd55e2f568c11124ec232f0d066ed75a6fdcbb"
INV = E / "native-complete-skin-containment-audit-890/pair-summary-v3.csv"
INV_SHA = "a43484aa8d65b207cc498bb470097ec5f06ef2bb9872837bde9f197e19e815e3"
PINNED_STATIC = {
    SKIN: SKIN_SHA, SKIN_REPORT: SKIN_REPORT_SHA,
    NHA: NHA_SHA, NHA_COMPOSITION: NHA_COMPOSITION_SHA, NHA_MANIFEST: NHA_MANIFEST_SHA,
    TISS: TISS_SHA, TISS_MANIFEST: TISS_MANIFEST_SHA,
    RUNNER: RUNNER_SHA, INV: INV_SHA,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_hash(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink() and sha(path) == expected,
            "pinned file missing, symlinked, or changed: " + str(path))


def option(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        raise ValueError("native invocation lacks " + name)


def load_runner():
    check_hash(RUNNER, RUNNER_SHA)
    spec = importlib.util.spec_from_file_location("_skin_exact_audit_1172_for_1194_step20000", RUNNER)
    require(spec is not None and spec.loader is not None, "cannot load pinned 1172 audit owner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Reuse only the pinned owner functions; the new NHSKIN is explicitly source-pinned here.
    module.SKIN = SKIN
    module.SKIN_SHA = SKIN_SHA
    return module


def validate_static_sources():
    for path, expected in PINNED_STATIC.items():
        check_hash(path, expected)
    skin_report = json.loads(SKIN_REPORT.read_text())
    assets = skin_report["candidate_assets"]
    require(Path(assets["composed_candidate_path"]).resolve() == SKIN.resolve()
            and assets["composed_candidate_sha256"] == SKIN_SHA,
            "1187 report does not bind the selected b2d NHSKIN")
    require(assets["triangle_indices_and_order_identical"] is True
            and assets["all_86_bindings_identical"] is True,
            "1187 candidate topology or 86-binding identity is not preserved")
    anatomy = json.loads(NHA_MANIFEST.read_text())
    require(anatomy["payload"]["path"] == str(NHA)
            and anatomy["payload"]["sha256"] == NHA_SHA,
            "1178 manifest does not bind the selected NHA")
    composition = json.loads(NHA_COMPOSITION.read_text())
    final_nha = composition.get("outputs", {}).get("final_nha", {})
    require(final_nha.get("path") == str(NHA) and final_nha.get("sha256") == NHA_SHA,
            "1178 composition report does not bind the selected NHA")
    tissue = json.loads(TISS_MANIFEST.read_text())
    tissue_payload = tissue.get("payload", {})
    require(tissue_payload.get("file") == TISS.name
            and tissue_payload.get("sha256") == TISS_SHA,
            "039 manifest does not bind the selected NHTISS")
    runner = load_runner()
    owner_pins = {
        runner.AUDIT_CORE: runner.AUDIT_CORE_SHA,
        runner.HELPER: runner.HELPER_SHA,
        runner.PRED: runner.PRED_SHA,
        runner.CLEARANCE: runner.CLEARANCE_SHA,
        **runner.PREDICATE_SOURCE_PINS,
    }
    for path, expected in owner_pins.items():
        check_hash(path, expected)
        PINNED_STATIC[path] = expected
    return {str(Path(path).resolve()): digest for path, digest in PINNED_STATIC.items()}


def preflight(run, out, scan):
    run = Path(run).expanduser().resolve()
    require(run == RUN_DEFAULT.resolve(), "run must be the designated 1194 native-run directory")
    require(run.is_dir() and not run.is_symlink(), "1194 native-run is not available as a real directory")
    fixed = validate_static_sources()
    invocation_path = run / "invocation.json"
    metadata_path = run / "run-metadata.json"
    log_path = run / "native.log"
    require(invocation_path.is_file() and metadata_path.is_file() and log_path.is_file(),
            "native invocation, run metadata, or native log is missing")
    invocation = json.loads(invocation_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    argv, assets = invocation.get("argv"), invocation.get("asset_sha256")
    require(isinstance(argv, list) and isinstance(assets, dict), "invocation lacks argv or asset map")
    require(metadata.get("exit_code") == 0 and metadata.get("argv") == argv
            and metadata.get("asset_sha256") == assets,
            "native execution metadata does not prove the same successful invocation")
    runtime = metadata.get("loaded_metal_runtime", {})
    require(runtime.get("verified") is True, "loaded native Metal runtime is not verified")
    changed_sources = metadata.get("source_files_changed_during_run")
    if isinstance(changed_sources, list):
        require(not changed_sources, "native source files changed during the run")
    elif changed_sources is not None:
        require(changed_sources is False, "native source-change state is not a clean false value")
    require(Path(option(argv, "--skin-payload")).resolve() == SKIN.resolve()
            and assets.get(str(SKIN)) == SKIN_SHA,
            "actual native invocation did not load the exact 1187 b2d NHSKIN")
    require(Path(option(argv, "--torso-anatomy-payload")).resolve() == NHA.resolve()
            and assets.get(str(NHA)) == NHA_SHA,
            "actual native invocation did not load the exact 1178 NHA")
    require(Path(option(argv, "--soft-tissue-payload")).resolve() == TISS.resolve()
            and assets.get(str(TISS)) == TISS_SHA,
            "actual native invocation did not load the exact 039 NHTISS")
    dt, roots = float(option(argv, "--muscle-step-seconds")), int(option(argv, "--muscle-step-count"))
    require(dt == DT and roots == ROOTS, "native run is not the declared 40 s, 2 ms run")
    require("--resting-hip-capsule-reference" in argv and
            float(option(argv, "--resting-hip-capsule-scale")) == HIP_CAPSULE_SCALE,
            "native run hip-capsule reference/scale differs from this sensitivity audit")
    for step in STEPS:
        pack = run / "accepted-geometry" / ("step-%d.mrvpack" % step)
        receipt = pack.with_suffix(".receipt.json")
        require(pack.is_file() and not pack.is_symlink() and receipt.is_file() and not receipt.is_symlink(),
                "required accepted capture is missing or symlinked: step %d" % step)
        receipt_doc = json.loads(receipt.read_text())
        require(int(receipt_doc.get("accepted_step", -1)) == step,
                "accepted receipt identifies a different step: %d" % step)
        expected_time = step * DT
        require(abs(float(receipt_doc.get("accepted_time_s", float("nan"))) - expected_time) < 2e-6,
                "accepted receipt time does not match the requested capture: %d" % step)
    out = Path(out).expanduser().resolve()
    require(out == OUT_DEFAULT.resolve() and not out.exists(),
            "audit output must be the fresh designated skin-audit-1194-step20000-exact directory")
    candidate_input_paths = [invocation_path, metadata_path, log_path]
    for step in STEPS:
        candidate_input_paths.extend((run / "accepted-geometry" / ("step-%d.mrvpack" % step),
                                      run / "accepted-geometry" / ("step-%d.receipt.json" % step)))
    fixed.update({str(p.resolve()): sha(p) for p in candidate_input_paths})
    # Recheck every asset hash declared by the actual invocation, including binary,
    # shader, NHCNT, and source payloads. This is a read-only identity check.
    for asset_path, expected_sha in assets.items():
        asset = Path(asset_path).resolve()
        check_hash(asset, expected_sha)
        fixed[str(asset)] = expected_sha
    return run, out, invocation, metadata, fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN_DEFAULT,
                        help="read-only native-run directory; fixed to the 1194 run path")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT,
                        help="fresh audit output path; fixed to the 1194 sibling directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-check", action="store_true",
                      help="verify pinned static assets/owner sources only; no native-run required")
    mode.add_argument("--scan", action="store_true",
                      help="after strict preflight, run the CPU-only exact geometry audit")
    args = parser.parse_args()
    if args.prepare_check:
        tracked = validate_static_sources()
        runner = load_runner()
        keys, _ = runner.inventory()
        require(len(keys) == 859, "expected all 859 target surfaces")
        runner.load_predicates()
        print(json.dumps({"status": "static_preparation_pins_valid_no_run_or_scan",
                          "targets": len(keys), "steps": list(STEPS),
                          "NHSKIN_sha256": SKIN_SHA, "NHA_sha256": NHA_SHA,
                          "NHTISS_sha256": TISS_SHA, "pinned_static_files": len(tracked)}, sort_keys=True))
        return 0
    run, out, invocation, metadata, tracked_before = preflight(args.run, args.out, args.scan)
    runner = load_runner()
    keys, expected = runner.inventory()
    require(len(keys) == 859, "expected all 859 target surfaces")
    ci, clearance, validator, core = runner.load_predicates()
    results = []
    declaration = {
        "schema": "numi.human.hip-reference-1194-step20000-skin-audit.declaration.v1",
        "status": "prepared_only_not_scanned" if not args.scan else "scan_started",
        "native_run": str(run),
        "native_invocation_sha256": sha(run / "invocation.json"),
        "native_run_metadata_sha256": sha(run / "run-metadata.json"),
        "native_log_sha256": sha(run / "native.log"),
        "dt_seconds": DT, "root_steps": ROOTS, "captures": list(STEPS),
        "hip_capsule_reference": {"enabled": True, "scale": HIP_CAPSULE_SCALE},
        "NHSKIN": {"path": str(SKIN), "sha256": SKIN_SHA,
                   "source_report": str(SKIN_REPORT), "source_report_sha256": SKIN_REPORT_SHA},
        "NHA": {"path": str(NHA), "sha256": NHA_SHA,
                "composition_report_sha256": NHA_COMPOSITION_SHA,
                "manifest_sha256": NHA_MANIFEST_SHA},
        "NHTISS": {"path": str(TISS), "sha256": TISS_SHA,
                   "manifest_sha256": TISS_MANIFEST_SHA},
        "target_inventory": {"path": str(runner.INV), "sha256": INV_SHA,
                             "surfaces": 859, "ocular_surfaces": [[51010, n] for n in range(381, 398)]},
        "predicates": {"runner": str(RUNNER), "runner_sha256": RUNNER_SHA,
                       "audit_core_sha256": runner.AUDIT_CORE_SHA,
                       "accepted_pack_validator_sha256": runner.HELPER_SHA,
                       "exact_intersection_sha256": runner.PRED_SHA,
                       "skin_pack_reader_sha256": runner.CLEARANCE_SHA},
        "scope": "One actual accepted native capture at step 20000 only; exact captured Float32-lattice NHSKIN-versus-all-859-target triangle pairs, whole-skin self intersections, and degenerate triangles. No contact exemptions, displacement reconstruction, continuous-time claim, or physiology qualification.",
        "inputs_before_scan": tracked_before,
    }
    if not args.scan:
        print(json.dumps({"status": "preflight_passed_no_scan", "run": str(run),
                          "steps": list(STEPS), "targets": len(keys),
                          "NHSKIN_sha256": SKIN_SHA, "NHA_sha256": NHA_SHA,
                          "NHTISS_sha256": TISS_SHA}, sort_keys=True))
        return 0
    out.mkdir()
    declaration["status"] = "running_exact_single_capture_audit"
    (out / "declaration.json").write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    for step in STEPS:
        results.append(runner.run_pose(step, run, out, ci, clearance, validator, core,
                                       keys, expected, NHA_SHA))
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "status": results[-1]["status"],
                                     "elapsed_s": results[-1]["elapsed_s"]}, sort_keys=True) + "\n")
    tracked_after = {path: sha(path) for path in tracked_before}
    unchanged = tracked_before == tracked_after
    complete = len(results) == len(STEPS) and [r["accepted_step"] for r in results] == list(STEPS)                and all(r["pair_coverage_complete"] for r in results)
    free = complete and all(r["all_skin_crossing_pair_count"] == 0
                            and r["skin_self_crossing_pair_count"] == 0 for r in results)
    summary = {"schema": "numi.human.hip-reference-1194-step20000-skin-audit.summary.v1",
               "status": "complete_exact_intersection_free" if free and unchanged else
                         ("complete_with_intersections_or_invalid_geometry" if complete and unchanged else "incomplete_or_input_changed"),
               "native_run": str(run), "steps": list(STEPS), "dt_seconds": DT,
               "root_steps": ROOTS, "hip_capsule_reference": {"enabled": True, "scale": HIP_CAPSULE_SCALE},
               "NHSKIN_sha256": SKIN_SHA, "NHA_sha256": NHA_SHA,
               "NHTISS_sha256": TISS_SHA, "target_surface_count": 859,
               "all_steps_complete": complete, "all_target_and_skin_self_pair_coverage_complete": complete,
               "intersection_free": free, "input_hashes_unchanged": unchanged,
               "inputs_before": tracked_before, "inputs_after": tracked_after,
               "pose_results": results,
               "qualification": "Exact geometry evidence for one accepted native capture only; raw crossings and degeneracies are retained. No interpolation between frames and no physical/physiological qualification."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "steps", "all_steps_complete", "intersection_free", "input_hashes_unchanged")}, sort_keys=True))
    return 0 if complete and unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
