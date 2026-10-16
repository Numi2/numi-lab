#!/usr/bin/env python3
"""Reproduce the native497 full-damped candidate strict-reference screen (CPU only)."""
import contextlib, hashlib, io, json, pathlib, sys, time, warnings
import numpy as np

Q = pathlib.Path("/Users/n/cardboard-implementation-20261014-host-macmini")
A = Q / "analysis"
CAND = A / "native497-full-damped-candidate-set-007"
REPLAY = A / "native497-nonlinear-correction-001/replay.py"
TOOLS = Q / "snapshots/source-portable-paper-reference-185/matter/tools"
EXPECTED = {
    "candidate_manifest": "f6b1d1dd4d8c1abc45a4d6f6bd7c542ef432d1ab35b4da131b940f8cb7fbfabe",
    "replay": None,
    "base": "a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995",
    "strict": "760ea825449857d565c78e754c755845e4547134b381fa167d02aab2ae63720c",
    "dd": "8edfd60beb2c38d181a08023d9f97bea0fec80b0f6fa04f8690c7dfe234330c2",
}
ALPHAS = [0.0, 0.25, 0.5, 1.0]
CANDIDATES = [
    ("svd_lambda_1e-06", "svd_lambda_1e-06_full_step.npy", "9d22c4683b00aab9f68bad99d9620dfcf290d479e838cd9eaba72afbcfdf9c3f",
     "full total step from zero, dense binary64 SVD Tikhonov lambda=1e-6"),
    ("svd_lambda_1e-05", "svd_lambda_1e-05_full_step.npy", "99db1d0f29cf579e84128f1e8540efa5971f1c4fe9e9a58b90a56c2b89514819",
     "full total step from zero, dense binary64 SVD Tikhonov lambda=1e-5"),
    ("svd_lambda_1e-04", "svd_lambda_0.0001_full_step.npy", "7ad95583ad2f1d60c68688d73f8c167012b2d0c0b30c0bb8ee3586e138a595df",
     "full total step from zero, dense binary64 SVD Tikhonov lambda=1e-4"),
    ("svd_lambda_1e-03", "svd_lambda_0.001_full_step.npy", "533aaa128012a232981c516ab36a4e56371c45c53e52712f3f6962c29d023702",
     "full total step from zero, dense binary64 SVD Tikhonov lambda=1e-3"),
    ("lsqr_lambda_0_capped", "iteration_limited_lsqr_lambda_0_full_step.npy", "959214302a800a25a8118c52578d82e45155bc5c3f5c2183e64a04df52e8b3f1",
     "incomplete full total step from zero, LSQR lambda=0, status 7 at 5000-iteration cap"),
]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.output.resolve()
    if not out.exists():
        out.mkdir(parents=True)
    if any(p.name != "run_screen.py" for p in out.iterdir()):
        raise SystemExit("refusing to overwrite nonempty output namespace")
    hashes = {"candidate_manifest": sha(CAND / "manifest.json"),
              "replay": sha(REPLAY),
              "base": sha(TOOLS / "cardboard_bilinear_fp64_reference.py"),
              "strict": sha(TOOLS / "cardboard_compensated_fp64_reference.py"),
              "dd": sha(TOOLS / "cardboard_dd_fp64.py")}
    for key, expected in EXPECTED.items():
        if expected is not None and hashes[key] != expected:
            raise SystemExit(f"source binding mismatch: {key}: {hashes[key]}")
    src = REPLAY.read_text()
    old = "alphas=[0,1/128,1/64,1/32,1/16,1/8,1/4,1/2,1]"
    if old not in src:
        raise SystemExit("baseline replay alpha-grid contract changed")
    src = src.replace(old, "alphas=[0.0,0.25,0.5,1.0]")
    ns = {"__name__": "native497_candidate_replay"}
    exec(compile(src, str(REPLAY), "exec"), ns)
    side_m, side_a = ns["sidecar"]()
    state = side_a["candidate_nodes"].reshape(2, 882, 20)[0]
    free = np.flatnonzero((state[:,11] != 1) & (state[:,11] != 3) & (state[:,3] > 0))
    if len(free) != 846:
        raise SystemExit(f"unexpected selected free node count {len(free)}")
    raw_capture_cgls = ns["cgls"]()
    results = []
    observed_warning_counts = {}
    for name, filename, expected_sha, semantics in CANDIDATES:
        path = CAND / filename
        candidate_sha = sha(path)
        if candidate_sha != expected_sha:
            raise SystemExit(f"candidate hash mismatch for {filename}")
        candidate = np.load(path, allow_pickle=False)
        if candidate.shape != (846 * 3,) or not np.isfinite(candidate).all():
            raise SystemExit(f"candidate shape/finite check failed: {filename}")
        candidate_f32 = np.asarray(candidate, dtype=np.float32)
        xbuf = np.zeros_like(raw_capture_cgls["x"])
        xbuf[free, :3] = candidate_f32.reshape(846, 3)
        ns["cgls"] = lambda base=raw_capture_cgls, x=xbuf: {**base, "x": x}
        caseout = out / name
        caseout.mkdir()
        ns["OUT"] = caseout
        capture = io.StringIO()
        with warnings.catch_warnings(record=True) as warns, contextlib.redirect_stdout(capture):
            warnings.simplefilter("always")
            ns["main"]()
        for w in warns:
            key = (name, w.category.__name__, str(w.message))
            observed_warning_counts[key] = observed_warning_counts.get(key, 0) + 1
        rp = caseout / "report.json"
        report = json.loads(rp.read_text())
        report["candidate_direction"] = {
            "name": name, "file": filename, "sha256": candidate_sha, "semantics": semantics,
            "original_array_dtype": str(candidate.dtype), "trial_update_dtype": "float32",
            "free_node_count": 846, "free_xyz_count": 2538,
            "free_node_order": "ascending native node IDs selected from exact captured candidate nodes by nonfixed tag and positive mass",
            "vector_l2_m_s": float(np.linalg.norm(candidate)),
            "trial_vector_float32_l2_m_s": float(np.linalg.norm(candidate_f32)),
            "candidate_set_manifest_sha256": hashes["candidate_manifest"],
            "note": "External vector evaluated as a total trial correction; no CGLS certificate or native nonlinear acceptance is inherited."
        }
        report["evidence_class"] = "CPU-only strict-reference nonlinear direction screen; no native dispatch or qualification"
        report["native_armijo_newton63"] = {"not_applicable": "Original captured Newton candidate only; it is not the external candidate's trial."}
        report["interpretation"]["limits"].append("External candidate was generated from the captured FP32-cache operator; strict compensated FP64 replay is not native acceptance.")
        write_json(rp, report)
        (caseout / "README.md").write_text("# External damped direction (CPU only)\n\n" + semantics + ". Evaluated at alpha 0, 0.25, 0.5, and 1 using strict compensated FP64 material projection/stress, anchored to the captured accepted state and center residual. Candidate components are narrowed to FP32 for the production-shaped velocity/displacement update. No native trial or solver certificate is inferred.\n")
        case_files = {}
        for f in sorted(caseout.iterdir()):
            if f.is_file():
                case_files[f.name] = {"bytes": f.stat().st_size, "sha256": sha(f)}
        write_json(caseout / "manifest.json", {"schema":"numi.cardboard.analysis-package.v1","package":name,"files":case_files,"scope":"CPU-only external direction replay"})
        results.append({
            "name": name, "sha256": candidate_sha, "norm_m_s": float(np.linalg.norm(candidate_f32)),
            "curve": [{k: row[k] for k in ("alpha","strict_all_paper_projection_pass","strict_fail_count","free_residual_l2_Ns","mass_only_residual_l2_Ns","internal_only_residual_l2_Ns","mass_delta_l2_Ns","internal_delta_l2_Ns")} for row in report["curve"]]
        })
    root = {
        "schema":"numi.cardboard.native497-full-damped-nonlinear-screen.v1",
        "evidence_class":"CPU-only strict compensated FP64 reference screen of external complete direction vectors; no native trials or acceptance",
        "source_case":str(ns["CASE"]), "root":20, "newton_iteration":63, "environment":0,
        "bindings": {
            "capture_operator_fd":sha(ns["FD"]), "accepted_state_json":sha(ns["SJ"]), "accepted_state_binary":sha(ns["SB"]),
            "raw_cgls_capture":sha(ns["CG"]), "compiled_package":sha(ns["CASE"]/"output/compiled.nmatterpack"),
            "candidate_set_manifest":hashes["candidate_manifest"], "candidate_set_report":sha(CAND/"full_step_svd_report.json"),
            "iteration_limited_lsqr_report":sha(CAND/"iteration_limited_lsqr_report.json"),
            "replay_source":hashes["replay"], "reference_base":hashes["base"], "reference_strict":hashes["strict"], "dd_helper":hashes["dd"]
        },
        "reference_contract":{"strict_residual_gate":2e-12,"dynamic_correction_gate":2e-11,"iteration_ceiling":80},
        "trial_contract":{"alphas":ALPHAS,"candidate_is_total_x_from_zero":True,"candidate_order":"candidate set nodal XYZ order bound by original SVD/LSQR free-XYZ mapping; embedded on ascending captured free native node IDs","cast_before_trial_update":"FP32"},
        "results":results,
        "observed_reference_warnings":[
            {"candidate":k[0],"category":k[1],"message":k[2],"count":v}
            for k,v in sorted(observed_warning_counts.items())],
        "limits":["All candidates are external CPU vectors, not native trials.","The LSQR lambda=0 vector stopped at iteration cap 5000 with status 7 and is incomplete.","No physical/full-coupon qualification; all reported residuals are strict-reference candidate screens.","No damping candidate receives the original Newton/CGLS convergence certificate.","Original base reference emits an eager-branch divide warning at zero denominators in np.where; outputs remain finite and all strict gates passed."]
    }
    write_json(out/"report.json", root)
    (out/"README.md").write_text("# Native497 full-damped candidate nonlinear screen\n\nCPU-only strict-reference nonlinear screen of five external total-step candidates at alpha 0, 1/4, 1/2, and 1. All source arrays, reference files, exact capture identity, and strict gates are bound in report.json. The capped LSQR solution is retained as incomplete. No result is a native trial, solver certificate, or full-coupon qualification.\n")
    inventory = {}
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name != "manifest.json":
            inventory[str(f.relative_to(out))] = {"bytes":f.stat().st_size,"sha256":sha(f)}
    write_json(out/"manifest.json", {"schema":"numi.cardboard.analysis-package.v1","package":"native497-full-damped-nonlinear-screen-003","scope":"CPU-only strict-reference direction screen","files":inventory})
    print(json.dumps({"output":str(out),"results":[{"name":r["name"],"curve":r["curve"]} for r in results],"warnings":len(observed_warning_counts)},indent=2))

if __name__ == "__main__":
    main()
