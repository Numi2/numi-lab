from pathlib import Path
import importlib.util
import argparse
import json
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT
W = D / "D/contained-clearance-545"
N = D / "D/surface-clearance-546"
N.mkdir(exist_ok=True)
ADAPTER = N / "refresh_surface_constraints546.py"
TARGET_BUILDER = D / "D/source-outward-proposal-500/build_outward_targets_multi_source_546.py"
TARGET_BUILDER_SHA = "dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
spec = importlib.util.spec_from_file_location("cont545_prepare546", W / "continue545.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def legacy_refresh(a, O, st, base, prior, q, trial, trial_audit):
    """Compatibility path for contexts without a previous 159 report."""
    args = [m.PY, D / "D/focused-clearance-541/constraints/refresh_guards_541.py",
        "--candidate", base, "--audit130", st["point_summary"], "--audit159", st["surface_report"],
        "--query-dir", q, "--ancestry-parent-candidate", a.ancestry_parent_candidate or base,
        "--seed-source-candidate", trial, "--seed-source-audit130", trial_audit,
        "--seed-source-pairs", trial_audit.parent / "surface-self-pairs.npy",
        "--prior-guard-candidate", prior["candidate"], "--prior-guard-report", prior["guard_report"],
        "--prior-guard-npz", prior["guard_npz"], "--output-dir", O / "constraints",
        "--expected-candidate-sha256", m.sha(base),
        "--expected-audit130-sha256", m.sha(st["point_summary"]),
        "--expected-audit159-sha256", m.sha(st["surface_report"])]
    m.run(args, O / "refresh", True)
    rr = m.read(O / "constraints/refresh-report.json")
    z = np.load(base, allow_pickle=False)
    pz = np.load(prior["candidate"], allow_pickle=False)
    if not np.array_equal(z["source_O_faces"], pz["source_O_faces"]):
        raise ValueError("topology changed; pass --prior-surface-report to use the topology-aware adapter")
    pairs = np.unique(np.vstack([np.load(prior["seed_pairs_path"], allow_pickle=False),
                                 np.load(rr["seed_pairs_path"], allow_pickle=False)]), axis=0)
    seed = O / "constraints/merged-self-pair-seeds.npy"
    np.save(seed, pairs)
    owner = D / "publication-101/evidence/204/source/fit/self_separation_026.py"
    if m.sha(owner) != "c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3":
        raise ValueError("026 seed owner pin mismatch")
    rows, _ = m.loadmod(owner).build_pair_rows(z["points_world_m"].astype(float) * 1000,
                                                z["source_O_faces"], pairs, margin_fraction=.25)
    rhs = max((float(row["lower"]) for row in rows), default=0.0)
    if rhs > 2e-8:
        raise ValueError(f"merged 026 seeds fail zero-step certificate: {rhs}")
    seed_report = O / "constraints/seed-preflight.json"
    m.write(seed_report, dict(schema="numi.human.continuation-self-seeds.v1",
        status="PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY", candidate_sha256=m.sha(base),
        zero_step_feasible=True, zero_step_max_rhs_mm=rhs, rebuilt_linear_row_count=len(rows),
        output_pairs_path=str(seed), output_pairs_sha256=m.sha(seed), merged_unique_pair_count=len(pairs),
        pair_owner_sha256=m.sha(owner), prior_seed_path=prior["seed_pairs_path"],
        prior_seed_sha256=m.sha(prior["seed_pairs_path"]), new_trial_seed_path=rr["seed_pairs_path"],
        new_trial_seed_sha256=m.sha(rr["seed_pairs_path"]), native_admission=False, accepted_physical_steps=0))
    rr.update(seed_pairs_path=str(seed), seed_pairs_sha256=m.sha(seed),
              seed_report_path=str(seed_report), seed_report_sha256=m.sha(seed_report))
    return rr


def topology_aware_refresh(a, O, st, base, prior, q, trial, trial_audit, prior_surface):
    if not all((trial.is_file(), trial_audit.is_file(), Path(prior["seed_pairs_path"]).is_file(),
                Path(prior.get("seed_report", "")).is_file())):
        raise ValueError("topology-aware refresh requires prior seeds and failed-trial candidate/audit")
    args = [m.PY, ADAPTER,
        "--candidate", base, "--audit130", st["point_summary"], "--audit159", st["surface_report"],
        "--query-dir", q, "--ancestry-parent-candidate", a.ancestry_parent_candidate or base,
        "--prior-candidate", prior["candidate"], "--prior-surface-report", prior_surface,
        "--prior-guard-candidate", prior["candidate"], "--prior-guard-npz", prior["guard_npz"],
        "--prior-guard-report", prior["guard_report"], "--prior-seed-pairs", prior["seed_pairs_path"],
        "--prior-seed-report", prior["seed_report"], "--failed-seed-candidate", trial,
        "--failed-seed-audit130", trial_audit,
        "--failed-seed-pairs", trial_audit.parent / "surface-self-pairs.npy",
        "--output-dir", O / "constraints"]
    m.run(args, O / "refresh", True)
    return m.read(O / "constraints/refresh-report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prior-context", type=Path, required=True)
    ap.add_argument("--seed-source-candidate", type=Path)
    ap.add_argument("--seed-source-audit130", type=Path)
    ap.add_argument("--prior-surface-report", type=Path,
                    help="previous-state 159 audit; also read from prior context when available")
    ap.add_argument("--ancestry-parent-candidate", type=Path)
    a = ap.parse_args()
    O = a.out
    O.mkdir(parents=True, exist_ok=False)
    st = m.read(a.state)
    base = Path(st["candidate"])
    q = Path(st["point_summary"]).parent / "query"
    prior = m.read(a.prior_context)
    for key in ("candidate", "point_summary", "surface_report"):
        if m.sha(st[key]) != st[key + "_sha256"]:
            raise ValueError(f"current state hash mismatch: {key}")
    point = m.read(st["point_summary"])
    if (point.get("exact_PLC_surface_pairs") != 0 or point.get("inside_node_bank_pairs") != 0
            or point.get("complete_surface_audit") is not True):
        raise ValueError("surface-only refresh requires complete exact zero-self, zero-containment current state")
    m.write(O / "baseline.json", st)
    trial = a.seed_source_candidate or W / "fit-001/candidate.npz"
    trial_audit = a.seed_source_audit130 or W / "check-001-diagnostic/audit-130/summary.json"
    prior_surface = a.prior_surface_report or (Path(prior["surface_report"]) if prior.get("surface_report") else None)
    if prior_surface is not None:
        rr = topology_aware_refresh(a, O, st, base, prior, q, Path(trial), Path(trial_audit), prior_surface)
        seed_path = Path(rr["seed_pairs_path"])
        seed_report = Path(rr["seed_report_path"])
        guard_npz = Path(rr["fixed_guard_npz_path"])
        guard_report = Path(rr["fixed_guard_report_path"])
        refresh_report_path = O / "constraints/refresh-report.json"
    else:
        rr = legacy_refresh(a, O, st, base, prior, q, Path(trial), Path(trial_audit))
        seed_path = Path(rr["seed_pairs_path"])
        seed_report = Path(rr["seed_report_path"])
        guard_npz = Path(rr["fixed_guard_npz_path"])
        guard_report = Path(rr["fixed_guard_report_path"])
        refresh_report_path = O / "constraints/refresh-report.json"
    context = {"candidate": str(base), "candidate_sha256": m.sha(base),
        "seed_pairs_path": str(seed_path), "seed_report": str(seed_report),
        "seed_report_sha256": m.sha(seed_report), "guard_npz": str(guard_npz),
        "guard_report": str(guard_report), "surface_report": str(st["surface_report"]),
        "surface_report_sha256": m.sha(st["surface_report"]), "point_summary": str(st["point_summary"]),
        "point_summary_sha256": m.sha(st["point_summary"]), "refresh_report": str(refresh_report_path),
        "refresh_report_sha256": m.sha(refresh_report_path), "native_admission": False,
        "accepted_physical_steps": 0}
    context_path = O / "constraint-context.json"
    m.write(context_path, context)

    depth_args = m.read(W / "depth-argv.json")
    changes = {"--candidate": base, "--point-summary": st["point_summary"],
        "--surface-report": st["surface_report"], "--query-dir": q, "--output-dir": O / "depth",
        "--expected-candidate-sha256": m.sha(base), "--expected-point-summary-sha256": m.sha(st["point_summary"]),
        "--expected-surface-report-sha256": m.sha(st["surface_report"]),
        "--expected-query-receipt-sha256": m.sha(q / "query-receipt.json")}
    for flag, value in changes.items():
        depth_args[depth_args.index(flag) + 1] = str(value)
    m.run(depth_args, O / "depth", True)
    if m.sha(TARGET_BUILDER) != TARGET_BUILDER_SHA:
        raise ValueError("pinned multi-parent 500 target builder hash mismatch")
    m.run([m.PY, TARGET_BUILDER,
        "--candidate", base, "--surface-report", st["surface_report"], "--query-receipt", q / "query-receipt.json",
        "--output-dir", O / "surface-targets", "--clearance-m", ".00005", "--direction-mode", "outward-nearest"],
        O / "surfaces", True)
    m.write(O / "prepared.json", dict(status="FRESH_CONSTRAINTS_DEPTH_SURFACE_TARGETS_PREPARED",
        state=str(a.state), context=str(context_path), topology_aware_constraint_refresh=(prior_surface is not None),
        surface_target_builder_path=str(TARGET_BUILDER), surface_target_builder_sha256=TARGET_BUILDER_SHA,
        native_admission=False, accepted_physical_steps=0))
    print(json.dumps(m.read(O / "prepared.json")), flush=True)


if __name__ == "__main__":
    main()
