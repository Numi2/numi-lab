#!/usr/bin/env python3
"""Run the existing independent construction and edge-gradient checks together.

This is construction verification only. A passing receipt never admits a physical
skin asset or asserts that unresolved crossings have been cleared.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
VERIFIER = D / "D/independent-verifier-494/verify_construction_candidate_494.py"
EDGE = D / "D/D/area-local-backtrack-514/check_408_edge_gradient_514.py"
OWNER = D / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
REFERENCE = D / "full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def receipt_passes(proof, edge, selected_sha, returns, unchanged):
    rows = edge.get("candidate_checks", [])
    return bool(
        unchanged and returns == [0, 0]
        and proof.get("status") == "ROOT_INDEPENDENT_CONSTRUCTION_PASS"
        and proof.get("candidate_sha256") == selected_sha
        and len(rows) == 1 and rows[0].get("label") == "selected"
        and rows[0].get("predicate_pass") is True
        and rows[0].get("candidate_sha256") == selected_sha)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backtrack_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    back = a.backtrack_dir.resolve()
    cfgpath = back.parent / "backtrack-config.json"
    cfg = read(cfgpath)
    selected = read(back / "selected-result.json")
    fit = Path(cfg["trial"])
    report = Path(cfg["parent_fit"])
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pinned = {str(p): sha(p) for p in [VERIFIER, EDGE, OWNER, REFERENCE, cfgpath,
              back / "selected-result.json", Path(selected["candidate"]), fit, report]}
    checks = [
        [sys.executable, str(VERIFIER),
         "--candidate", selected["candidate"], "--base", cfg["state"]["candidate"],
         "--fit-candidate", str(fit), "--fit-report", str(report),
         "--source-owner", str(OWNER), "--source-reference", str(REFERENCE),
         "--output", str(out / "independent-verifier-494.json")],
        [sys.executable, str(EDGE),
         "--config", str(cfgpath), "--support-source", str(fit),
         "--pass-root", str(back), "--passes", "",
         "--selected-result", str(back / "selected-result.json"),
         "--output", str(out / "edge-gradient-check.json")]]
    returns = []
    for i, argv in enumerate(checks):
        (out / f"argv-{i}.json").write_text(json.dumps(argv, indent=2)+"\n")
        with (out / f"check-{i}.log").open("w") as log:
            r = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        returns.append(r.returncode)
    unchanged = all(sha(p) == digest for p, digest in pinned.items())
    # Both original tools determine their own pass; never infer acceptance from
    # process completion alone. Their receipts are retained on failure as well.
    proof_path = out / "independent-verifier-494.json"
    edge_path = out / "edge-gradient-check.json"
    proof = read(proof_path) if proof_path.is_file() else {}
    edge = read(edge_path) if edge_path.is_file() else {}
    result = dict(
        classification="construction-only combined verification",
        candidate=selected["candidate"], candidate_sha256=selected["candidate_sha256"],
        source_pins=pinned, source_pins_unchanged=unchanged,
        tool_returncodes=returns,
        independent_receipt=str(proof_path), edge_receipt=str(edge_path),
        native_admission=False, accepted_physical_steps=0,
        unresolved_surface_pairs=(proof.get("exact_159") or {}).get("candidate_pairs"),
        unresolved_contained_pairs=(proof.get("exact_130") or {}).get("candidate_inside_pairs"))
    # Tools' exit codes and explicit evidence are checked below after examining
    # their retained schema; fail closed if a receipt is absent or incomplete.
    result["passed"] = receipt_passes(proof, edge, selected["candidate_sha256"], returns, unchanged)
    (out / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
