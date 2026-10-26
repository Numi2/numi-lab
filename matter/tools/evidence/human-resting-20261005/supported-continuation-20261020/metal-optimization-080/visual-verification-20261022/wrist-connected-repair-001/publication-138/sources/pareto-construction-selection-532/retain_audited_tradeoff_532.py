"""Retain a complete audited tradeoff without relabeling legacy merit failure."""
from pathlib import Path
import argparse
import hashlib
import json

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backtrack_dir",type=Path)
    ap.add_argument("iteration",type=int)
    a=ap.parse_args()
    back=a.backtrack_dir.resolve(); cfg=json.loads((back.parent/"backtrack-config.json").read_text())
    p=back/f"pass-{a.iteration:03d}"
    result=json.loads((p/"result.json").read_text())
    jm=result["joint_merit"]; pm=result["point_merit"]
    # This selection does not claim the stricter legacy merit passed.
    if not (jm["after_total_pairs"]<jm["before_total_pairs"]
            and jm["after_point_pairs"]<=jm["before_point_pairs"]
            and jm["maximum_depth_nonincreasing"] is True
            and jm["squared_depth_nonincreasing"] is True):
        raise ValueError("not a depth-safe decreasing-total construction tradeoff")
    point=json.loads((p/"audit-130/summary.json").read_text())
    surface=json.loads((p/"audit-159/report.json").read_text())
    candidate=p/"candidate.npz"
    sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if not (point["candidate_sha256"]==surface["candidate_sha256"]==result["candidate_sha256"]==sha
            and point["complete_surface_audit"] is True and point["exact_PLC_surface_pairs"]==0
            and surface["complete"] is True and surface["bank_count"]==surface["expected_bank_count"]==185):
        raise ValueError("incomplete or mismatched exact audits")
    out=back/"selected-result.json"
    if out.exists():raise FileExistsError("never overwrite an existing selection")
    retained=dict(result)
    retained.update(status="ROOT_RETAINED_PARETO_CONSTRUCTION_TRADEOFF",
        candidate=str(candidate),point_summary=str(p/"audit-130/summary.json"),
        surface_report=str(p/"audit-159/report.json"),parent_fit=cfg["parent_fit"],
        selection_policy=dict(legacy_strict_joint_merit_passed=jm["construction_progress"],
            retained_by="decreasing total pairs, nonincreasing contained pairs and depth moments, complete zero-self audits",
            legacy_execution_and_pass_result_unchanged=True,
            surface_count_increase=jm["after_surface_pairs"]-jm["before_surface_pairs"],
            unresolved_geometry=True,native_admission=False,accepted_physical_steps=0))
    out.write_text(json.dumps(retained,indent=2)+"\n")
    print(json.dumps(dict(candidate_sha256=sha,surface=jm["after_surface_pairs"],
        contained=jm["after_point_pairs"],legacy_merit_passed=jm["construction_progress"],
        requires_independent_construction_and_edge_checks=True)))
if __name__=="__main__":main()
