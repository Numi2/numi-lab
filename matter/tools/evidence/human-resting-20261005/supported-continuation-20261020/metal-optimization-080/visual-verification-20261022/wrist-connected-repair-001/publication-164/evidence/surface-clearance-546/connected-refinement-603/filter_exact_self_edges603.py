from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
WORK = ROOT / "D/surface-clearance-546/connected-refinement-603"
BASE = ROOT / "D/surface-clearance-546/component-topology-599/run-001"
IN_CAND = BASE / "lineage/candidate.npz"
IN_REPORT = BASE / "full-check/audit-159/report.json"
BAD_SELF = WORK / "baseline-599-affine-001/baseline-check-001/audit-130/surface-self-pairs.npy"
FIRST_REPORT = WORK / "baseline-599-affine-001/report.json"
OUT = WORK / "baseline-599-affine-filtered-002"
CANDIDATE_SHA = "05b499db61f22492f4ee4611951bdb4e02e196a3175205dda862bc08befc7729"
CROSSING_SHA = "2eedcda0de5ec8ce7cb543c7b489f4e0579271a7b4b76cc265409ca7fd76a785"
BAD_SELF_SHA = "7956b3dc21480eba13fb527892f656da6e2bbe90065b65fa33d5cb8d012bc1d6"
FIRST_REPORT_SHA = "036168d8d5f3af3ebc266887e1e951ca6050383c82c4f42ea08b0df12dcdca79"
REF_SHA = "0853407a1ce7fb9aed2d91a51ea712f8a85dadfcfb4b048e29979453dabab4dc"
PREP_DIR = ROOT / "D/surface-clearance-546/edge-only-refinement-585"
sys.path.insert(0, str(PREP_DIR))
import prepare_edge_only_refinement585 as prep
import edge_only_refinement585 as owner

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def array_sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8).tobytes()).hexdigest()

def main():
    for p, expected in ((IN_CAND, CANDIDATE_SHA), (IN_REPORT, CROSSING_SHA),
                        (BAD_SELF, BAD_SELF_SHA), (FIRST_REPORT, FIRST_REPORT_SHA)):
        actual = sha(p)
        if actual != expected:
            raise SystemExit(f"source pin mismatch {p}: {actual}")
    source = {k: v.copy() for k, v in np.load(IN_CAND, allow_pickle=False).items()}
    crossing = json.loads(IN_REPORT.read_text())
    first = json.loads(FIRST_REPORT.read_text())
    self_pairs = np.asarray(np.load(BAD_SELF), dtype=np.int64).reshape(-1, 2)
    expected_pairs = np.asarray([[43557, 43614], [43558, 43614], [43576, 43614],
                                 [43577, 43614], [43579, 43614]], dtype=np.int64)
    if not np.array_equal(self_pairs, expected_pairs):
        raise SystemExit("exact self-witness set changed")
    dropped = np.asarray([[16015, 19715], [16015, 20870], [19715, 20870]], dtype=np.int64)
    dropped = np.sort(dropped, axis=1)
    selected, policy = prep.choose_high_incidence_edges585(
        source["source_O_faces"], crossing, max_edges=None)
    selected = np.sort(selected, axis=1)
    if len(selected) != 1042:
        raise SystemExit(f"expected 1042 all-crossed-face edges, found {len(selected)}")
    for edge in dropped:
        if not np.any(np.all(selected == edge, axis=1)):
            raise SystemExit(f"self-witness midpoint edge absent from original selection: {edge.tolist()}")
    retained = np.asarray([e for e in selected if not np.any(np.all(dropped == e, axis=1))],
                          dtype=np.int64).reshape(-1, 2)
    if len(retained) != len(selected) - len(dropped):
        raise SystemExit("operation removal count mismatch")
    policy.update({
        "mode": "all_edges_of_validated_crossed_faces_minus_exact_self_witness_edges",
        "candidate_edge_count": int(len(selected)),
        "selected_edge_count": int(len(retained)),
        "excluded_candidate_edge_count": int(len(dropped)),
        "unselected_edge_rows_remain_unresolved": True,
        "excluded_edges": [
            {"edge": e.tolist(), "reason": "midpoint row participates in exact 130 self-intersection witness"},
            {"edge": e.tolist(), "reason": "midpoint row participates in exact 130 self-intersection witness"},
            {"edge": e.tolist(), "reason": "midpoint row participates in exact 130 self-intersection witness"}
        ],
        "witness_self_pairs": self_pairs.tolist(),
        "witness_rows_sha256": sha(BAD_SELF),
        "first_attempt_candidate_sha256": first["output_candidate_sha256"],
        "selection_only_filter": True
    })
    if OUT.exists():
        raise SystemExit(f"refuse to overwrite {OUT}")
    arrays, report = owner.build_refined_arrays585(
        source, crossing,
        source["area_reference_037_points_world_m_f64"],
        source["area_reference_037_source_O_faces"],
        candidate_sha256=CANDIDATE_SHA,
        crossing_report_sha256=CROSSING_SHA,
        area_reference_sha256=REF_SHA,
        selected_edges_override=retained,
        selection_policy=policy)
    OUT.mkdir(parents=True)
    cand = OUT / "candidate.npz"
    with cand.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = dict(report)
    report.update({
        "input_candidate_path": str(IN_CAND),
        "input_candidate_sha256": CANDIDATE_SHA,
        "first_attempt_report_path": str(FIRST_REPORT),
        "first_attempt_report_sha256": FIRST_REPORT_SHA,
        "first_attempt_self_pairs_path": str(BAD_SELF),
        "first_attempt_self_pairs_sha256": BAD_SELF_SHA,
        "filtered_edges": dropped.tolist(),
        "filtered_edge_rows_sha256": array_sha(dropped.astype("<i8")),
        "output_candidate_path": str(cand),
        "output_candidate_sha256": sha(cand),
        "geometry_moved": False,
        "geometry_admitted": False,
        "containment_or_clearance_claim": False,
        "physical_steps": 0,
        "selection_policy": policy,
    })
    rep = OUT / "report.json"
    rep.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in (
        "status", "candidate_sha256", "crossing_bank_triangle_pair_count",
        "unique_crossed_parent_face_count", "affected_parent_face_count",
        "output_candidate_sha256", "face_centroid_count", "geometry_admitted",
        "physical_steps", "selected_edge_subset_policy")}, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
