#!/usr/bin/env python3
"""Correct interpretation of reference-F32 metrics in the immutable fit report."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
ROOT=Path("/Users/n/Work/numi-human-connected-surface-069")
TOPOLOGY=Path("/Users/n/Work/294-source-frame-audit-001/connected-interior-reference-fit-001/bistellar-internal-017/run-001/candidate-tetrahedra.npz")
FIT=ROOT/"experimental-interior-fit-008/human_skin_fat_connected_repair.py"
RUN=ROOT/"connected-topology-fit-021/run-002"
REPORT=RUN/"fit-report.json"
CANDIDATE=RUN/"candidate.npz"
BASE_PAIRS=RUN/"baseline-o-self-pairs.npy"
CAND_PAIRS=RUN/"candidate-o-self-pairs.npy"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location("connected_fit_008",FIT);fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
src=np.load(TOPOLOGY);can=np.load(CANDIDATE)
R64=np.asarray(src["reference_points_AP_m"],dtype=np.float64)
R32=R64.astype(np.float32)
X0=np.asarray(src["target_points_AP_m"],dtype=np.float32)
X1=np.asarray(can["target_points_AP_m"],dtype=np.float32)
T=np.asarray(can["candidate_tetrahedra"],dtype=np.int64)
refJ32,_,refq32,_=fit.cell_values_gradients(R32,R32,T)
refJ64,_,refq64,_=fit.cell_values_gradients(R64,R64,T)
startJ32,_,startq32,_=fit.cell_values_gradients(X0,R32,T)
fitJ32,_,fitq32,_=fit.cell_values_gradients(X1,R32,T)
basepairs=np.load(BASE_PAIRS);candpairs=np.load(CAND_PAIRS)
report={
 "classification":"metric_interpretation_correction_not_a_new_fit",
 "as_run_fit_report_sha256":sha(REPORT),"candidate_npz_sha256":sha(CANDIDATE),
 "topology_input_sha256":sha(TOPOLOGY),"fit_helper_sha256":sha(FIT),
 "corrected_reference_geometry_f32":{"reference_j_nonpositive":int((refJ32<=0).sum()),
    "reference_q_below_1e-5":int((refq32<1e-5).sum()),"reference_q_min":float(refq32.min()),
    "reference_j_min":float(refJ32.min()),"reference_j_max":float(refJ32.max())},
 "reference_geometry_f64":{"reference_j_nonpositive":int((refJ64<=0).sum()),
    "reference_q_below_1e-5":int((refq64<1e-5).sum()),"reference_q_min":float(refq64.min())},
 "candidate_vs_reference_f32":{"current_j_nonpositive":int((fitJ32<=0).sum()),
    "current_j_below_0_35":int((fitJ32<.35).sum()),"current_j_above_2_5":int((fitJ32>2.5).sum()),
    "current_q_below_1e-5":int((fitq32<1e-5).sum()),"current_j_min":float(fitJ32.min()),
    "current_j_max":float(fitJ32.max()),"current_q_min":float(fitq32.min())},
 "starting_candidate_vs_reference_f32":{"current_j_nonpositive":int((startJ32<=0).sum()),
    "current_q_below_1e-5":int((startq32<1e-5).sum()),"current_j_min":float(startJ32.min()),
    "current_j_max":float(startJ32.max()),"current_q_min":float(startq32.min())},
 "field_correction":{"as_run_starting_reference_F32_nonpositive_was_direct_reference_determinant_count":True,
    "as_run_starting_reference_F32_q_below_was_mislabeled_candidate_vs_F32_reference_q":True,
    "as_run_emitted_reference_F32_nonpositive_and_q_below_were_mislabeled_candidate_vs_F32_reference_j_and_q":True},
 "exact_coarse_O_self_pairs":{"baseline_count":int(len(basepairs)),"candidate_count":int(len(candpairs)),
    "candidate_pairs_face_ids":candpairs.tolist(),"baseline_pair_sha256":sha(BASE_PAIRS),
    "candidate_pair_sha256":sha(CAND_PAIRS),"predicate":"owner315 exact Float32 indexed-surface pair audit"},
 "stop_condition":"Candidate fails current-cell and coarse-O exact self gates; no further fit was run."}
(RUN/"metric-correction-001.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))
