from pathlib import Path
import json,hashlib,numpy as np
Q=Path("/Users/n/cardboard-implementation-20261014-host-macmini")
D=Q/"analysis/native541-critical23-branch-adjudication-001"
R=json.loads((D/"report.json").read_text())
old=json.loads((Q/"analysis/native541-critical23-adjudication-001/report.json").read_text())
rows=old["endpoint_derivatives"]["all_pairs"]
by={}
for x in rows: by.setdefault((x["tet"],x["basis_column"]),[]).append(x)
fd={}
for key in ("native_centered_fd_relative_to_strict_tangent","strict_realized_endpoint_secant_relative_to_tangent","native_relative_to_strict_secant"):
 a=np.array([x[key] for x in rows],dtype=float)
 good=[x for x in rows if x["signal_exceeds_endpoint_error_sum"]]
 av=np.array([x[key] for x in good],dtype=float)
 fd[key]={"all_46_min":float(a.min()),"all_46_median":float(np.median(a)),"all_46_max":float(a.max()),
  "signal_above_discrepancy_n":len(good),"signal_above_discrepancy_min":float(av.min()),"signal_above_discrepancy_median":float(np.median(av)),"signal_above_discrepancy_max":float(av.max()),
  "all_46_le_1e-3":int((a<=1e-3).sum()),"all_46_le_1e-4":int((a<=1e-4).sum()),"all_46_le_1e-6":int((a<=1e-6).sum()),
  "both_powers_le_1e-3_of_23_pairs":sum(all(x[key]<=1e-3 for x in v) for v in by.values()),
  "both_powers_le_1e-4_of_23_pairs":sum(all(x[key]<=1e-4 for x in v) for v in by.values()),
  "both_powers_le_1e-5_of_23_pairs":sum(all(x[key]<=1e-5 for x in v) for v in by.values()),
  "both_powers_le_1e-6_of_23_pairs":sum(all(x[key]<=1e-6 for x in v) for v in by.values())}
R["finite_difference_scope"]={"pair_count":46,"tet_basis_pair_count":23,"powers_per_pair":2,
 "native_signal_exceeds_endpoint_discrepancy_sum":sum(x["signal_exceeds_endpoint_error_sum"] for x in rows),
 "native_signal_not_above_endpoint_discrepancy_sum":sum(not x["signal_exceeds_endpoint_error_sum"] for x in rows),
 "error_metrics":fd,
 "qualification":"No finite-difference gate is qualified from this selected batch. The endpoint signal/error comparison only distinguishes 30 potentially informative pairs from 16 discrepancy-dominated pairs; among the 30, native centered FD vs strict tangent exceeds 1e-3 for 20 and exceeds 1e-6 for all 30. Two powers per selected pair are not an all-nine-directions adjacent-power study. Strict FP64 realized secants are a separate reference self-check and do not validate native arithmetic noise."}
R["indicator_erratum"]="The earlier report field native.strict_reference_branch_mismatches is a yield-sign indicator mismatch, not proof of switching a discontinuous constitutive branch. Source equations are smooth at y=0 because fb_epsilon>0; tensile hardening weights use sqrt(sigma^2+width^2). The FP64 reference's positive_y formula branch is a cancellation-safe Jacobian evaluation choice, algebraically equivalent to the nonpositive_y formula, and the den>1e-14 rationalized-residual choice is reference-only. Retain the observed y-sign disagreements as numerical indicators."
(D/"report.json").write_text(json.dumps(R,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(hashlib.sha256((D/"report.json").read_bytes()).hexdigest())
print(json.dumps(R["finite_difference_scope"],indent=2))

