# Native 450 independent review

This read-only review checks the source142 CGLS capture and its same-protocol capture-off run. It independently rehashes every prediction input for both runs, parses the captured binary snapshot, checks row masking/status/norms, verifies the pretarget trace against run445, compares the captured 450 and 452 output trees and Armijo/true-residual records, and independently reconstructs the directional actions from the retained production tangent capture at run419.

The supported finding is limited: at root1/Newton48, the Newton Armijo loop exhausted 24 healthy trials; an explicitly selected, 64-iteration unregularized CGLS direction then passed the unchanged residual Armijo test at alpha1, reducing the recorded free-node residual from 1.8444375e-4 to 4.85483652e-5. The direction did not inherit the Newton FGMRES certificate. The native CGLS vector snapshot reports status `exhausted`, 64 completed iterations, zero error bits, 846 free nodes in the selected mask, and exact zeros in the field backing region and quiet environment. The native directional `A*x` and `A^T*r` comparisons independently recompute to relative differences 1.44803814e-7 and 1.43712278e-7 against the cached FEM operator reconstructed from run419. These are directional-action checks, not a full operator certificate.

Capture-on and capture-off runs have the same 35 output files byte-for-byte, matching status5 payload, parsed Armijo rows, and parsed true-residual-cycle rows; run450 also matches the first 48 Armijo iterations in run445. Both complete attempts exit with status2, accept only one control step, and report `physical_validation: false`.

The Armijo trace is capped at 64 records: it contains Newton iterations0–47, both direction records at Newton48, and Newton49–62. The final recorded Armijo acceptance is Newton62. No Newton63 Armijo record exists. The true-residual log has 64 recorded cycle rows for each Newton index0–63 and a final-cycle marker at Newton63, but those rows do not reveal its Armijo decision. Therefore the run does not show whether the 64-step nonlinear budget would accept another CGLS fallback at Newton63. A selector that requests the fallback at every Newton could affect that unobserved iteration, so this report does not call an additional run redundant.

The rich operator capture is a separate run at matching world fingerprint, compiled pack, root/Newton index, and dimensions; its captured RHS matches the CGLS snapshot bitwise. The CGLS snapshot has no candidate-node payload, so complete candidate-state byte identity is not established. The run is a failed, rolled-back attempt rather than a completed coupon or physical validation.

Relevant source gates in the frozen snapshot are `base_cgls_runtime.metalinc:22,27-33`, `fgmres.metalinc:5255-5263,5287-5294`, `fem.metalinc:4651-4666,4696-4728`, and `runtime.mm:11911-11939,15812-15829,19162-19176,22904-22914`. The direction helpers operate only with Armijo metric state `w==3`; the accepted primary Armijo path sets `w=1`, and exhausted primary Armijo can set `w=3`. This supports the narrow statement that the fallback direction helper is inactive on observed accepted Newton49–62; it does not establish the unobserved Newton63 outcome.

Run the independent checks with:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/native450-independent-review-001/audit.py
```

The script reads retained artifacts only. It does not run the native solver or alter source/run evidence.
