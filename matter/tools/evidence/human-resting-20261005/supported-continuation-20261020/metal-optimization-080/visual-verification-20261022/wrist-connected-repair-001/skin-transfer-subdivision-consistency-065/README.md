# Skin baseline prolongation consistency

Optional compiler-authored parent interpolation keeps the presentation baseline consistent when a physical skin edge is subdivided. It adds no mechanical coordinates, forces, or GPU row ABI. Existing callers retain their arithmetic. Invalid, cyclic/forward, nonnormalized or rest-inconsistent ancestry is rejected.

The mixed-donor regression reproduces a 3.54 mm subdivision discrepancy and reduces it below 1 nm. Three existing transfer regressions pass. Actual elbow rest points and mechanical deltas reproduce all 79 midpoint rows exactly. These are transfer and ancestry checks, not a natural-joint-motion or anatomy qualification. The native D16 source initialization still rejects separately for a duplicated donor-local registration residual; that evidence is retained unchanged.
