# External diagnostic residual-only evaluator

DOP853 uses only the energy gradient, but the validated external assembly
also computed a full tangent-vector at every ODE evaluation. The diagnostic
can avoid that unused calculation while retaining the same constitutive law,
nodal plane energy, residual assembly, state, solver and tolerances. This
change concerns only the independent CPU research reference, not Matter or
either disabled experimental option.

## Checks registered before execution

First compare the original and residual-only evaluators at identical n=2
retained states 0, 150, 200 and 319, and at n=4 initial, uniform active
plane, and seeded nonuniform active geometries. Relative L2 force differences
must be <=1e-12 and absolute energy difference <=1e-12 J, and all gaps must
remain positive. Preserve actual values. The reduced evaluator must return
the original tangent when requested; the omitted tangent is never a reference
for tangent qualification.

Then rerun the fine DOP853 10 ms n=2 and n=4 trajectories with the respective
same mesh, tolerances and max step. Compare each 1 ms force bin and integrated
impulse to the retained original external trajectory; both relative errors
must be <=1e-10 using its peak bin and total impulse. Require positive
sampled gap, momentum balance <=0.25% impulse, and contact work difference
<=1e-10 J. Use sequential 120 s and 180 s timeouts, one BLAS thread, no GPU;
stop on any failure. Retain both implementations, outputs and failure. These
are implementation parity and resource checks, not an altered 0.25% spatial
force/impulse qualification gate or a production performance test.
