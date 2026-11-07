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

The initial runner stopped on its first state: JSON could not serialize a
NumPy boolean. No trajectory was run. The failed invocation and traceback
are retained; a corrected recorder will be separately identified and does
not alter numerical checks.

## Result

The corrected runner passes all seven declared identical-state checks:
energy, residual and requested-tangent differences are exactly zero for
the four retained n=2 and three defined n=4 states. The residual-only n=2
and n=4 fine 10 ms trajectory JSON outputs each match the original output
byte-for-byte, including force bins, impulse, work, clearance and momentum.
The n=2 and n=4 runs took 10.4 s and 27.4 s; the earlier full-tangent
diagnostics took 19.5 s and 46.6 s. These separate execution times are
resource guidance, not controlled performance measurements or a product
speed claim.

The JSON-recorder failure from the initial attempt remains retained. This
check supports using the residual-only evaluator for further independent CPU
reference experiments; it does not prove a spatial reference or pass the
0.25% contact force or impulse gates.

[Evidence](evidence/contact-external-residual-only-20261108/) retains the
source hashes, both output hashes, state checks, trajectories, timeouts,
work and the initial failure. Neither experimental option nor GPU execution
was involved.
