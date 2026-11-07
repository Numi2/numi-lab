# External n=16 P1 comparison

This continues the independently assembled scikit-fem P1 and DOP853
comparison at n=16. The residual-only evaluator has identical states and
full n=2/n=4 trajectories to the full-tangent evaluator in the preceding
study. The immutable fixed area contact law, geometry, material, density,
gravity and initial state are unchanged. This is same-family spatial
refinement, not an independent high-order spatial reference.

## Prediction and resource plan registered before execution

Run full 10 ms trajectories at DOP853 max step 16 and 8 µs, rtol 1e-8 and
1e-10, position atol 1e-12 and 1e-13 m, velocity atol 1000 times greater.
The larger max steps reduce execution cost but are not presumed accurate:
force and impulse temporal checks and comparison to retained n=16 C++ RK4
h=0.5 µs must each meet the original 0.25% limits. Normalize max 1 ms bin
differences by the fine n=16 peak bin and total-impulse differences by fine
n=16 impulse. Require positive sampled gap, nonpositive gaps rejected at
ODE evaluation stages, and momentum imbalance <=0.25% of impulse.

After these prerequisites, compare the earlier external n=8 fine trace
against fine n=16 using the same n=16 peak-bin and impulse normalizers and
the unchanged 0.25% spatial limits. The retained C++ result predicts failure;
passing would not establish n=16 as a converged spatial reference.

The 16-division mesh has 24,576 tets and 4,913 nodes, 8x n=8 tetrahedra.
From measured n=8 full-tangent run time and the n=4 residual-only fraction,
a rough estimate is 6 minutes coarse, 11 minutes fine; actual costs may
differ substantially and are not performance claims. One BLAS thread, no
GPU, sequential 900/1800-second process-group timeouts, stop on failure.
Capture host load and memory before and after each run, exact identities,
stdout, stderr and timeout. On the assessment snapshot the host had 16 GiB
RAM, reported 85% free system memory, 31 GiB free disk and load averages
2.52/2.85/2.81. Native simulation timing is not measured. Preserve failures.

## Result

Both runs completed: coarse 370.6 s and fine 738.3 s. Their maximum force-bin
and total-impulse differences were 1.724e-6 and 7.318e-7. The fine external
trajectory versus retained n=16 RK4 differences were 1.563e-6 and 5.144e-7.
Both comparisons pass the registered 0.25% gates. Minimum sampled gap was
93.310 µm; the ODE evaluator also rejects nonpositive stage gaps. Maximum
global momentum imbalance was 1.112e-14 of fine total impulse. Fine contact
work was -3.09266e-7 J; this is a diagnostic, not an energy verdict.

The external n=8→16 comparison **fails**: 27.238% for maximum binned force
and 6.659% for integrated impulse, normalized respectively by the fine n=16
peak bin and total impulse. These are unchanged 0.25% gates. Agreement with
an independently assembled and integrated n=16 implementation does not
establish same-family spatial convergence or material realism; it does not
resolve the earlier n=16→32 failure.

Host memory-free snapshots ranged from 83% to 86%, and peak reported RSS was
264.3 MB. Load averages at the beginning were 2.15/2.73/2.76, and at the
end 3.53/3.62/3.16. These are resource observations, not a performance
measurement. The [evidence](evidence/contact-external-n16-20261108/) retains
host snapshots, exact inputs and code hashes, analytic controls, raw
trajectories, timeout and comparison. No GPU was used.
