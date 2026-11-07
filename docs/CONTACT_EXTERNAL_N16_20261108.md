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
