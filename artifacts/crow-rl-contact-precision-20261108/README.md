# Crow contact precision diagnosis

The substep-seven native rejection reproduces byte-for-byte. Its scaled
residual is 5.1471453e-8. Native replay also rejects the rounded FP64 reference
and the best maximum-norm lattice candidate. The prior substep-two positive
control still accepts its warm state with zero Newton iterations and exact
velocity bytes. No production source, native setting, tolerance or policy
changed in this slice.

The FP32 FMA emulator reproduces all 480 captured native rows exactly. A
compensated sum correctly rounds every row but reduces the largest row error
only 1.44 times, below the predicted eightfold reduction. At the same velocity,
reprojection using those rows still has residual about 7.06e-8. Compensated
row sums alone do not solve this problem.

The predeclared adjacent, coupled-coordinate, lattice and dual-guided searches
all fail to find a point below 1e-10. Their best results are preserved. Failure
to find a point is not used as a proof of impossibility. The best dual-guided
point has independently computed residual 1.5885271e-10; native iteration from
it ends at 5.61687e-8.

An exact rational witness gives a scaled lower bound of 1.1510348e-10 on the
stipulated constant-branch/base-spacing lattice in the 1e-4 box. An independent
struct/Fraction implementation verifies all 23 identities, phase, gradient and
denominator bound without LLL, matrix inversion or the producer's LDL solve.
The first, weaker witness was below the gate; that failed prediction is retained.

Exact interval and exponent-bin checks discharge the branch and spacing
hypotheses for every actual FP32 velocity in a smaller box: radius 1/40960000
(2.44140625e-8 per generalized-velocity coordinate) around the rounded
reference. The certified lower bound there is 1.1726369e-10, above the actual
compiled tolerance. Excluded coordinates can vary within that box; their
dual coefficients vanish exactly. This is a local proof only. The original
native failed velocity is outside this box, so the prediction that the proof
would cover that velocity failed. No global FP32 impossibility, inevitable
native rejection, whole-world accuracy or ecological qualification is claimed.

The result rules out reaching the gate by merely improving local search inside
that proven neighborhood. The next solver work must address the precision
representation and arithmetic contract while checking actual physical outcomes.
This evidence does not authorize a tolerance relaxation or a policy promotion.

Plans were committed and pushed before observations. All failed predictions
and the initial missing-parent preparation error are retained. The field
candidates_seen in chebyshev-search-result.json counts distinct iteration-best
states, not all candidate evaluations. The payload and companion Git-recovery
directory preserve exact sources, numerical inputs, native shader/binary,
captures, logs, results, executed recovery and scoped cleanup.
