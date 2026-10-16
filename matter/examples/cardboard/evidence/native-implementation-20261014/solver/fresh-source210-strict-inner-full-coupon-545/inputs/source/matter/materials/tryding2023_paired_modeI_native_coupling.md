# Native paired-face Mode-I coupling slice

This implementation couples the `tryding2023_bleached_paperboard_modeI_coupon`
material to an authored, duplicated triangular FEM interface. It is a source-
bounded constitutive and solver slice, not a corrugated-board adhesive recipe.
The calibrated paper is the single-ply bleached clay-coated grade described by
Tryding et al. (2023), with cohesive parameter attribution to Biel et al.
(2022); neither source supplies starch-glue parameters for the Nagasawa
corrugated board.

The compiler admits only explicit reference-displacement FEM with one
source-seeded positive opening: the corresponding side nodes must have
coincident reference coordinates, and the current normal gap must equal the
opening implied by the paper's initialized damage `kappa=0.01`. A paired face
uses three nodes per side and identifies its two adjacent tetrahedra. It is
immutable and requires deformable self-contact to be disabled, since the
current contact path does not distinguish a cohesive pair from a closure
contact row.

For a current face normal `n`, reference area `A0`, and source-evaluated normal
traction `T`, the residual receives `+A0*T*n/3` at each node on side A and the
equal opposite at each node on side B. The matrix-free operator differentiates
both the scalar Mode-I traction and the current face normal. Using reference
area makes the force the derivative of the source potential per reference
area. The local material deformation is the one-dimensional map
`F33=1+delta/h`, where `delta` is the mean paired-node opening and `h` is the
reported sheet thickness.

Accepted face state stores irreversible damage, maximum opening, current
traction, and signed trapezoidal traction-work estimate. Candidate face state is
projected from accepted state and is committed only with a successful FEM
microstep; rejected steps restore the accepted face record. Work is signed,
not an independently identified irreversible-dissipation variable. The
source material itself uses the explicit max-history projection and a declared
`1e-8` complementarity smoothing; its native evaluator documentation bounds
the resulting tangent claim at the history junction.

The runtime rejects nonpositive opening, tangential relative separation, or
nonuniform nodal opening. This intentionally leaves compression, contact on
closure, mixed-mode I/II/III, mode interaction, finite-thickness glue layers,
and a physical separation/deletion cutoff unsupported. No corrugated-glue
strength, fracture energy, rate dependence, or debonded-contact law is inferred
from the bleached-paper coupon. The paired-face runtime check passes 4,004
checks, including 56-step paths on an equal-area one-face/two-tetrahedron
fixture and a three-face/six-tetrahedron refinement. Loading, unloading,
reloading, reopened damage, area work and reaction traction agree with an
independent source-law oracle. Native reaction finite differences reproduce
the source unloading stiffness to about 3.2e-5 relative error on both meshes
in `cohesive/native-003`. The later `native-tests-009/cohesive` repeat also
passes all 4,004 checks; its coarse and refined errors are 3.17262e-5 and
6.77552e-4. Preserve this measured variation rather than reporting the older
smaller error as the repeat's result.
All bulk nodes are prescribed in the refinement path, so this is a
manufactured interface coupling check, not physical board qualification or
free-mesh convergence. The matrix-free tangent is tested through a reaction
finite-difference proxy rather than direct access to the operator.
