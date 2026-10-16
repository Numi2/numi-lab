# Selected FEM operator and residual comparison

`NM_FEM_OPERATOR_FD_SELECTOR=<control-step>:<zero-based-Newton-iteration>`
selects one bounded diagnostic before the native Newton correction is applied.
Both components must contain decimal digits. With the selector unset, no FD
arenas are reserved and no diagnostic dispatches are encoded.

For the actual velocity correction `dV`, the diagnostic applies the production
FEM element tangent and FGMRES node gather to compute `A*dV`. It separately
reassembles the production mechanical impulse residual at private candidates
with `v +/- epsilon*dV` and `x +/- dt*epsilon*dV`. On free nodes it compares
`A*dV` against `-(R_plus-R_minus)/(2*epsilon)`, for epsilon 1/32 and 1/64.
The sign follows the native residual convention `R = dt*f_internal-M*(v-v_old)`
and mechanical operator `A = M-dt^2*K`. The recorded `position_delta` is the
scale factor `dt*epsilon`; `maximum_position_perturbation` is the largest nodal
Euclidean length of `dt*epsilon*dV` in metres.

`NM_FEM_OPERATOR_FD_EPSILON_SCALE` optionally multiplies both epsilon values.
Only the exact strings `1`, `2`, `4`, `8`, and `16` are admitted; the default is
`1`. This changes the diagnostic perturbations only. Comparing scales helps
expose finite-difference sensitivity, but larger perturbations also span more
nonlinear and active-set changes.

The scope is one immutable explicit FEM object without contact, mixed fields,
cohesive interfaces, topology mutation, learned materials, or coupled Human or
vascular unknowns. Limits are four environments, 4,096 nodes, 8,192 tetrahedra,
and material state stride 1–32. Every authored nodal mass must be finite and
strictly positive; immutable runtime restore binds accepted mass to that layout.
This keeps the residual and operator row spaces consistent when nodes are
released or gripped. Allocation checks the complete diagnostic and
reused Armijo scratch estimate against 32 MiB; readback is capped at 16 KiB.

All candidate, force, residual, operator-action and status outputs are private.
The tangent and residual use the same accepted material history and authored
tetrahedron frame/reference data. A bitwise before/after comparison checks
those inputs and production statuses. The diagnostic precedes pure-FEM
residual Armijo and may run with `NM_FEM_RESIDUAL_ARMIJO_ROOT=all`: subsequent
Armijo trials reinitialize their reused scratch before reading it. Capsule and
BOX Armijo selectors remain excluded from this diagnostic's allocation scope.

The JSON trace reports paired stage statuses, minimum J, norms, relative error,
dot product/cosine, input integrity, and perturbation size. Failed stages and
nonfinite values produce null metrics. A zero correction is explicitly marked;
its undefined cosine cannot be read as a successful tangent comparison.
Unsupported scope is logged as skipped. A requested iteration outside the
encoded budget is logged as not captured. No diagnostic field certifies a
solve, and no tolerance, material parameter, or acceptance rule changes.

For the retained no-contact nx16 protocol, the selected diagnostic is used with
`NM_FEM_RESIDUAL_ARMIJO_ROOT=all` and `NM_FEM_OPERATOR_FD_SELECTOR=19:3`.
Run a separately sealed baseline with the FD selector unset, then compare the
native histories and meshes. A finite-difference discrepancy can reflect local
projection error, active-set changes, or FP32 perturbation resolution; two
epsilon values alone do not identify the cause or establish convergence.

The frozen-031 scale-1 capture measured relative operator/residual differences
of 1.022397 at epsilon 1/32 and 1.024829 at epsilon 1/64. The corresponding
operator-action/centered-FD cosines were -0.067131 and 0.001745; paired stages
were valid with minimum J about 0.9952. Both the selector-off baseline and the
diagnostic accepted 19 native steps and rejected step 19. Their CSV, result
JSON, compiled material pack, and all 41 exported mesh hashes were identical,
so the trace did not alter the native path. The second environment's correction
was zero, making its zero error a vacuous control. These measurements establish
a discrepancy at this selected state, not its cause or a general tangent claim.

A hash-bound CPU audit of the last accepted state (step 18) found some tetrahedra
near the Hill yield surface. That makes yield-transition crossing a testable
hypothesis, but the audit has neither the Newton-3 candidate nor the two
projected perturbation states; it does not show that they cross yield or explain
the measured discrepancy. Keep that hypothesis separate from the observed
operator/FD values.

Measured results and their exact source/runtime bindings belong in the
[native evidence report](../examples/cardboard/native-damage-and-assembly.md).
This instrument is a numerical consistency check, not an independent material
law or physical cardboard calibration.
