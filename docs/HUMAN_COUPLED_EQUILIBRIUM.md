# Coupled Human equilibrium preparation

`NumiHumanMuscleEquilibrium` compiles a stationary source pose and bounded
muscle recruitment outside the physics hot loop. It retains NHEQ1 ideal
equalities and geometrically admitted unilateral source stops. It does not
substitute for NHEQ2/NHLIM1 source-compliant dynamics or active control.

The source-ordered recruitment initializer is followed by a coupled,
bound-constrained Gauss-Newton correction. The native active-set quadratic
proposal uses diagonal scaling and a `1e-12` numerical damping term. Local
evaluations of the exact static muscle law supply activation derivatives;
the current mass/equality/reaction tangent retains cross-muscle coupling.
Exact nonlinear muscle forces and a fresh physical reaction certificate
decide acceptance. Numerical proposal damping is not tissue compliance.

Posture search compares scalar trials and simultaneous posture/activation
trials. Its load scales remain fixed while forming a search direction so
renormalization cannot erase a changing spring load. Accepted objective
evaluation uses the complete candidate state. Loaded stops remain fixed
until their reactions release. The existing native support-placement compiler
fits loaded contact planes after a block proposal; root coordinates, source
limits, and configured displacement bounds remain authoritative. Recruitment
continues from the candidate's activation state before posture selection.

The owning contact solver receives warm reactions by source DoF, force sign,
and current mass scaling. Its default `1e-11` KKT tolerance, 200-iteration
budget, and independent `1e-8` unregularized complementarity check are
unchanged. A failed initial reaction solve returns `constraintSolveFailure`
without replacing the destination. Failed numerical search evaluations are
counted and rejected; the valid current state remains available. They are
not reported as accepted physical states or successful solver evaluations.

`numi_human_static_support` covers coupled force sharing, actuator bounds,
source-order reversal, replay, a two-coordinate analytic posture equilibrium,
source stops, equality-dependent reactions, and support geometry.

The visual probe's `--whole-body-support-certificate` accepts
`--whole-body-activation-sweeps` and `--whole-body-pose-sweeps 0..256`.
The latter sets the existing posture-search budget; it changes no convergence
threshold. Zero disables posture search after any explicitly requested initial
support placement. The probe exports:

- `compiled_equilibrium_q`: complete FP64 compiled coordinates;
- `compiled_equilibrium_reactions`: full generalized acceleration and forces;
- `compiled_equilibrium_search_trace`: initialization, accepted posture updates,
  and final objective/residual, with numerical rejection counts;
- `compiled_equilibrium_muscles`: FP64 reference activation, its FP32 transport
  values, reference fibre lengths, and signed source actuator forces.

Reference fibre length follows the existing muscle architecture: a legacy
rigid-tendon law reports path length. These exports describe offline preparation.
They do not qualify FP32 loaded dynamics, registered tissue, physiological
calibration, standing, walking, or performance. A returned finite state may
still have `balanced=false`; balance requires the unchanged normalized RMS
threshold of `0.05`.
