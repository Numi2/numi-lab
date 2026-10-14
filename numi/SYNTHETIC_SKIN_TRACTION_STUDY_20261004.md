# Synthetic wound-lip traction study — preregistration

Registered 2026-10-04 before collecting this study's simulation samples.
Purpose: exploratory native-simulation comparison; this is not a human-skin,
animal-skin, surgical-device, or clinical study.

## Question and model

On one fixed, deterministic synthetic skin-wound coupon, does paired external
traction at two authored bite sites reduce the top-surface mean wound-lip gap
by at least one contact slop compared with an otherwise identical zero-force
trajectory?

The coupon is authored as a homogeneous 30 x 24 x 1.5 mm block with a 16 mm
incision, 0.6 mm initial gap, and a 30 x 32 x 8 tetrahedral grid. The material
is the uncalibrated neo-Hookean research material in
`matter/materials/synthetic_skin_wound.nmatter` (SHA-256
`379945f593f12396b44c7148026f0250344a86278e01b7b26cb5f6d823a51e35`). No
thread, needle, robot, wound healing, or clinical tissue behavior is modeled.

## Prediction and refutation

Prediction: at one or more loaded samples, the zero-force control mean gap
will exceed the loaded-arm mean gap by at least 10 micrometres. This threshold
equals the authored 10 micrometre contact slop. The loaded minimum gap must
remain in the original lip order (nonnegative signed y separation); solver
validity, active tetrahedra, and mass conservation must pass the instrument's
existing checks. A valid pair with less than
10 micrometres of gap reduction refutes the predicted resolvable closure for
this exact model and force protocol. It will be retained as a negative result.

The zero-force control must remain within 1 micrometre of its initial mean gap
throughout the trajectory. Excess control drift, a failed state, invalid
deformation certificate, tetrahedron loss, or relative mass error above
`1e-5` makes the pair inconclusive rather than supporting the prediction.
Solver residual and rebound after unloading are recorded descriptively, with
no pass threshold beyond native validity checks.

## Controlled protocol

- Two matched arms, run in this fixed order: zero-force control, then loaded.
- Both arms start from the same compiled coupon, material, timestep, solver,
  deterministic mode, zero gravity, and initial geometry.
- The loaded arm applies 10 mN per bite site, with equal and opposite forces
  on the two lips at each site. The sites are at x = -4 mm and +4 mm; force
  is distributed over the authored 1.6 mm half-width patches with the same
  1 mm Gaussian weighting.
- Each arm has 60 accepted 1 ms steps. The force ramps over steps 0-23, holds
  through step 39, and releases over steps 40-59.
- The sole experimental unit is this paired deterministic condition. The 60
  samples are repeated measurements within trajectories, not 60 replicates.
  No population uncertainty or general material effect will be inferred.
- The maximum budget is one 60-step run per arm. Any failed or interrupted
  attempt is preserved and not silently rerun.

## Measurement and analysis

The primary observable is the arithmetic mean Euclidean separation of all
paired top-incision lip nodes, in metres, compared at matching accepted steps.
Report the per-step control and loaded values, their paired difference, the
minimum loaded gap, the final loaded-control difference, maximum tissue
displacement, minimum deformation determinant, maximum nonlinear residual,
active tetrahedra, relative mass error, and GPU time. A video may be rendered
only from the retained accepted loaded-arm lip coordinates and samples.

The native Numi dispatcher is not installed on the Mac mini, so this retained
plan, raw CSVs, logs, binary/library hashes, and analysis are the equivalent
local record. The executable remains the owning Matter probe; no host-side
physics is added.
