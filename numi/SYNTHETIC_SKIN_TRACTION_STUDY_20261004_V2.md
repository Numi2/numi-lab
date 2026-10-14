# Synthetic wound-lip traction study — amended preregistration

Registered 2026-10-04 after the invalid no-sample preflight recorded in
`SYNTHETIC_SKIN_TRACTION_PREFLIGHT_FAILURE_20261004.md`, and before collecting
any new trajectory samples. Purpose: exploratory native-simulation
reattempt of the original paired condition; this is not confirmation of a
biological or population-level effect.

## Amendment and measurement instrument

The first build stopped because the probe incorrectly required compiled FEM
state capacity to equal authored mesh size. The coupon intentionally reserves
64 node slots and 128 tetrahedron slots for topology mutation. No physics step
or GPU dispatch occurred. The revised probe resolves authored node identities
from active `NMFEMTopologyNodeGPU` records, checks the rest coordinates, and
has a CPU-only `--validate-only` mode. The v1 attempt remains invalid and
unchanged; the new source and executable have separate hashes and a new
isolated run directory.

Before either trajectory, `--validate-only` must verify that every authored
node maps uniquely to an active compiled node with rest-position error no
greater than 0.1 micrometre, that all top incision lip nodes and both bite
patches map, and that the capacity includes the declared reserved slots. This
preflight performs no Metal dispatch and is not an experimental trial.

## Question, model, and prediction

On one fixed, deterministic synthetic skin-wound coupon, does paired external
traction at two authored bite sites reduce the top-surface mean wound-lip gap
by at least one contact slop compared with an otherwise identical zero-force
trajectory?

The model remains the homogeneous 30 x 24 x 1.5 mm coupon, 16 mm incision,
0.6 mm initial gap, and 30 x 32 x 8 authored tetrahedral grid. The material
remains the uncalibrated neo-Hookean model at
`matter/materials/synthetic_skin_wound.nmatter`, SHA-256
`379945f593f12396b44c7148026f0250344a86278e01b7b26cb5f6d823a51e35`. No
thread, needle, robot, wound healing, or clinical tissue behavior is modeled.

Prediction: at one or more loaded samples, the zero-force control mean gap
will exceed the loaded-arm mean gap by at least 10 micrometres, equal to the
authored 10 micrometre contact slop. A valid pair with less reduction refutes
resolvable closure for this exact model and force protocol. It remains a
single deterministic paired condition, not a general material claim.

## Controlled protocol and validity

- Run one 60-step zero-force control, then one 60-step loaded trajectory.
- Hold coupon, material, deterministic mode, 1 ms timestep, zero gravity,
  solver, and initial geometry fixed between arms.
- Apply 10 mN per bite site in the loaded arm, equal and opposite on the two
  lips at x = -4 mm and +4 mm, distributed over the 1.6 mm half-width patches
  with 1 mm Gaussian weighting.
- Ramp over steps 0-23, hold through step 39, and release over steps 40-59.
- The zero-force mean gap must drift no more than 1 micrometre. Every sample
  must preserve lip order, a valid determinant above 0.20, all authored
  tetrahedra active, and relative mass error at or below `1e-5`; any failed
  accepted-state check makes the pair inconclusive.
- Report residuals and rebound descriptively. Sixty within-trajectory samples
  are repeated measures, not independent replicates. Do not infer uncertainty
  or clinical relevance.
- Maximum budget: one trajectory per arm after successful CPU-only preflight.
  Preserve any failed or interrupted attempt and do not silently rerun it.

The primary observable is the arithmetic mean Euclidean separation of all
paired top-incision lip nodes, in metres, compared at matching steps. Retain
both arm CSVs, complete logs and hashes, and report paired differences,
minimum signed lip gap, maximum tissue displacement, minimum determinant,
maximum solver residual, active tetrahedra, relative mass error, and GPU time.
Render only from the accepted loaded-arm coordinates and samples.

The `numi` dispatcher is unavailable on the Mac mini. This preregistration,
pre-run build binding, raw outputs, and analysis are the retained local record;
the native Matter executable remains the owner of the simulation.
