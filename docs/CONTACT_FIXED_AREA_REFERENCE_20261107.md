# Fixed reference-area FEM–plane contact: first temporal qualification

The new law remains an opt-in research instrument. It is not a qualified
control or a replacement for Matter's default contact law. The directional CCD
and augmented-Lagrangian options remain disabled by default. This experiment
does not alter the earlier contact-onset or CCD evidence.

The narrow fixture uses a flat, coplanar plane-facing exterior FEM patch.
Reference-surface lumped quadrature gives each node one third of each projected
face area. For positive gap g below d, its energy per reference area is
-k(g²-d²)² log(g²/d²), with fixed k = 1e16 N/m⁵ and d = 1e-4 m.
The area multiplies the same energy, force and tangent. This does not implement
general deformable surface quadrature. Unsupported geometry is rejected.
The implementation and its area are fixed independently of timestep and mesh.
The scope is a single static plane, reference-displacement FEM, and no friction
or coupling; it is disabled by default.

An independent FP64 scalar pressure–gap check gives 9.6970704133e-5 m
equilibrium gap under 0.086328 N on 0.0004 m², with energy derivative checked
by finite difference. This only validates the uniform scalar contact law: it
is not an independent elastic FEM impact reference. The CPU quadrature check
verifies constant total area across four triangulation resolutions and rejects
a tilted patch.

## Native temporal test

Four 10 ms runs used the same physical law, 2 divisions per axis, 0.2 mm initial
gap, 16 Newton and 64 FGMRES iterations, and the physical residual policy.
Plan, source commit, binary and metallib hashes, detailed traces, diagnostics,
and analysis are in [the retained evidence](evidence/contact-area-reference-20261107/).
The observed source commit is cac0c53c, Apple M6. Tests were sequential and
instrumented; their timings are not performance claims.

| Coarse h | Fine h | Peak 1 ms-bin force change | Integrated impulse change |
| --- | --- | --- | --- |
| 0.5 ms | 0.25 ms | 20.27% | 1.918% |
| 0.25 ms | 0.125 ms | 14.86% | 0.998% |
| 0.125 ms | 0.0625 ms | 23.92% | 0.579% |

Every rollout completed all steps and met the instrument's local residual,
clearance, momentum balance, rollback and deterministic replay checks.
Nevertheless, **none of the force comparisons meets the predeclared 0.25%
reference gate**. Contact episodes also differ; the two finest runs show
release and renewed contact at different times. Completion and residual checks
do not establish accurate force evolution. The scalar law check is not enough
to decide whether the apparent oscillatory behavior is physically correct.

At the time of the native tests, no mesh qualification, hard-contact limit,
independent dynamic reference, near-contact initial-state qualification, or
accepted-time performance comparison had been performed. The necessary next investigation is a separately
verified physical dynamic reference and refinement that resolves force and
release behavior without changing this law or relaxing its gates. The retained
failed temporal series must remain part of that assessment.

## CPU diagnosis of the changing episodes

The retained traces show the first nonzero contact moving from the 4.00 ms
step at h = 0.5 ms to the 4.4375 ms step at h = 0.0625 ms. A continuous
point-mass free-fall onset for the same starting gap is 4.515 ms. The fixture's
discrete kinematic equation is backward Euler, so its onset is step-quantized.

A separate FP64 point-mass calculation uses the same gravity, total mass,
reference area and pressure law. Its linearized contact stiffness at static
equilibrium is 56,416 N/m, giving a 2.482 ms point-mass period. The exact
backward-Euler amplification of an undamped linear oscillator predicts
amplitude retained per period of 9.3%, 18.8%, 38.8% and 61.2% at the four
timesteps, respectively. The independent nonlinear backward-Euler point-mass
solutions also change contact episodes as h decreases. A separate explicit
RK4 integration of that *point-mass analogy* changes its 1 ms binned force by
1.53e-10 and its total impulse by 8.32e-11 under the selected 1 µs to 0.5 µs
refinement. Scripts and output are retained in the evidence directory.

This demonstrates a plausible resolution and numerical-damping mechanism, not
a measured modal analysis of the FEM pad or proof of a solver defect. The
point-mass model has no internal deformation modes and does not reproduce the
pad's force curve; it cannot qualify that curve. No correction to the contact
law, acceptance tolerance or integrator follows from this analogy alone. The
testable next step is an independent evaluation of the actual FEM dynamics,
then temporal refinement at fixed mesh and law with the same force, impulse,
release and state gates.


## Independent FP64 FEM dynamics

A later CPU-only implementation independently constructs the same 2-division
27-node, 48-tetrahedron mesh and integrates its Neo-Hookean semidiscrete
equations with explicit FP64 RK4. The model uses the same density, gravity,
initial gap, area-lumped plane force and fixed regularized law. At a perturbed
state, its force agrees with an independent energy finite difference to
1.79e-8 relative to the peak nodal force. The first finite-difference check
used a larger perturbation and failed its threshold; its result is retained.

Over the fixed 10 ms duration, RK4 h = 1 µs versus 0.5 µs changes the maximum
1 ms binned force by 1.98e-9 and integrated impulse by 4.33e-10, below the
0.25% temporal-reference gate. Momentum-balance errors are below 1.2e-17 N s.
This supports temporal convergence of this *independent semidiscrete model
on this fixed mesh*, not mesh convergence or physical material validation.

Compared with the finest CPU result, native h = 0.0625 ms differs by 50.90%
in the maximum peak-normalized 1 ms force bin and 8.19% in total impulse.
The discrepancies are not monotonic over the tested native timesteps. The
fine CPU trajectory has several distinct contact episodes, so release is not
in itself evidence of error. The tested native trajectories remain unqualified
and cannot establish a reliable force control. Backward Euler damping is a
plausible contributor, but these comparisons cannot isolate the cause or
establish bitwise equivalence of independently compiled model constants.
The FP64 implementation is a diagnostic reference, not a runtime fallback.

## Independent backward-Euler parity

A second independent FP64 implementation solves the same semidiscrete model
using backward Euler with a dense finite-difference Jacobian and a distinct
Newton/line-search implementation. At the four retained native timesteps,
it reproduces every retained step's force, minimum gap, centre of mass and
momentum. At 0.0625 ms, maximum discrepancies were 7.70e-6 N, 1.06e-9 m,
0.785e-9 m and 1.24e-9 N s respectively. The CPU equations' maximum nonlinear
position residual was below 8.2e-14 m over the four runs. This comparison
strongly supports backward-Euler discretization as the cause of the large
difference to the refined semidiscrete trajectory for this fixture; it does
not establish parity for other contact or coupling paths.

A subsequent bounded native capture recorded all node positions and velocities,
compiled reference positions, masses, areas and tetrahedron connectivity.
At 31.25 and 15.625 µs, all 320 and 640 steps completed, and the instrument's
replay and forced-rollback checks passed. At 15.625 µs the worst per-component
native–FP64 differences were 26.9 nm in position and 9.00e-5 m/s in velocity;
normal force differed by at most 0.138 mN. Re-evaluated potential and kinetic
energy differed by at most 3.78e-10 J and 4.92e-10 J. Connectivity matched;
compiled reference coordinates, nodal masses and area weights agreed to
0.787 nm, 1.22e-10 kg and 9.80e-12 m² per node. The independently implemented
constitutive parameters are matched to the declared fixture, rather than read
from the compiled material program.

[Parity evidence](evidence/contact-nodal-parity-20261107/) includes the exact
source and binary identities, commands, captured nodal histories and comparison
outputs. A concurrent CPU-only metadata check may have affected timing; neither
the capture timings nor the calculated timestep costs are performance claims.

An independent CPU backward-Euler timestep sweep reached 0.9765625 µs without
meeting either 0.25% gate: peak-normalized binned force differed by 5.064% and
integrated impulse by 1.118%. The two finest samples suggest first-order
extrapolations of approximately 48 ns for the force gate and 218 ns for
impulse, but no passing backward-Euler timestep was established. Extrapolation
is not evidence of native FP32 convergence.

## CPU-only consistent midpoint candidate

A separate CPU FP64 diagnostic applies implicit midpoint consistently to
positions, velocities, elasticity and the same fixed contact force. Its
sweep retains all outcomes, including a 31.25 µs force error of 0.308%, which
fails the unchanged 0.25% gate. At 15.625 µs it produces:

- Maximum 1 ms binned-force error 0.0855% and total impulse error 0.0110%,
  compared with the separately refined FP64 RK4 result on the same mesh.
- Minimum endpoint gap 87.10 µm against the 0.1 µm clearance floor.
- Normal contact work -3.04e-10 J and work plus the change in stored contact
  potential 2.52e-10 J. Maximum total-energy deviation was 2.47e-9 J.
- A basic rejected-trial copy check that left the CPU authoritative state
  unchanged; the diagnostic has no persistent multiplier history.

A CPU CTest repeats the fixed-mesh force, impulse, clearance and energy
checks. Linear interpolation between the two endpoints stays clear in this
plane fixture, but this is not swept collision certification for general
geometry. Midpoint has not been implemented in Matter: changing its
integration rule would require a consistent coupled contact operator, FP32
qualification, moving-contact and friction checks, and full transactional
restore and failure handling. Spatial refinement and independent physical
material calibration remain open. The midpoint result is a research lead,
not a force-qualified production control.

## Midpoint discrete consistency and runtime boundary

For this stateless fixture let q be the end position, q₀ and v₀ the accepted
state, M the lumped mass matrix, and U the elastic, gravity and contact
potential. The midpoint equations give

    r(q) = q - q₀ - h v₀ + (h²/2) M⁻¹ ∇U((q+q₀)/2)
    Φ(q) = ½ (q-q₀-hv₀)ᵀ M (q-q₀-hv₀) + h² U((q+q₀)/2)
    ∇Φ = M r;  ∇²Φ = M + (h²/4) ∇²U((q+q₀)/2).

At a deformed, active-contact state an independent finite-difference check
finds relative differences of 9.80e-10 between ∇Φ and the mass-weighted
residual, 7.01e-9 for tangent bilinear symmetry, and 4.94e-7 for the tangent
against a mixed finite difference of Φ. At 15.625 µs the maximum individual
step midpoint contact work plus potential change is 2.65e-10 J. These
checks establish consistency of the standalone equations and numerical
prototype, not the as-yet-unimplemented Metal tangent.

The current Matter path is built around a backward-Euler endpoint candidate:
the mechanical residual, trial candidate and line search use its velocity and
endpoint; nm_fem_integrate publishes x₀+h v₁; contact samples and histories
carry the associated step impulse. A half-step backward-Euler solve can
represent the midpoint equations for this restricted stateless problem, but
reflecting that candidate into the full endpoint after the existing
certificate changes the geometry and the physical impulse that was certified.
Current validation of the midpoint alone cannot certify the reflected
endpoint or all geometry on the committed path.

A safe integration must therefore add, at minimum, endpoint/path clearance
and final material-validity checks before transaction commit; correctly
convert contact impulse and history to the physical interval; recompute or
appropriately define the final convergence certificate; and preserve
checkpoint, rollback and snapshot semantics after any late failure. It must
also distinguish actual timestep and midpoint in the runtime fingerprint.
The existing contracts cannot simply be reused unchanged to provide those
properties. No partial midpoint option has been added to Matter.
