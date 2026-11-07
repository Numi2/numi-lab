# Contact-onset diagnosis, 7 November 2026

The legacy CCD acceptance containment is independent of the contact work.
Both contact/CCD experimental defaults remain disabled. This report closes
specific stopping/precision defects and retains a completed, residual-qualified
IPC trajectory. **A force-qualified control has not been obtained.** No AL
promotion, speed comparison, solver-budget increase or contact-law tuning follows.

The [previous experiment](FEM_PLANE_AL_EXPERIMENT_20261107.md), including failed
calibration, loose-tolerance replay and the tightened series, is unchanged.
Its momentum balance on accepted pre-contact AL states is not evidence of
accurate nonzero-multiplier response. Its earlier nonzero-multiplier replay is
transaction evidence from a physically unqualified series.

## First rejected IPC solve

The original 27-node, 8.8 g pad at h=0.5 ms fails on step9. The residual decreases
approximately 9.1686e-5 -> 5.0039e-6 -> 2.5979e-7 -> 6.53235e-9 N s, then the
nonlinear tail latches with a zero correction through iteration15. The inner
true residual is similarly small. The outer latch used the matrix diagonal
floor (1e-8, inertia units) as a residual floor, while the final physical
certificate required 1e-9 N s. These stopping decisions disagreed.

The repair removes that dimensional floor from the nonlinear stopping tests.
Both stopping and publication use the same optional explicit mechanical impulse
criterion. Existing all-zero policy fields retain legacy normalization; matrix
regularization remains in the operator. The new policy is admitted only for
pure FEM mechanical rows and contributes to the compiled physics fingerprint.
No Newton/FGMRES budget was increased:16 Newton, restart16/total64 FGMRES.

For the fixture, define S=||m_i*g*h||_2 and m_min=min(m_i). The authored policy is:

- Absolute tolerance A=0.00025*m_min*g*h N s.
- Relative tolerance R=0.00225*m_min/||m_i||_2.
- Native acceptance: ||residual||_2 <= A+R*S = 0.0025*m_min*g*h.
- Independent local gate: max_i ||residual_i||/(m_i*g*h+1e-10 N s) <=0.005.
- Independent whole-body impulse balance <=1e-7 N s, positive clearance
  greater than0.1 micrometre, exact restoration and deterministic continuation.

The local and global checks are separate. The first physical-policy attempt
completed several runs but failed the local gate; that failed calibration is
retained. The revised scale was declared before its runs and reserves local
roundoff headroom; the physical gate itself was not loosened.

At tighter scales, world-position arithmetic stalled around contact onset
despite full line-search steps and small independently recomputed linear
residuals. The existing reference-displacement FEM mode alone did not fix it.
A narrowly scoped precision repair now evaluates reference-mode FEM against an
immutable plane using the compensated position expansion and the same IPC
radial force/curvature/potential. It retains the gap's low limb in an unused
sample lane. It changes arithmetic, not activation distance, coefficient law,
clearance or material parameters; AL is unchanged. Runtime defaults still use
their existing coordinate representation. This qualification is limited to
the explicit reference-mode fixture and does not establish other line-search
or coupled-domain precision.

## Retained equations and operator checks

Read-only GPU snapshots record accepted/working nodes, compiled tetrahedra,
fully assembled RHS before Krylov scratch reuse, proposed correction, line-search
and rigid-clearance restrictions, contact samples, accepted/trial dual histories,
and the state before/after multiplier updates. Full coupled captures retain the
same RHS and the native production tangent-vector action. Two diagnostic
capture defects were corrected: active contact indices were labelled as pair
records, and empty optional typed arenas were treated as truncated records.

The offline NumPy FP64 evaluator independently computes log-J neo-Hookean
stress, contact response and analytic tangent action from these retained
discrete states. It performs no integration and never feeds runtime acceptance.

| Retained nonzero correction | RHS difference, L2 N s | Relative tangent-action difference |
| --- | --- | --- |
| IPC, h=62.5 us, step72, iteration2 | 1.09117e-11 | 3.00955e-8 |
| AL, h=0.5 ms, step12, iteration15 | 8.36871e-11 | 4.62827e-8 |

The zero-correction IPC capture is also retained but is not used as evidence
for a nontrivial tangent test. The evaluator freezes captured IPC stiffness
when differentiating, matching the production linearization. These are local
equation checks, not physical force calibration.

In the frozen AL schedule, a corrected primal iteration is followed by one
inexact multiplier update. In the first rejected strict run, iteration15's
FP64 residual is 2.41577e-10 N s before updating the trial multiplier and
3.07905e-8 N s afterwards. The inner linear residual is approximately
1.62429e-10 N s and alpha remains1; the rigid-clearance limiter is inactive.
The final certificate rebuild agrees with the residual after the update.
Accepted dual history remains exactly zero throughout the rejected step;
trial history reaches2.70084e-5 N s.

Thus this failure exposes inadequate convergence of the current primal-dual
progression within its frozen budget. The observed operator agrees closely
with the independent evaluator. This does not reject the broader barrier-free
method or justify changing its schedule before the control is force-qualified.

## Completed duration, failed force reference

Noncontact free fall now completes with the discrete update, independent local
accounting, zero contact impulse and exact replay/rollback checked. An instrument
crash when the free-fall scene had no rigid proxy was fixed; its failed attempt
is retained.

For contact, the unchanged 20 mm pad and0.2 mm initial gap complete10 ms on a
fixed n=2 mesh at all six reference timesteps below. Every run passes local
residual, global balance, clearance, fingerprint rejection, exact replay and
forced rollback. Normal impulse and both J dot dx/h and J dot midpoint-v work
are recorded for every accepted step; rejected work is not counted as accepted.

| h, us | Accepted steps | Maximum independent local scaled residual |
| --- | --- | --- |
| 62.5 | 160 | 0.00064334 |
| 31.25 | 320 | 0.00082693 |
| 15.625 | 640 | 0.00079255 |
| 7.8125 | 1280 | 0.00040841 |
| 3.90625 | 2560 | 0.00044673 |
| 1.953125 | 5120 | 0.00031802 |

The final halving changes1 ms binned force by **10.1048%**, against the declared
0.25% reference gate, although total impulse changes only0.0423183%.
The sequence is nonmonotonic. The force-qualified control and its1% accuracy
gate therefore remain unpassed. Lower residuals, completed steps and excellent
overall balance do not establish accurate contact forces.

A separate n=4 probe at fixed h=15.625 us completes and passes residual/
transaction checks, but its roughly50% first-bin force change is only mesh
sensitivity. Temporal error is not qualified there, so it is not a spatial
reference. Timestep refinement was performed at fixed mesh; no combined
time/mesh refinement is presented as a reference.

The unchanged IPC coefficient includes an inertial term proportional to
1/(inverseResponse*h^2*activation_width^2), plus its impact term. Its numerical
regularization therefore changes with timestep and nodal response. This is a
concrete confound for interpreting refinement as fixed regularization; it is
not proof of the cause of the observed force discrepancy. The AL penalty also
remains fixed per node and is not a mesh-invariant physical stiffness.
Neither law or penalty was retuned to make this gate pass.

Near-contact initial states were retained in the repaired series:

| Initial gap | IPC | Frozen AL |
| --- | --- | --- |
| 50 micrometres | 20/20 accepted | 5 accepted, then status10 |
| 0.5 micrometre | 20/20 accepted | first step rejected, status10 |

Both IPC cases pass local accounting, clearance and replay/rollback. Both AL
failures restore, and no accepted nonzero multiplier exists in these runs.
IPC contact work is positive, approximately1.79555e-5 and7.99696e-5 J respectively.
These initial states contain stored barrier potential. A complete energy
assessment must include that potential and the integration rule; work alone
is not an energy-creation verdict. The historical JSON field named
maximum_accepted_multiplier_impulse_Ns also reports IPC's contact-history
impulse for IPC runs; it is a persistent AL multiplier only in the AL arm.

## Reproduction and decision

Build the actual Matter owners with the existing focused tests/contact CMake
entry point on Apple Silicon. The two focused CTests pass: physical-policy
admission/fingerprinting/disabled defaults, and completed-duration residual/
noncontact/geometry/restoration checks. These test names and their stdout
explicitly make no force-refinement claim. The separate refinement tool
returns failure for this retained dataset.

[Temporal reference gate](evidence/contact-onset-diagnosis-20261107/temporal-refinement.json),
[IPC operator](evidence/contact-onset-diagnosis-20261107/ipc-operator.json),
[AL operator](evidence/contact-onset-diagnosis-20261107/al-operator.json),
[AL iteration trace analysis](evidence/contact-onset-diagnosis-20261107/al-iterations.json).

The evidence archive retains plans and hashes, source patches, native binary/
shader hashes, all successful/failed physical and precision attempts, full
captures, traces, near-contact tests, build failures and CTest logs. Plans are
diagnostic declarations and revisions, not a registered paired performance
study. Some diagnostic GPU work overlapped; recorded timings are observations
and cannot support performance promotion.

The unresolved deliverable is a completed **force-qualified** IPC control with
separately qualified temporal/spatial references. AL iteration changes and
either candidate's performance tuning remain stopped. No experiment is promoted.
