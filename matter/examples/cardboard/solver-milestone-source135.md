# Exact-state descent repair and remaining solver qualification

The source135 native experiment repairs the captured primary Armijo rejection
at control 1, Newton 48. Its replacement direction accepts at fraction 1 and
reduces the actual assembled residual from 1.84443750e-4 to 1.60052223e-4 Ns
(13.224%). Newton 49–52 then accept. Newton 53 rejects; the entire control
rolls back to the last accepted geometry. The 27-control coupon remains
incomplete after one accepted control. This is an exact-state repair, not a
completed nonlinear-solver qualification.

The fallback is an explicit, default-off experiment inside the native runtime.
It preserves the base assembled right-hand side before FGMRES and computes
`g = Aᵀ b`, `tau = (g·g)/(Ag·Ag)`, and `p = tau g` using the native cached
unsymmetric operator. It runs only after all primary Armijo trials reject
from a healthy base. Every replacement trial uses the same accepted material
history, recomputes the production nonlinear residual, and uses the tested
fraction in the eventual candidate update. The original Newton linear
certificate is explicitly not a certificate for `p`; the final production
nonlinear certificate and transactional rollback remain required.

Native433 tests the actual GPU transpose, forward action, reduction and
direction builder against independent FP64 assembly. Relative action errors
are below 9.2e-8 and the step-length discrepancy is 1.03e-7. Fixed rows,
inactive field rows, unselected environments, invalid field inputs and
nonfinite residual controls pass their recorded checks. Native434, with the
fallback disabled, reproduces source125/native415's complete attempted
numerical history: 64 Armijo records, 4,096 true-residual records and every
physical output. These tests have separate scopes from native435's failed
full-coupon experiment.

The captured 2,538-by-2,538 matrix is full rank at the recorded FP64 cutoff,
but its 2-norm condition number is 4.5897467e7. The weakest right singular mode
contains 99.999846% of the captured correction's squared norm. Its exact
linear correction is approximately 188 m/s, whereas the raw residual-descent
candidate is approximately 0.000143 m/s. This explains a numerical difficulty;
it does not establish that conditioning alone causes every nonlinear failure.
No mass, stiffness, physical parameter or acceptance threshold was relaxed.

The independent endpoint replay now recovers the previously missed medium
tet1691 plus root by continuation with the same equations and accepted
history. All 1,728 paper endpoint projections pass the unchanged strict
reference gates. The complete native and independent finite secants agree
within 0.285%, while the independent secant differs from the center action by
109.49%. The earlier incomplete endpoint comparison and direct-reference
failure remain retained. No observed multiplier-activity or yield-gap-sign
flip proves that every relevant branch is unchanged. This finite-step result
does not qualify a derivative or justify a universal FP32 noise-floor claim.

Diagnostic native436 replays smaller regularized directions. All tested
positive fractions have valid recorded material statuses; fraction 1 gives
the lowest measured residual among 1 through 1/32 for each proposal. The best
recorded regularized proposal reduces the actual residual by about 77%, but
the less regularized proposals depart substantially from their linear
predictions. These are residual replays with zero accepted physical
transactions. The CPU CGLS direction exports are proposals awaiting native
testing, not a new production solver.

Projection reuse is also experimental. Native431/432 reproduce 311,040
cached matrix words and the recorded and random action words exactly in each
fixture. Source136's requested full coupon/native437 rejects initialization:
the new admission check excludes the existing elastic starch elements and an
unconditionally enabled IPC capability flag. That failure is preserved; the
isolated cache checks do not establish a speedup or full-history parity.
Source137 builds 69 selected targets and passes 74 CPU tests and five native
operator regressions. The direct tools exercise original and cached modes 0
and 1; mode 2 is source reviewed rather than directly selected by this suite.
Its repeated fallback selector and the cache optimization still need native trajectory
qualification. Source137 also retains two stale trace-cap labels: the selected
all-Newton observer ceiling is 512 although two notice strings say 64. The
review records this metadata error; the observer ceiling does not select or
accept a physical step.

| Verdict axis | Current boundary |
|---|---|
| Solver correctness | Exact-state descent repair and selected material/operator checks pass; complete nonlinear/contact and native derivative gates remain open. |
| Numerical convergence | No completed mesh, timestep or slower-loading series. |
| Component calibration | Published packages and component checks exist; calibrated full corrugated damage and independent debonding remain open. |
| Analogue validation | No completed held-out whole-fixture validation. |
| Physical assembly in simulation | No scored, folded, bonded, released and settled shipping checkpoint. |
| Crush/impact prediction | No qualified first peak, capacity-derived scenario masses, gravity crush or drop results. |

[Reference022](evidence/native-implementation-20261014/publication-binding-reference022.json)
retains commands, exact reconstructable source snapshots, full attempted
histories, independent comparisons and the failed diagnostics. The
[six-axis ledger](evidence/native-implementation-20261014/summary.json) keeps
these numerical results separate from component calibration, physical
assembly and prediction. The kraft/NSSC/starch recipe, restart ceiling of 16,
shipping dimensions and all later physical qualification requirements remain
unchanged.
