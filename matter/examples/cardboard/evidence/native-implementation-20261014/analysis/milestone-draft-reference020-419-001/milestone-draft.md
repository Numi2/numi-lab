# Cardboard simulation evidence milestone — draft, source020 plus run419

**Scope.** This is a source-bound status draft assembled from reference020 and the independent CPU adjudication of native run419. It is an evidence summary, not a claim that the modeled carton is validated. The base publication record binds 56 selected cases and 850 files. A separate read-only audit independently verified all 813 direct copies and 37 gzip copies; all copied byte counts and SHA-256 values match the index, with no duplicate targets or current external-source drift.

## Six verdict axes

| Axis | Current verdict | Evidence and exact boundary |
|---|---|---|
| Solver correctness | **Component checks qualified; complete contact-cycle solver behavior remains open.** | Run419 center projection passes all 1,728 material rows. Native422 fresh operator action matches the captured action exactly; independent FP64 center action differs by 2.8227e-5 relative. However, native424 has no tested epsilon whose finite-difference residual/action agrees with the captured A direction; the best admitted relative error is 1.0948 at 2^-14. This is not a full solver pass. |
| Numerical convergence | **Unqualified.** | No complete independent spatial, timestep, or slower-loading convergence series. The reference020 selector-6 topology check is compile/preflight only and executes zero physics. Native419 is a failed partial coupon run; it does not establish convergence. |
| Component calibration | **Source data and executable constitutive slices exist; native forward calibration remains open.** | The run419 center local tangent checks and source-equation checks are component evidence only. The Robertsson absolute-energy convention remains open. Kraft/NSSC, starch, and hot-melt parameters are not calibrated by transferring another grade's coefficients. |
| Analogue validation | **Open.** | Published source curves and separate analogue series are retained, but no completed held-out whole-fixture validation qualifies the present material recipe. |
| Physical assembly in simulation | **Unexecuted.** | There is no complete accepted scored, folded, bonded, released, and settled shipping-box checkpoint. The explicit-flute shipping path still has admission/runtime blockers. |
| Crush and impact prediction | **Unexecuted.** | There is no qualified box-compression curve, frozen peak force, capacity-derived mass, gravity-crush result, or drop-impact result. |

## Run419 adjudication

The selected source125 Newton48 center projects all 1,728 tetrahedron rows, with maximum independent scaled residual 1.994e-10 and maximum dynamic correction 1.921e-8. Native fresh A·dv is exactly equal to the captured native action. The independently assembled FP64 A·dv differs from the native action by 5.2065e-9 Ns (2.8227e-5 relative). The mass action agrees within 1.59e-12 Ns; the remaining center-action difference is in the stiffness contribution. Native422's cached-versus-fresh action difference is 2.7817e-8 Ns, which is measurable but does not make the fresh/captured operator comparison fail the 1e-3 gate.

Independent FP64 endpoints at epsilon 2^-22 pass all center and endpoint projections. The local liner and medium tangent discrepancies are 3.66e-8 and 1.07e-7, while the assembled secant discrepancy remains 2.2462e-5. Smaller FP64 steps worsen the assembled secant comparison, showing subtraction sensitivity in this small total action; this is not a universal noise-floor estimate. The larger epsilon 1/64 trajectory has substantial state changes and lambda-activity crossings, and one liner endpoint projection fails, so its complete independent assembled secant is unavailable.

Native424 reproduces every retained run419 endpoint node and residual bit-for-bit, but no native epsilon resolves the A-direction derivative. At epsilon 1/64 the relative error is 17,199.84; the best admitted native result is 1.0948 at epsilon 2^-14. At epsilon 1/32 both side statuses are 5 and the secant is inadmissible. Those side statuses are propagated environment/stage status, not per-tet local failure flags.

A separate first-status5 record captures one tet (1464) in a private Armijo internal-force evaluation at Newton26. Its determinant is 0.995947; its reported maximum stress is 1.17918528 GPa against a 1.0 GPa validity ceiling. This single private trial is not the status of every tet in the later FD side arrays and does not identify the terminal coupon failure by itself.

## Remaining sequence

1. Preserve all failed and partial runs. Continue source-bound diagnosis of the exact Newton/contact/material failure; do not promote the 419 center slice into a full solver verdict.
2. Close the local and assembled derivative gates with exact captured high/low inputs, accepted history, complete material branch evidence, and independently replayed native residual endpoints. Keep unavailable/failed endpoints null and fail closed.
3. Complete the full coupon protocol and all paired controls through loading, withdrawal, release, and settling, with accepted-only observables and transactional rollback evidence. A prefix or one successful pair is not completion.
4. Run independent refinement in x, y, and z, then combined spatial refinement; run timestep refinement using observed accepted native dt/count and a separate slower-load case. Compare peak force and work within 2%, the full curve within 5%, signed springback within 1 degree, and damage/debonded fractions within 2 percentage points. These gates apply only after complete, compatible histories and certificates are present.
5. Complete grade-specific component calibration and held-out analogue validation. Resolve the own-grade Robertsson energy convention from source evidence; keep kraft/NSSC and adhesive parameters attached to their own source or explicitly labeled analogue sets.
6. Complete the explicit-flute box assembly transaction, including score, fold, glue/contact, release, and settle checkpoints; validate accepted reaction, moment, springback, damage, and interface observables.
7. Only after those gates pass, freeze a qualified compression peak and capacity-derived mass, then run gravity-crush and drop-impact predictions as separate validation cases.

**Explicit non-claims:** no complete coupon result; no spatial, timestep, or loading-rate convergence; no calibrated full material recipe; no complete physical assembly in simulation; no capacity determination; no crush or impact prediction. The archive-integrity audit verifies preservation of evidence bytes only, not scientific correctness.

## Bound records

- [Reference020 archive binding](/Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/native-implementation-20261014/publication-binding-reference020.json) — 56 cases, 850 indexed files, summary SHA-256 `588335cbdd7062ab2c33d7b80b7e086168ab86548ab56885f788f65060115abd`.
- [Independent reference020 copy audit](/Users/home/cardboard-implementation-20261014/analysis/milestone-draft-reference020-419-001/index-audit.json) — 813 direct and 37 gzip payloads verified; no mismatches.
- [Run419 independent operator/FD adjudication](/Users/home/cardboard-implementation-20261014/analysis/operator-fd-e2-independent-adjudication-419-001/report.json) — report SHA-256 `9d840db47f6e898779998b0f0f7ab51355c0f323798e85dc64823bf2d88a4193`.
- [Run419 exact-input small-step report](/Users/home/cardboard-implementation-20261014/analysis/operator-fd-e2-adapter-001/independent-smallstep-419-003/report.json) and [native424 endpoint audit](/Users/home/cardboard-implementation-20261014/analysis/native424-residual-sweep-audit-001/report.json).
- [Source020 milestone summary](/Users/home/numi-cardboard-20261006/matter/examples/cardboard/evidence/native-implementation-20261014/summary.json) remains unchanged by this draft.
