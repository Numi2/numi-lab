# Source149: bounded native right scaling

The existing native CGLS direction experiment now supports opt-in
`NM_FEM_BASE_CGLS_RIGHT_SCALE=block3`. It constructs the current nodal principal
blocks, uses their inverse as a right map S, applies the distinct transpose Sᵀ,
and converts the final direction back before evaluating the original unscaled
operator and unchanged installer. Scratch is private. Singular or invalid
selected blocks fail closed. The physical operator, material recipe, Armijo
residual, final certificate and rollback remain unchanged.

This is bounded conditioning software, not a complete solver qualification.
It retains the existing contact-free cached pure-FEM limits E≤2, N≤4096,
T≤32768, CGLS32/64 and FGMRES restart16. It does not admit the shipping mesh or
contact by changing an environment variable.

## Measured evidence

- Native460 runs the actual scale, transpose and CGLS Metal kernels against
  manufactured nonsymmetric matrices:678 identity-path and886 scaled-path
  checks pass. These helpers perform zero physical transactions. The literal
  preparation label `gpu_checker_not_run_here=1` is stale text embedded in the
  helper; the retained execution receipt and host command results adjudicate it.
- Native463 captures the same physical Newton48 state as455 and419. All17
  required state-array joins are exact. The native inverse blocks differ from
  independently assembled A_nn⁻¹ by at most `1.07e-6` relative. Native q, r and s
  action differences are recorded separately; the same-run rich operator
  observer skips at two environments, so the419 operator is used only after the
  exact physical-state joins. Device/program identities differ and are retained.
- Native463 accepts all16 remaining private full updates, ending at
  `9.4420302e-6 N s`, compared with `1.49501593e-5 N s` for the unscaled baseline.
  It still fails control1 and rolls back. The64-step CGLS capture is exhausted,
  not converged. One physical control is accepted. Private Newton updates do not
  count as physical steps.
- Native462 omits right scaling. Its entire attempted history matches459:
  35 output files,112 Armijo records,4096 true-residual records,64 CGLS records,
  and exact CGLS/state/first-status binary payloads. Only declared paths,
  independently bound device/program identities and new null disabled-feature
  metadata differ. Both runs fail; this is not prefix-based convergence evidence.
- Native461 is retained as an input-seal rejection, before GPU execution: its
  executable path was not the executable named by its build. The corrected
  command is a fresh case463. No solver failure is inferred from461.

The source-bound predictions are in `solver/cgls-block-right-primary48-463` and
`solver/cgls-block-right-disabled-parity-462`. The strict lineage audit and full
disabled-history comparator accompany the publication. A smaller linear residual
alone is insufficient: earlier native456/457 full-step mechanical residual
replays were worse after scaling, and those counterexamples remain retained.

## Remaining qualification

No complete coupon, mesh/time study or slower-loading study has passed. The
regional-conditioning observer and all-control/budget extensions remain separate
experiments until their native evidence is available. The restart-generation
reader ambiguity must be resolved or explicitly rejected before augmented-map
replay. FP32 finite differences remain inconclusive without resolved absolute
noise floors.

Constituent damage calibration, identifiable starch debonding, deeper scoring,
the12-score shipping blank, fixture-free bonded assembly, compression capacity,
gravity crushing and drop impact remain open under their separate verdict axes.
No first compression peak or heavy-object mass has been qualified. Exact-recipe
calibration and impact validation remain limited by the published measurements.
