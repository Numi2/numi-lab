# Material tangent reuse and inexact Newton experiments

All builds and native checks ran on SSH Numi (Apple M6). These are numerical and short mechanics results, not supported-rest, physiological or clinical qualification.

## Shared local factorization

Implemented an opt-in, compiler-proved 15-state compensated tangent path. The ordinary cache projects once, evaluates A and B once, performs one pivoted compensated LU factorization, solves nine right-hand sides without forming an inverse, and publishes the existing cache only after success. Direction-dependent or ineligible programs retain the legacy path. Factors are not reused across changed linearizations.

checks9.log: compiler proof rejects DF in A/B and permits fixed base RATE; GPU nine-RHS tests give exact legacy solution parity and maximum local residual 1.07843334263e-9, including later pivots and near-singular rejection. Full native material test passes affine/random-direction tangent actions, elastic and plastic loading/unloading, accepted history comparisons, replay and late-rejection rollback. Both old compensated and accurate/shared fixture modes remain covered. Two failed test-fixture assumptions were corrected; checks7/8 and their exact source patches are retained.

checks10.log: all nine selected residual, multiphysics, rollback, contact, geometry/Armijo and vascular regressions pass.

The uploaded Human world reports **0 eligible materials out of 20**. Checked-in Human sources are stateless; eligible 15-state sources are paperboard. Therefore this is not a Human performance gain. No eligible-material speedup measurement is claimed. NM_FEM_SHARED_LOCAL_FACTORIZATION remains off by default.

## Rejected short-run performance experiments

Canonical stateless tangent: 43.650811834 -> 36.7266905 body seconds in one matched four-root pair, but max_J failed the unchanged 1% comparison gate. Runtime eligibility was 2/20 materials. Default off; no accepted throughput gain.

Residual-adaptive Newton forcing: matched work-counter pair 43.496946208 -> 36.183565542 body seconds, total used Krylov columns 8297 -> 3923, but anchor reaction failed the 1% comparison gate. Default off; no accepted throughput gain. Failed evidence and diagnostics are retained rather than relaxing gates.

These runs are unbalanced release transients at 0.125 ms, not supported resting throughput. No 0.1x integrated qualification, long-run baseline or respiratory intervention is claimed.

## Remaining work

Human needs sharing/specialization of its actual stateless material programs or other measured work reduction, plus stronger admissible preconditioning, a supported checkpoint and qualified larger timesteps. Full packed unknown reduction, model reduction, cross-step recycling and long coupled qualification remain incomplete. A separate inverse-transpose helper audit is recorded in INVERSE-AUDIT.md; affected nonsymmetric callers are not silently changed by this optimization.

Exact source, artifacts, invocations, compact logs, comparisons and failed trial patches accompany this report. Generated output directories remain local and are excluded from publication.

Human one-root shared-switch smoke: exit 0, FEM and fascia terminal audits passed, body/physiology clocks matched. Only 0.125 ms simulated; timing is not sustained throughput. No visual verification performed.
