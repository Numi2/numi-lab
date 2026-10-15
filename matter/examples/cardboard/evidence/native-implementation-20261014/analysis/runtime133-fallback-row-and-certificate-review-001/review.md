# Runtime133 Cauchy fallback row and certificate review

Status: read-only source review. No source was edited and no GPU/native run was performed for this package.

## Bound source

The reviewed baseline is the frozen Source133 snapshot at `snapshots/source-base-state-cauchy-armijo-133`. Its `frozen-source.json` SHA-256 is `48d0c2da92cc8557507faa67bf80b3730c669075e25fd0d8ce202ff08ba2d973`.

Source135 was checked at `snapshots/source-cauchy-positive-mass-admission-135`. Its `frozen-source.json` SHA-256 is `40f0196428da88be0ecb951a03a9989c51e44405abd8393f13970e1f7239b9a8`. The two frozen manifests have the same 785 source paths; only `matter/src/runtime.mm` differs. The Source135 runtime diff is the fallback-only immutable-mass admission branch at lines 11545-11559.

## Correction and certificate

The ordinary correction metric is measured before Armijo. Runtime dispatches `nm_fgmres_measure_correction` at `runtime.mm:21984-21998`; the kernel computes a max relative ratio from the original FGMRES `solution` and writes it to `fgmresStates[environment].nonlinear.z` at `fgmres.metalinc:1434-1512`.

When the selected primary Armijo sequence exhausts its trials, the kernel marks private metrics.w=3, leaves the production status healthy, and sets the line-search alpha to zero. The optional second pass then builds `p = tau A^T b` into `femSolution`, clears the FGMRES completion latch, and does not rewrite `nonlinear.z` (`fem.metalinc:4556-4572`; `fgmres.metalinc:5265-5371`). Runtime encodes the second pass only for the selected fallback Newton state (`runtime.mm:22217-22230`).

After the final line-search alpha has been synchronized, `nm_fem_apply_solution` adds `alpha * solution.xyz` to the candidate and scales `solution.xyz` in place to `alpha * p` (`fem.metalinc:3327-3359`; runtime binding `runtime.mm:23622-23635`). The final `nm_mixed_certify` receives that mutated `femSolution` as its correction input and computes `certificate.nonlinear.y` from it (`runtime.mm:24179-24201`; `mixed_fem.metalinc:597-700`). Therefore the terminal certificate's correction telemetry represents the applied, alpha-scaled fallback update, not the stale original delta.

The gate is residual-authoritative: `nm_mixed_certify` checks the assembled residual/volume/pressure/transport bounds and only requires the reported correction to be finite; it does not impose a correction-magnitude threshold (`mixed_fem.metalinc:850-885`). The FGMRES nonlinear latch does carry the pre-Armijo `nonlinear.z` forward, but the later check uses it only for finiteness (`fgmres.metalinc:1330-1368`). So the final certificate does not substitute the old delta norm, although intermediate FGMRES telemetry/latch state is not rewritten to the Cauchy correction.

## Skipped-environment trial-status scratch

In the optional second global Armijo pass, environments whose first-pass metrics are already selected (metrics.w > 0.5) receive a private `trialStatuses` entry of `NM_STATUS_NONLINEAR_SOLVER_FAILURE` in `nm_fem_armijo_begin_trial` (`fem.metalinc:4248-4283`). The candidate-preparation kernel then copies the current candidate unchanged and returns on that unhealthy scratch status (`fem.metalinc:4344-4351`); measurement returns for metrics.w > 0.5 (`fem.metalinc:4435-4445`); and the decision kernel returns before reading the trial status or changing production status (`fem.metalinc:4502-4514`). This is an expected private skipped-trial marker. It can appear in trial diagnostics, but it is not the production `statuses` buffer used by the final certificate or commit.

## Source135 mass guard

The baseline fallback scope at `runtime.mm:10049-10086` had no positive-finite authored-mass check; the separate operator-FD scope already had one at `runtime.mm:10143-10152`. Source135 adds the same size/finite/positive condition only in the fallback selector branch, immediately before the shared Armijo scope (`runtime.mm:11545-11560`). It rejects unless the authored mass vector length equals FEM node count and every physical mass `mass[0]` is finite and strictly positive.

This is tied to the real row definitions: production gather removes only static constraint tags and otherwise includes the physical mass term (`fgmres.metalinc:5064-5124`); the Cauchy direction and its norm use free positive-mass rows (`fgmres.metalinc:5252-5262, 5309-5323, 5344-5352`); and the mechanical residual treats nonpositive mass as fixed (`fem.metalinc:2799-2821`). The mass values are captured from the authored world at runtime creation (`runtime.mm:3313-3328`) and validated as immutable on restore for non-mutable topology (`runtime.mm:29165-29190`). The Source135 guard is therefore a narrow admission check for the fallback's row-space consistency; it does not change normal Armijo admission.

## Limits

This package verifies source structure and frozen-manifest identity only. It does not claim that the Cauchy fallback converges, that its direction is a certified linear solution, or that a physical simulation qualifies.
