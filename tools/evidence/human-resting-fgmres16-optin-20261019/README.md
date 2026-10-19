# FGMRES 16 opt-in: source and bounded evidence

The source patch is based on Numi Human HEAD `54b853880d50d29f4dd3bc1719e80203331fad06`. It adds one opt-in app setting, `NUMI_HUMAN_RESTING_FGMRES_BUDGET=16`, for the existing runtime FGMRES iteration-budget setter. The accepted values are unset, `0`, and `16`; unset and `0` do not call the setter, so the cooked runtime budget remains the default. The patch does not change Newton iterations, tolerances, restart width, rate exponent, timestep, or runtime sources.

## Accepted one-step trial

Profile 249 compiled and linked successfully, then completed one joint 0.25 ms step with aortic wall, skin-fat FEM, and prostate FEM enabled; MPM remained disabled. The runtime reported cooked FGMRES budget 64, effective budget 16, restart width 16. The run exited 0. Its coupled CSV is byte-identical to profile 248's default-budget one-step CSV and prior profile-006 joint CSV (2,556 bytes; SHA-256 `7e5d177ae7cbfe5c7c470bcfa8814e394f469678d1cd62289bedbb6a94b0a800`). This is a single accepted-step functional comparison. It does not establish longer-horizon stability, performance qualification, or visual correctness.

The exact tested profile-249 app source is preserved in `evidence/profile-249/tested-app-source.mm`. It contains the same budget guard and also an environment-gated iteration diagnostic added for profiling; that diagnostic env was unset for the accepted run and is excluded from the canonical source patch. The build receipt pins the tested app object, executable, and reused runtime library.

## Iteration-count limit

Profile 248's accepted default-budget diagnostic saw 256 completed microticks, Newton budget 12, configured FGMRES budget 64, restart width 16, and a maximum 16 FGMRES iterations in the last microtick. The runtime counter scope is only the maximum used in the last microtick; the 3,072 budget records are 256 microticks times 12 Newton-entry budgets, not 3,072 actual iterations. Timing sampling was incomplete: 32 samples totaling 694.701 ms and 3,040 counter-buffer allocation failures. It does not explain the full solve time. See `evidence/profile-248/diagnostic-evidence.json` and its notes.

## Failed Newton-budget experiment

Profile 250 combined FGMRES 16 with Newton budget 2. It exited 1 before the first accepted microstep (`matter_status=10`, `completed_microsteps=0`, failing index `4294967295`). No accepted-state parity was measured. The Newton override is excluded from the proposed source patch; the failure is retained as a limit, not papered over or retried.

## Skin contact label correction

Historical logs contain `contact_surface=skin_outer_only`; this label is stale. The source classifies exposed outer skin, exposed deep-fat boundary, and paired skin-fat interface vertices only on open patch sidewalls as external-contact eligible. The shared internal interface is ineligible. The source predicate, not that historical log label, describes the contact set.

Visual verification was disabled in all referenced runs. These results are software/runtime evidence only, not biological or clinical qualification.
