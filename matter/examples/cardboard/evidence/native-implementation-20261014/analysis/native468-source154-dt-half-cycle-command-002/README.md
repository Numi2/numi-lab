# Native 468 dt/2 complete-cycle command preparation

Prepared only; no native process, GPU launch, or build was run.

## Decision

The source154 clock permits an exact schedule-preserving dt/2 command: 467 uses 12/3/6/3/3/3 steps at an effective runtime dt of 9.9999997473787516e-05 s. 468 uses 24/6/12/6/6/6 at 4.9999998736893758e-05 s. Both total 0.0029999999242136255 s. For all 30 original sample timestamps, phase, target angle, right-grip mode, and all 1620 right-grip target float components match exactly. The intermediate fine endpoints lie on the same piecewise-linear angle(t) curve.

The prescribed angular rate is 312.500008 deg/s in both loading and unloading. The per-step chord-speed difference is 9.29618582e-09, a sampling effect on the same curved rotational path.

The command sets the certificate scalar from 1e-6 to 5e-7. Under the source runtime's pure-mechanical residual normalization of max(reference, 1), while the residual scale stays 1, this preserves a conditional 0.01 N force bound at both dt values. The new command must verify that scale row-by-row and inspect the CSV force-imbalance columns after execution; this prepared package makes no force-accuracy claim.

The nonlinear ceiling changes from 128 to 64, as requested for capacity/runtime bounding. This is therefore a dt/2 diagnostic with a different nonlinear budget, not a pure time-refinement qualification. Other physical and solver settings from 467 are retained.

## Native 467 predecessor status

467 was interrupted by its capacity guard after only control step 0 completed for both environments. It was not a nonlinear solver failure or completed cycle. The case receipt, partial observations/mesh/manifest, launch guard, and capacity-audit report are hash-bound as external command inputs. 466 full-attempt default64 parity, source154 build-topology erratum, the 787 source-file hashes, and binary/build-metadata hashes are also bound.

## Limits

- No native 468 run has occurred.
- The C++ observation field `right_grip_constrained` is wrong during post-release relaxation: the target writer frees the right grip in both release phases, but the CSV column marks post-release as constrained. Use the phase/target source binding, not that CSV field, for post-release classification.
- Exact common-time prescribed target equality does not prove equal discrete FEM trajectories or reactions; dt/2 changes the integration discretization.
- This is not physical/calibration qualification.

## Command

See `native-468-command.json`. Inputs are copied under unique basenames in `inputs/`; every copy is hash-bound in `external-input-bindings.json`.

Question: Under source154, does dt/2 complete the same physical-duration coupon path with the same force-accuracy bound using an explicitly lower 64-Newton budget?

Prediction: Run a dt/2 diagnostic under source154. Double each of the six phase counts and halve dt so the loading, hold, unloading, relaxation, release and post-release durations remain 0.003 seconds total and the source prescribed angular path/rate is identical at common physical times. Keep geometry, mesh, materials, material iterations, line-search, FGMRES and all-control CGLS64/block3/primary48 settings from source154 command 467. Tighten the solver certificate from 1e-6 to 5e-7: in this pure mechanical case the residual norm is an impulse in N s with implicit unit normalization while its scale remains below 1; at dt=5e-5 this conditionally preserves the 0.01 N bound. Use a 64 rather than 128 Newton ceiling to bound runtime; therefore this is a dt/2 diagnostic with a different nonlinear iteration budget, not a pure time-refinement qualification. Preserve full histories and verify post-run residual scaling and force imbalance; no physical/calibration qualification is implied.
