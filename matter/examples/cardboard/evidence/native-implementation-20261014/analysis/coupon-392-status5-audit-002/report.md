# Source-392 / tet-858 status-5 replay erratum and trace harness

## Result

The first CPU replay report (`source392-status5-audit-001`) reported a valid 14-update local projection. That result used the raw residual norm as its line-search merit. It is not the source-121 production-policy replay: the captured medium program has flags 213, and the source-121 accurate projection path uses frozen scaled row norms for Armijo while retaining the raw residual and scaled-correction final gates. The old report and executable evidence remain untouched and are explicitly superseded for this question.

The corrected CPU replay uses the captured candidate and compiled package, applies the scaled Armijo criterion, and recomputes the terminal residual after the last accepted update. It fails closed at the local update budget: 32 updates, terminal raw residual 4.3091416358947754, dynamic residual tolerance 4.1780484849596178e-08, scaled correction 17.340585708618164. Native point-401 reports 32 updates, terminal residual 4.3091416358947754, nonconvergence, and terminal failure 7 (iteration-budget exhaustion). The terminal residual differs by 0; high/low projected state sums differ by at most 8e-11. This is a close CPU/native replay match for the selected point, not evidence identifying the first production status-5 writer.

An independent FP64 source-law point solve converges to residual infinity norm `1.1102230246251565e-15` in 15 updates. This reference does not use the native FP32 projector and does not clear the native failure.

## Exact capture inputs and bounds

The point input is taken from the source-392 run's rejected-conditioning snapshot at control step zero, environment 0, tetrahedron 858, medium material index 1. Its candidate is explicitly not proven to be the generation that first latched status 5. Accepted material history is the all-zero 15-state rollback record. The native point record binds the 160 raw FP32 output words; the harness verifies the F high/low and Fdot words against that payload and binds source package, material asset, candidate, accepted history, and world fingerprint before dispatch.

The new CLI dispatches the existing projection trace kernel and the trace-off point batch for identical inputs. It compares direct stress, tangent-path stress, tangent action, and projected-state high/low words. It writes exact input/accepted-history bits and the entire 32-iteration trace. This is a diagnostic-only point run: no physical solver transactions, no source-law coefficients changed, and no calibration or full-solver qualification.

## Build and run

The new host source compiled under ObjC++ with `-Wall -Wextra -Wpedantic -Werror`; the source-121 Metal file compiled to AIR and linked to a metallib. No GPU dispatch was made. The exact native command is in `native-trace-command.txt`; replace only the output path with a fresh directory. The intended target is the existing `numi-matter-cardboard-bilinear15-capture-diagnostic-check`; no CMake/core change is required because the added CLI is in that target's host.

The host overlay SHA-256 is `b04bb295db1eba1c10c1001db3b2e90564099e2afb453fd630579bc5a6c76e0d`. The corresponding source-121 host SHA-256 is `f6c9a44a4d1c1142590e9818cb2de02ba61b152b613f5e6f41f8bd5f33b8aaa1`. The new code changes only the diagnostic host; the Metal shader is unchanged from source 121.

## Preserved evidence and open question

The failed source-392 coupon, point-401 diagnostic, audit-001 raw-merit replay, and corrected scaled-merit replay are all preserved. Source-level status-5 paths include constitutive/material failure and other FEM validation paths; this replay still cannot assign the native status to a producer. The trace harness is intended to expose the selected point's local projection path. A successful trace dispatch, if later run, will remain a point diagnostic and will not establish that this candidate was the first failure generation or that the coupon is qualified.
