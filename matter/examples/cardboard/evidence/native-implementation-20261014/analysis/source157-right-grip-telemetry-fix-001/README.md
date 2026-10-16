# Source157 right-grip telemetry fix

This is a one-file overlay against immutable source154. It changes only the `right_grip_constrained` value written to `observations.csv`; it does not alter target generation, timestep schedule, runtime inputs, solver calls, or acceptance logic.

`fillTargets` writes target mode `w=1` for constrained grip nodes and `w=2` for free right-grip nodes. Its source154 branch frees the right grip in both `release` and `post_release_relaxation`. The old CSV expression used only `phase == "release"`, so every post-release row was mislabeled as constrained. In a full schedule with three post-release steps, the old reporter incorrectly wrote `1` at steps 10, 11, and 12 while the generated target mode was `w=2`.

The patch reads the generated target vector for the current environment and each authored right-grip node. It emits `1` only when all those exact command modes are `w=1`, and `0` when they are `w=2`. Missing nodes, invalid modes, truncated ranges, or mixed modes fail explicitly because one CSV boolean cannot faithfully encode them. The existing `fillTargets` function is byte-for-byte unchanged (function-body SHA-256 `cfec49a4be92145586acbf0a3d99c59585475790148859165835e36bafafdbb4`).

The CPU source-contract/schedule check passed four schedules: complete with and without post-release, zero release, and a terminal one-step release. It checks the driver's phase-branch text and target-mode assignments, verifies unchanged target-generation code, and compares every sampled boolean with the commanded node modes. The old phase-name counterexample is retained in the report. The check is a CPU schedule/source-contract test, not a native solver run.

Objective-C++ syntax-only compilation passed using the source154 probe compile command and headers, with only the existing unused `writeShippingNominalCurveArtifact` warning. The first syntax invocation without the original `matter/tools` include directory failed to find a local header; the recorded successful invocation adds that source154 include directory. No library build or GPU/native simulation was run.

## Files

- `source157.patch`: patch that applies cleanly to source154.
- `source/matter/tools/cardboard_probe.mm`: patched source copy.
- `tests/check_right_grip_telemetry.py`: source-contract and sample-schedule test.
- `tests/report.json`: retained passing CPU report.
- `syntax-check.txt`: retained Objective-C++ syntax-only command and diagnostics.
- `manifest.json`: input and output hashes.

This overlay fixes telemetry semantics only. It does not validate the physical meaning of grip release, solver convergence, or a full loading experiment.
