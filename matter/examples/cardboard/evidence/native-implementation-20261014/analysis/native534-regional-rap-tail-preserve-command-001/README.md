# Native 534 source201 regional RAP helper command

This package prepares two sequential runs of the source201 production Metal helper checker. Package creation does not execute GPU work. The runner requires --execute; --preflight-only checks bindings, disk, output uniqueness, and the live native-worker guard without launching the checker.

The first arm uses the Debug host binary (C++ command has -g and no optimization flag); the second uses the Release/O3 host binary. Both load the same integrated O3 NumiMatter.metallib. A separately compiled -O0 Metal AIR is bound as compile evidence only; it is not a runtime metallib. The CTest test is matter.metal.cardboard_regional_rap_tail_preserve with RUN_SERIAL=TRUE and TIMEOUT=120.

The checker dispatches the source201 restart RAP build/factor/fallback/identity/clear/reset production kernels. Its synthetic E2 fixture checks active environment 0 while environment 1 has tail marker set, with valid and invalid preserved RAP/LU/fallback payloads. After the Newton reset it checks both environments rebuild through the E*m*m dispatch extent. It performs zero physical transactions. Limitation: the helper seeds the tail marker word directly; it does not observe a real completed FGMRES event. Its reset assertion verifies the tail and factor-valid words, not every reset payload.

The guarded runner verifies the frozen 817-file source map, both source-build receipts and all three artifacts per build, CMake/CTest registration, the O0 Metal AIR, and the unique output path. It requires 2 GiB free before launch, preserves a 512 MiB post-run reserve, refuses live native workers, scrubs NM_ and NUMI_ environment variables, rechecks the worker guard between arms, and caps the combined sequential run at 120 seconds. It compares exact stdout/stderr hashes and exit codes.

Run only when the worker guard is clear and disk preflight passes:
 /Users/n/cardboard-implementation-20261014-host-macmini/venvs/cardboard-reference-185/bin/python3 /Users/n/cardboard-implementation-20261014-host-macmini/analysis/native534-regional-rap-tail-preserve-command-001/runner.py --execute

The unique output path is native/regional-rap-tail-preserve-534. Logs and the receipt are retained on failure. No O0/O3 run has occurred.

## Separate full-coupon prefix-parity outline (not run here)

A separate source201 full 60-control coupon should start from zero with the same numerical arguments and settings as the source198 run. Save the source201 checkpoint after control 16 as a comparison milestone, and compare controls 0 through 16 against source198 accepted-prefix/checkpoint 530. Compare accepted states and transaction histories per environment; do not require unconditional whole-world byte equality because source201 fixes the E*m*m restart launch coverage and may alter environment 1. Any continuation must resume a checkpoint produced by that source201 run. This helper package contains no coupon command and makes no solver or physical qualification claim.

Preflight history: the first runner check exposed an incorrect assumption about source-map paths (CMakeLists is relative to the snapshot root); that attempt and its no-launch error are retained in preflight-source-root-mismatch-001.json. After correction, the final worker-guard preflight verified all source/build/artifact/CTest bindings and refused with exit 75 because native532-root17-diagnostic-off was live (wrapper PID 75552; runner/probe PIDs 75555/75558). No native534 output directory was created. This is a guard refusal, not a checker result.
