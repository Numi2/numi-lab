# Solver telemetry and sparse-basis trial — 2026-10-21

All build and GPU execution in this record ran on SSH Numi (`Mac18,5`, Apple M6, 16 GiB, macOS 27.0 build 26A425). Toolchain: Apple clang 21.0.0 (`clang-2100.3.34.2`), Apple Metal 32023.883, CMake 4.4.4, Ninja 1.13.2. No viewer inspection was performed.

## Published increment

Base revision: `72dd867e1d4af7dec4bd23c3bf2ff8f3d27203bc` on `numi-human-aortic-contact-perf`.

The telemetry change adds per-microtick counters and terminal snapshots for active Krylov columns, exhausted solves, Newton iterations, residual norms and thresholds, latch state, and status. It does not alter solver control, physical residuals, acceptance, or rollback. Source patch SHA-256: `f01485879f55614ecec291fe9a0f952110195c40ed37a62240eff4b269c4474e`.

Build command:

```sh
cmake --build build-integrated-validation-20261019/profile-build-039 --target numi_matter_runtime numi_matter_shaders numi-human-native --parallel 4
```

The runtime, native Human executable, and fresh Metal library built successfully. The existing `matter.compiler.registered_skin_fat_reduction_labels` CTest passed 1/1. Build output contained compiler warnings in surrounding existing kernels; no build errors.

## Integrated control run

Exact source and artifact closure: [`native-build073-source-provenance.json`](native-build073-source-provenance.json). Invocation and logs are under `build073-control-1000us/`; the original asset identities and hashes are recorded in its `invocation.json`.

The existing resting-human invocation ran one outer step of `0.001 s`, with `NUMI_MATTER_WORK_COUNTERS=1` and sparse basis encoding off. Process wall time was `31.1303369 s`, including initialization and first-use setup, so the initialization-inclusive rate is `3.21e-5×`. This is about 3,110 times below `0.1×`; it is not a sustained-run throughput estimate.

The native terminal audit passed for one accepted step and eight completed microticks. All eight ticks ended at Newton iteration 23 of 24 with the domain nonlinear latch false. The mechanical domain norms ranged from `2.50e-5` to `8.33e-5` against a `1.25e-5` threshold. The run used 2,691 Krylov columns and recorded four budget-exhausted solves; the final reported nonpressure relative residual was `6.41e-5`. The status remained zero and the existing terminal audit passed. These short-run results expose unresolved convergence margin; they do not qualify sustained physiology or performance.

The large native state archive (`resting-human.mrvpack`, 297,078,784 bytes) and all run logs remain in the run directory and were not added to Git.

## Rejected sparse coefficient trial

The sparse consumer candidate was built from the same base revision with the telemetry and sparse changes combined. Its exact closure is `native-build072-source-provenance.json`; combined source patch SHA-256: `f1c8d9f1deacd6c2a3530e18591b1dd73dca34cdd4af7f9b0bb807cd6bb6c5ac`. The standalone sparse patch used during review has SHA-256 `ec10d317bae53a018ef66ce60f3e39c71a717d2d64be09484d867672f9d24a15`.

With sparse coefficient encoding disabled, that candidate completed one accepted 1 ms step in `36.42 s`, but still missed the domain latch. With `NM_FEM_HUMAN_PARTITIONED_SPARSE_BASIS=1`, the integrated run exited with status 10 at failing index 16 and accepted zero steps. Turning off paired tangent projection did not remove the rejection (status 10, zero accepted steps). Attempts to remove the tangent path entirely were rejected during runtime admission because the remaining selector-6 options require it; those attempts did not reach GPU simulation.

The sparse consumer code is therefore withheld from the publishable source. Its compile and synthetic CPU checker pass are not sufficient to overcome the integrated failure. It needs a dense-versus-sparse consumer parity check and a full-Human accepted-step comparison before reconsideration.

## Remaining result

This increment improves diagnosis, not throughput or convergence. No solver-limit or physical-acceptance gate was relaxed. The minimum `0.1×` target remains unachieved; neither the one-step control nor the rejected sparse trials qualify the full coupled run.

The native body summary reported a 1 ms step in 17.0287361 s, or 5.8724267e-5 real-time factor. The frozen-invocation process took 31.1303369 s including initialization and reporting, or 3.21e-5x end-to-end for this single step. Neither is sustained throughput; both are far below 0.1x. The current experiment interface caps the Newton override at 24 and FGMRES at 64. No convergence tolerance or acceptance gate was relaxed.
The retained standalone candidate patch is whitespace-normalized for clean Git storage (archive SHA-256 de24dc40760242f58b298cb84bc5ee437225c08fb61498bceb3f20942c51a4ff); the exact combined source patch used by the rejected run remains identified by the provenance digest above.
