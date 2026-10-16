# Native 468 source157 dt/2 command preparation

Preparation only: no native process, GPU launch, or simulation was run. The command is a source157 version of sealed command002; command001/002 remain unchanged and unrun.

## Rebound source/build identity

- Snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157`; frozen-source digest `2976540a244ba9af686e8033783a3d18da1ac48b030a949b2109fe81108f7cc0`; 787/787 source entries and 1034/1034 configure-support entries verified.
- Fresh build: `/Users/home/numi-cardboard-20261006/build-cardboard-provenance-157`; source-build provenance `78cbe2fac180130174e0585ea2398b7c7f07b2b7a5227ed99f1cd8ee9e6581b5`; source157 build package `5d2a07d62a17ad3d48a5f1f44255b539aeed8ce15cf9c74bfbdeffccd5b9e6a6`; compile-only 64 and 128 both report zero accepted steps and no runtime execution.
- Metallibs are byte-identical to source154: NumiMatter `89f9d756a0298fad44786b5333d67172636a8255e8c4f71b2696e4e5d6271114`; physical digest `0401b29d3ac2165153f36bea19786be653a10a242eed140e1c9008595ec1f729`. The host probe executable is source157-specific and bound separately.
- CMake source-relative root is `matter`; target is `numi-matter-cardboard-probe`. Source157 used 2 build jobs versus source154 4; this topology difference is recorded rather than called identical.
- `fillTargets` is byte-identical at function-body SHA `cfec49a4be92145586acbf0a3d99c59585475790148859165835e36bafafdbb4`. The telemetry-only change reads the commanded target mode for `right_grip_constrained`.

## Preserved protocol versus 467

The command retains the already-declared phase counts 24/6/12/6/6/6, dt `.00005`, outer Newton budget 64, certificate scalar `5e-7`, material iterations 32, FGMRES restart/limit 16/1024 and CGLS all-control settings from command002. Compared with 467 (12/3/6/3/3/3, dt `0.0001`, Newton 128, certificate `1e-6`), phase durations and total `0.003` second duration match. The 64-vs-128 Newton ceiling is different, so this is a dt/2 diagnostic, not pure temporal qualification.

The regenerated `protocol-clock.json` and `common-clock-target-comparison.csv` bind coarse source154 to refined source157. At all 30 common samples, phase, time, angle, target mode, and all 1620 right-grip target float components match; max coarse interval-average rate discrepancy is 0 deg/s. This is schedule/target transcription only, not native trajectory equivalence.

## Telemetry correction

The old source154 log expression marks post-release relaxation constrained (`phase == release ? 0 : 1`) despite the unchanged target generator assigning free mode `w=2`. Source157 logs the generated constraint mode and emits 0 for post-release. `post-release-telemetry-difference.csv` lists the six post-release steps across both environments. This changes reporting only; it does not change targets, solver state, or physical contact/retention behavior. No post-release physical qualification is implied.

## Conditional force bound

The source certificate is `||generalizedResidual|| / max(referenceScale, 1)`. For the pure mechanical coupon, while `referenceScale` remains below 1, the denominator is 1 and the certificate is numerically an impulse residual in N s. The selected 5e-7 threshold at dt 5e-5 corresponds conditionally to 0.01 N. After a run, verify scale==1 on each accepted row and separately evaluate `free_force_imbalance_l2_N` and `free_force_imbalance_max_N`. This command preparation makes no force-accuracy claim.

## Capacity and execution

At preparation, observed free space was 125775872 bytes; the requested 1.5 GiB native-launch minimum is 1610612736. The observed value is below that threshold. Root must recheck storage and publication audit before launch. The command is not dispatched here.

The canonical argument vector is `native-468-command.json`; 70 prior inputs and 33 source157 admission inputs are copied with unique basenames and content hashes in `external-input-bindings.json`. No raw build tree or metallib is copied into this package.
