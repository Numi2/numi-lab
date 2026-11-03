# Human solver/contact finalization — SSH Numi, 2026-11-03

This is a measured initialization-prefix optimization, **not 0.1x performance, supported-rest qualification, or motion qualification**. Active linear solves remain the principal unresolved solver cost.

## Selected implementation

Builds and native runs used SSH Numi: Apple M6, Mac18,5, 16 GiB RAM, macOS 27.0 (26A425). No visual verification was performed.

1. **Parallel accepted FEM scheduler observation.** Replace the one-SIMD-per-object tetrahedron walk with immutable 256-tetrahedron tiles and min/max reduction. Reuse expired Armijo geometry scratch after final certification. MPM, contact, urgency, rejected-status and scheduler-update logic remains shared with the legacy kernel.
2. **Final exact-zero geometry reuse.** Extend the existing GPU exact-zero-correction and unchanged-working-set proof to final broadphase/BVH/candidate construction. Final current narrowphase, friction/Hessian work, physical residual measurement and contact certification still execute. Admission remains restricted to immutable single-environment FEM with no coupled candidate or rigid generalized coordinates.
3. **Suppress unused intermediate linear builders.** After the real current-RHS begin measurement, a healthy finite complete nonlinear-convergence latch may select zero work for the FEM diagonal, cavity-Schur and vascular principal-factor builders. Active solves, diagnostics and the final Newton refresh retain full work. Existing GPU indirect-dispatch storage is reused; no readback or large cache is introduced.

Opt-in settings, all default off:

- NM_SCHEDULER_TILED_FEM_OBSERVE=1 requires existing distributed immutable FEM tiles.
- NM_CONTACT_FINAL_ZERO_GEOMETRY_REUSE=1 requires the existing exact-zero geometry-reuse path.
- NM_FGMRES_SKIP_COMPLETED_LINEAR_BUILDERS=1 requires NM_FGMRES_SKIP_COMPLETED_CYCLE_WORK=1 and its existing single-environment admission.

Use the pinned configuration in completed-linear-builders-063/install-motion.plan.json, not these settings in isolation. Material laws, precision, contact floors, timestep, final tolerances, physical-state ownership and acceptance/rollback processing are unchanged.

## Measured result

Each trial accepted four 15.625 microsecond steps: only 62.5 microseconds of simulated initialization. These are means of four GPU command spans, correlated consecutive steps rather than independent steady-state repetitions.

| Trial | Configuration | Mean GPU step | Accepted-loop wall time |
| --- | --- | ---: | ---: |
| 059 | Matched control, detailed profiling off | 8.281006 s | 40.851158 s |
| 061 | Tiled scheduler + final geometry reuse | 6.548331 s | 34.023335 s |
| 063 | Above + unused intermediate builder suppression | 6.373134 s | 33.544269 s |

059→061 reduces mean GPU time **20.92%**. 061→063 reduces it **2.68%** in this single comparison. Combined 059→063 reduction is **23.04%**, approximately **1.299x GPU throughput**. Do not multiply this by unrelated historical improvements.

All three captured endpoint hashes match:
66c476e14b867eb9a19039ec112750a3d517a18ce039b24bdb5181fec7ac24c7

Their coupled physiology CSV hash also matches:
77edb932a1a06eacdba8efe1f97bc5492ff8185c37a2e31a5c8d09045eb41aee

All four roots retain 32 used Krylov columns, two active Newton begins and status 0. **No reduction in active linear work has been demonstrated.** Trial 063 reports accepted-loop real-time factor **0.00000186321x**. Initialization-inclusive times remain in the receipts. Native presentation qualification is pending; these are not sustained or presentation-inclusive results.

Selected detailed root-0 measurements from 056/057:

- Scheduler: 1,644.406167 ms serial → 8.849042 ms tiled producer + 0.804833 ms finish.
- Final certificate stage: 299.207667 ms → 234.562458 ms.
- 057 also contained the subsequently removed shared-patch experiment. Final-source uninstrumented 059/061 confirms the whole-root improvement without that experiment.
- Stage and selected-kernel intervals overlap; do not sum them. The after-candidate-ready stage measures after the callback, not its scheduler cost.

Three independent matched pairs and sustained supported-rest timing remain outstanding. The additional builder gain is provisional until repeated.

## Tests and rejected experiments

Final shader/runtime build passed:

- Full native FGMRES true-residual contract: **916,531 checks**.
- Focused completed-cycle/builder dispatch checker: **362 checks**, including 18 range scenarios and all-domain accumulator parity.
- Focused tiled scheduler checker: **91 checks**, including bitwise legacy parity, multiple environments, empty/inactive/nonfinite cases, failed status and canaries.
- Integrated 063: four accepted steps; all 26 source pins match after execution; endpoint and coupled CSV parity as above.

The full suite does not automatically run every focused-only checker. These counts are separate receipts, not a sum of independent coverage.

Rejected approaches remain with source snapshots and limitations:

- Shared contact directional actions, combined and split: exact component parity, no whole-run gain; split cache worsened the active linear stage. Large cache removed.
- Shared Schwarz patch values: exact parity but no useful active-stage gain; removed.
- Existing EW cap 0.001: same active columns, no useful gain.
- EW caps 0.005 and 0.05: four accepted steps but **failed predeclared regional force/strain fidelity gates**. The 0.05 fourth step increased Newton work. Neither is selected.
- Resource-stop, invalid CLI and timestamp-capacity trials remain failed or partial in the inventory.
- Initial scheduler fixture allocated one environment of tetrahedra while declaring two. Fixture corrected; no production gate weakened.
- Timestamp-preserving restore initially left stale Ninja outputs. Caught before 059 ran; sources were touched and shaders/runtime/checker rebuilt. Stale and corrected logs are retained.

## Reproduction and retained evidence

Execution checkout: /Users/numi/Work/skin-fat-native-integration-209

Remote evidence root: /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125

Each plan records the exact native invocation. Its retained runner executes the install-motion stage. The included run-finalization-056.py prepares a new uniquely named trial, refreshing source, binary, dependency and asset pins. Never rerun into completed evidence.

The current full endpoint reference on Numi is completed-linear-builders-063/install-motion.stdout.log. Compact GitHub logs omit only endpoint node/tetrahedron rows; full stdout hashes remain in receipts. Source snapshots, plans, compact traces, build/test logs, resource monitors and fidelity reports are retained here. Large MRVs and complete raw/compressed endpoint logs remain at their recorded remote paths.

Historical stdout compression was roundtrip-verified against execution receipts before raw-copy removal. The lossless-retained-log-compression manifest records restore paths and hashes. Identical generated outputs were hardlinked only after size/hash and fresh inode/mtime checks. No anatomical sources were removed.

Execution retains unrelated dirty work. Publication changes only four owned runtime/shader/checker files plus evidence. Current-source-parity verifies equality with execution source. Comparison and trial-inventory JSON are the compact entry points.

## Remaining work

The next performance experiment must address measured active operators or stronger conditioning while preserving physical gates. Profile 047 identifies gathering, FEM element application, patch/coarse preconditioning and the vascular operator as active costs. Tested action caches added traffic and did not help.

This prefix cannot establish long-horizon material/contact stability, supported rest, timestep qualification or anatomy-through-motion validity. Numerical parity does not establish physiological plausibility or clinical validation.

Archive consolidation: source-snapshots/manifest.json records canonical gzip and exact uncompressed archive identities. Per-trial archive paths are relative links; original compressed bytes remain recoverable from fc8ca33230895b1aeb0c831964e0eaa5403206f9. Frozen remote run records are unchanged.
