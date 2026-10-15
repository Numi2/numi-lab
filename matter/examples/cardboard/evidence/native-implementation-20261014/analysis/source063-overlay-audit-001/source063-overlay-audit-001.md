# Source063 overlay audit

**Result:** Static review found the two-file overlay consistent with the declared phase and unit contracts. The new analytic tangent is descriptive only; existing qualification gates are unchanged. The source61 replay outputs remain immutable and retain the old `oracle_P_local_Pa` scale, so they are not corrected by this source overlay. No repository files or GPU state were changed.

## Binding

Snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-phase-correct-rigid-bridge-and-implicit-reference-integrated-063`
Parent revision: `62371ed242d94ce47fbad16e77f7b44c9ff050e2`. The frozen source manifests compare as 750 files each with exactly two changed paths. Their hashes match `source-overlay-063.json`:

- `matter/tools/cardboard_shipping_rigid_driver.hpp`: `80294fc6e16ce9fe86797a1aff654a9c2078040827538882f56fa0086485680c` (parent `1702190c9b66a3e8f1f699edd394ad03a48e2d113217d6b7995378a23507282b`).
- `matter/tools/cardboard_paper_gpu_check.mm`: `7c4c79b3cda88589fa63a757e99544c05ef8989934e0118affabb65ca4ca91f4` (parent `8c1993463150e9c8e6c2d7d2c881853c666105d7bb2054072d029f5190da0aba`).
- Unchanged `include/metalrobo/MetalWorld.hpp`: `08767c992145bc4818fdc33a191151a72632230c0a48f8ec0922485d3ab8b2ff`.

## Findings

The rigid adapter now supplies `rigidContactConstraints`, `rigidContactStatuses`, and their stride only for `postCommit` (driver lines 978–986). That matches the public pass contract: these are final per-environment contact results, readable after the contact solve during `postCommit` only (MetalWorld.hpp lines 395–400). MetalWorld schedules the post-commit callback after the contact/commit path. In pre-dynamics, the zero-initialized `EncodeRequest` leaves those fields null/zero. The adapter revision constant changes from `0x43415244424f4152` to `0x4341524442523032`, so the execution fingerprint changes as intended. I found no phase-contract violation. Source61 fingerprints remain source61 fingerprints; no source063 native bridge execution was available in this read-only audit.

The paper oracle stores stiffness and first-Piola quantities in MPa (`cardboard_paper_reference.hpp`, `Parameters` and `firstPiola`). Native `P_local_Pa` is in Pa. The new serialization multiplies the FP64 oracle stress by `1e6` before writing `oracle_P_local_Pa` (checker lines 2006–2012), which is the correct conversion. The new complete implicit derivative is formed from the analytic 13-state Jacobian at the converged FP64 projected state and multiplied by `1e6`; with the supplied `F_direction_local_per_s`, its output unit is Pa/s. The field is nested inside `accurate_fp64_replay`, marks zero-direction relative error null, and labels its full-vector norm comparison descriptive. Its comparison is against the accurate Metal response tangent while the oracle derivative is evaluated at the independent FP64 projected state; that is a reference-vs-candidate comparison, not a same-state tangent-only error. This is clearly non-gating and is consistent with the existing FD reference comparison.

I found no changes to pass/fail thresholds or qualification predicates in the patch: the added analytic tangent does not feed `tangentPass`, `accuratePass`, or the exit status. JSON emission closes the new tangent object inside `accurate_fp64_replay`; the checker reparses every emitted JSONL record with `NSJSONSerialization`. This audit did not execute the source063 checker.

## Source61 replay interaction

The full replay `native/canonical-fp32-all-center-replay-192` explicitly binds to snapshot `source-canonical-fp32-replay-and-rigid-checkpoint-integrated-061`, uses the raw `current-failure-22-state-capture-153/operator-capture`, and has `executed_build_verified: true`. Source/build/output binding hashes:

- Raw capture: `f4b4bdbe7a64cdc6d09ca46aeb13fc525bb9c7678dbb5acc4b78aadf367f3bfa`; its manifest: `27147eb7c222e62e52f80283be9db265ec34ffcbb235d705053e7d83e29cd982`.
- Source61 build provenance copied into the run: `9be88d76384bfc632ca27375af10298e7c79995aadd1aa62da7212332951288b`.
- Stored source61 `center-replay.jsonl`: `6cfbe8daca85fe8fe26d9bc28596b2ee8260815d41f0114facc76b1c8c76bc58`; prediction: `416a2c9003210946635a0715229100b9d648df24435805fab8242b35226614c5`; receipt: `b417c69bf0c4e9fe2d5497336d93cdb560caf8c8b1adb82303c0b2124dae9ed9`.

All 2,880 source61 replay rows have the pre-fix unit error: `accurate_fp64_replay.oracle_P_local_Pa * 1e6` exactly equals `candidate_responses.fp64_root.P_local_Pa` for every component. Example row `(environment=0,tet_id=0,component=0)` stores `-0.7758534489` under the Pa label while the FP64 root response is `-775853.4489 Pa`. None of the 2,880 rows contains `independent_implicit_tangent`. These are historical source61 outputs; the raw capture itself remains a valid input, but corrected values and the new field require a fresh replay with the source063 checker. The targeted source61 replay 191 exited 0; the full replay 192 exited 2 (1,740 accurate nonpasses, 1,440 ordinary nonpasses), despite validating its emitted JSON. Both reported zero physical FEM steps, so neither is physical box validation.

The original Robertsson source/energy convention inconsistency is outside this two-file overlay and remains open.
