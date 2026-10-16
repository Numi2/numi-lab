# Source175 sparse private-contact scan integration overlay

This is a narrow source overlay over the immutable source173 checkpoint/warm-start snapshot. It applies only the eight source170 files listed in `three-way-proof.json`; it is not a copied source tree and does not modify source169, source170, source173, or their sealed evidence.

## Merge method

The source170 delta is based on source169. Seven of its eight files were unchanged from source169 in source173, so those seven files are byte-for-byte source170 candidates. `matter/src/runtime.mm` changed in both branches. The integration uses a three-way merge with source169 as the common base; the 169→173 warm/checkpoint hunks and 169→170 sparse scan hunks are disjoint, and `git merge-file` reports zero conflicts. The merged runtime retains the source173 checkpoint/warm changes and adds the source170 sparse scan changes. The generated patch applies to source173 and reconstructs the exact candidate hashes recorded in the proof.

## Validation

- Reverified all 804 source hashes for both frozen source169 and source173 manifests.
- Reverified the source170 overlay manifest, patch, and all eight candidate file hashes.
- Applied `source175-integration.patch` to a temporary source173 subset and verified exact candidate hashes for all eight changed paths.
- Sparse admission CPU contract executable passed five checks; shipping pair scan remains within the existing 36 MiB admission budget, with 6,335,568 pairs, 24,749 blocks, and 98,996 offset bytes.
- The integrated sparse native checker compiled and its `--cpu-only` route passed three contracts with zero Metal kernels executed.
- Objective-C++ syntax checks passed for merged `runtime.mm`, the full-contact checker, and the shipping memory checker.
- The full production Matter Metal translation unit compiled and linked at O0 and O3. The compiler emitted two inherited warnings in source173 `fgmres.metalinc`; both builds linked successfully.

No native differential or GPU dispatch was run. This overlay does not raise runtime caps, claim shipping admission, or qualify performance or physical behavior. Native checker execution remains separately gated by the source170 checker package and the parent GPU queue.

## Shipping admission barrier audit

The first explicit full-contact Armijo scope barrier for the nominal shipping topology is the `femNodeCount > 8192` check in `femCapsuleBoxResidualArmijoScope`; `femFullContactResidualArmijoScope` calls that scope before its cohesive and deformable-contact admission checks. The supplied 528,444-node topology is therefore rejected at that scope. The separate `femNewtonTraceScope` has a stricter 4,096-node diagnostic limit, but that is not the full-contact scope’s first check.

Removing the node gate alone would not admit the target: the box scope and self-contact scratch allocator both require at most 32,768 tetrahedra (target 1,477,728), while the scratch allocator also requires at most 16,384 nodes and at most 131,072 deformable-contact capacity. Cohesive trial evaluation needs the tiled path for 260,835 faces: the existing 8 MiB page holds 87,381 faces at 96 bytes per face and takes three pages. A non-tiled pass is capped at 4,096 faces. The sparse private-pair arena is not the blocker for the supplied one-environment arithmetic: its checked 36,189,188 bytes fit the unchanged 36 MiB limit. Two environments would require 72,378,360 bytes and be rejected.

The type-bound size probe and detailed formulas are in [shipping-admission-audit-001](shipping-admission-audit-001/README.md). It also records which arena sizes are per-array or conditional, and why the prior 10.5 GB native allocation observation cannot be treated as a peak measurement for the larger cohesive topology. No 16 GiB shipping admission or physical qualification is claimed.
