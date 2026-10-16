# Full-contact Armijo scratch accounting overlay 176-002

This isolated overlay fixes a concrete admission undercount in the optional private full-contact Armijo arena. The previous formula charged 480 bytes per deformable-contact slot and omitted `femArmijoDeformableCandidateStatuses`, which is allocated as one `NMMatterStatusGPU` per slot. The ABI is 48 bytes per status. The corrected exact slot cost is therefore 528 bytes, before node ranges and other typed buffers.

The helper reconciles the actual `privateScratch<T>` allocations and its 16-byte/one-element typed minimum. It checks arithmetic in 64 bits, includes the surface primitive/BVH/sort arrays only when they are not aliased with production surface scratch, and preserves the unchanged 128 MiB admission cap. It does not estimate total runtime peak: production arenas, solver vectors, regional factors, optional incidence-sort workspace, cohesive pages, shared framework allocations, and allocator overhead remain outside this private-Armijo-only subtotal.

## Exact allocation totals

The private per-slot inventory is:

- 16 B candidate record
- 48 B candidate status (`NMMatterStatusGPU`)
- 432 B contact record
- 4 B active index
- 24 B six-node incidence entries
- 4 B global active index

This is 528 B per slot. The persistent production per-slot inventory has no private candidate-status array, but has accepted, candidate, and checkpoint contact-history records (3 x 48 B), so it is 624 B per slot: 16 + 432 + 144 + 4 + 24 + 4.

For the shipping geometry counts used by the preflight (528,444 FEM nodes, 1,477,728 tetrahedra, 1,054,876 surface faces, one environment, zero additional grid primitives assumed in this *modeled component*):

- `C=131,072`, private full-contact Armijo, aliasing production surface buffers: 77,661,184 B (includes 16 B per-node range x 528,444 and typed scalar sentinels); E=2 is 155,322,304 B and fails the unchanged 128 MiB cap.
- `C=131,072`, private full-contact Armijo, no surface-buffer alias: 296,269,008 B and fails the cap.
- Production contact-slot arrays at authored default capacity `C=8*T=11,821,824`: 7,376,818,176 B, before surface/BVH, node ranges, sort workspace, solver, and other runtime arenas. This default capacity is outside the current full-contact Armijo scope; it is not silently clamped to 131,072.
- At the bounded `C=131,072`, stable corner-sort workspace is 12,582,912 B/E: logical records `6*C=786,432`, padded to 1,048,576, with 8-byte keys and 4-byte payloads.
- For the static mapped surface count P=1,054,876 (with no added grid nodes), `sizeof(NMContinuumSurfacePrimitiveGPU)=64`; the BVH leaf count is 2,097,152, and four float4 records per leaf consume 134,217,728 B/E. Four uint sort arrays use 16,878,016 B/E. These production buffers are not part of the private 128 MiB subtotal when they are aliased.

All counts are modeled byte arithmetic, not an observed device peak and not a shipping admission result. If grid nodes are present, they increase P and node-range bytes; the exact compiled dispatch must be used before any real allocation claim.

## Scope and remaining admission limits

The effective full-contact path calls `femCapsuleBoxResidualArmijoScope`, which calls `femNewtonTraceScope`. That nested scope rejects `femNodeCount > 4096`; the later direct full-contact check's 8192 limit cannot widen it. The same path rejects tetrahedron count above 32,768 and deformable contact capacity above 131,072. The independent true-residual scratch/capability path also requires N<=4096, T<=32768, and cohesive faces<=4096. Thus the actual shipping topology remains unsupported even if its scratch estimate fits a component budget.

These N/T limits are explicit current scope/capacity guards; the source reviewed here does not establish that 4096/32768 are fundamental Metal uint address limits. Separate concrete address guards exist: runtime checks surface primitive and environment products against uint32, checks the padded BVH bound index domain, and the sorted-incidence path checks node products, encoded contact/local IDs, logical/padded sort sizes, dispatch products, and host byte arithmetic. A valid explicit capacity C is not proof that all contact pairs fit: candidate count is measured, an overflow status is recorded if the exact count exceeds C, and the transaction is rejected rather than accepting a truncated candidate prefix.

The legacy node-incidence builder scans all active contacts and six nodes once per continuum node in both count and scatter, i.e. work proportional to `N*C*6` (twice, before inner validity skips). The opt-in stable corner-sort route emits the same canonical `(contact slot, local node)` order with O(C) records plus sorting; its environment contract is exactly `NM_CONTACT_NODE_INCIDENCE_SORT=1`. This reduces a major work-scaling barrier, but does not lift the N/T/C admission limits or prove the full shipping physical protocol.

Cohesive Armijo evaluation has an opt-in tiled path that limits its resident per-face-force page to 8 MiB and loops every page; the non-tiled path rejects more than 4,096 cohesive faces. The separate true-residual operator still rejects more than 4,096 cohesive faces, so tiling alone does not enable true-residual qualification at the shipping cohesive count.

## Validation

- CPU helper compiled with C++20, `-Wall -Wextra -Wpedantic -Werror`; 17 checks passed. They include exact struct sizes, 48-byte status accounting, E1/E2 capacity boundaries, checked-overflow rejection, typed zero-count sentinels, shipping-count node-range/surface estimates, and the unchanged cap.
- The same runtime candidate passed strict macmini Objective-C++ syntax checks at O3 and O0 with `-Wall -Wextra -Wpedantic -Werror`.
- No native run or full fresh source build is claimed for this overlay. A native full-contact checker command is prepared separately, but must use a fresh executable/metallib built from this overlay and a root-owned serialized GPU runner.
- The related source175 scan native run is separate evidence: 49 cases, 12,910,103 checks. It does not validate this overlay or full-contact shipping behavior.
