# Coupled capture scope overlay (source 085 → proposed 086)

This is an external, diagnostic-only overlay. It changes only `matter/src/runtime.mm` and is based on the immutable source-085 file below. It does not change solver equations, candidate state, contact/collision logic, acceptance, rollback, or tolerances. No GPU run is part of this package.

The prior source-084 capture attempt skipped because it rejected every nonzero particle count. The source-085 evidence showed one incidental particle record but no MPM grid or block topology. This overlay allows a bounded incidental particle record only when all grid/block counts are zero, and captures the particle accepted/candidate/default/checkpoint records, their material histories, and the available MPM indexing/dispatch metadata. The scope remains observational; it does not admit MPM mechanics into the coupled Newton candidate.

The selector and callback now call the same scope predicate, so full-RHS preservation cannot be enabled for a state that the later readback refuses. History stride is bounded by `NM_MAX_MATERIAL_STATE`; captured FEM histories are sized by tetrahedron count, particle histories by particle count. The existing 32 MiB payload limit is unchanged. Descriptor capacity is 192; the source contains 171 unique possible array append sites (optional entries are omitted when absent), with duplicate names rejected by the static check.

The captured input shape remains restricted to one environment, at most 2,048 FEM nodes, 8,192 tets, 64 particles, 64 rigid layouts, 8,192 generalized unknowns, 131,072 contact pairs, 16,384 deformable-contact capacity, 8,192 cohesive faces, and 64 objects. Any nonzero grid-node, MPM-grid, MPM-block, or MPM-block-lookup count is rejected. Capture records state and dispatch; it does not establish capture success until a native run emits the `.coupled.json` and `.coupled.bin` payloads.

## Bound inputs and outputs

- Base file: `/Users/home/cardboard-implementation-20261014/snapshots/source-accepted-correction-regional-recycle-085/matter/src/runtime.mm`
- Base SHA-256: `d482a43db6af754b9578f78ac7a814267b104c33ff2750bfe14101b9ec301919`
- Proposed runtime SHA-256: `f9899a9e7772089204fb6c30e4e61d1ef4ea26c45c5c29085719e6f6de586150`
- Unified patch SHA-256: `921125cc6816a226d01c1432b695134b50080d97888fbec40d124f30d0b7cbd2`
- Static scope check SHA-256: `dfe945c563b22f6596f61830cf9a9b02609563b43cf5d531659fb0073667afb0`

## Checks performed

- `python3 /Users/home/cardboard-implementation-20261014/analysis/coupled-capture-particle-scope-085-overlay/verify_overlay.py` — passed.
- Objective-C++ syntax-only compile using the frozen-085 `runtime.mm` compile flags and snapshot source include path — passed (exit 0).

No physical/native coupled-capture qualification is claimed. Root should integrate this overlay into a fresh immutable source-086 build, then rerun the same `1201:12` selected capture. A skip or invalid capture must remain a skip/failure, not be interpreted as a successful Newton or a qualified physical step.
