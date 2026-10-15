# Source122 first status-5 observer overlay

This package adds an opt-in, bounded diagnostic capture for the first `NM_STATUS_INVALID_DEFORMATION` event in each of two independent domains per environment: the ordinary/global transaction and a private Armijo trial. It is based on the immutable source122 snapshot named in `manifest.json`.

The observer runs immediately after the relevant status-producing dispatch and before a following dispatch or rollback. It copies the already-authored status, microstep, candidate and accepted node state, and—when the producer reports a tetrahedron index—the tetrahedron, exact float32 material parameters, accepted internal state, high/low deformation words, material-frame rate input, determinant, and timestep. It does not evaluate a constitutive program, write solver state, change status, or affect acceptance. Global and private records use separate slots, so a prior private rejection cannot consume the global slot.

Geometry validation is a special case: `nm_fem_armijo_validate_geometry` places the solver iteration in `failingIndex`, not a tetrahedron index. The observer preserves that index kind and status verbatim; it does not infer a tetrahedron or fabricate a material tuple. A geometry-stage record therefore has no material-input completeness flags. This distinction is explicit in each record's `capture.w` index-kind field and in the report consumer.

## Bounded scope and activation

Set both `NM_FEM_STATUS5_CAPTURE_ROOT=<decimal control step>` and `NM_FEM_STATUS5_CAPTURE_PATH=<absolute output stem>`. Only the selected control step captures. The output parent must exist, and the `.bin` and `.json` outputs must not already exist. The capture rejects unsupported layouts before encoding: it admits 1–8 environments, one object, nonempty FEM nodes/tetrahedra, no particles/cohesive faces/MPM grids/MPM blocks/human attachments/vascular unknowns, material state stride no greater than 32, at most 16 records, and at most 1 MiB of capture storage. It does not raise any production capacity.

The binary contains two zero-initialized records per environment in environment-major order: global at `environment*2`, private Armijo at `environment*2+1`. `capture.x` is the record-present marker; `capture.yzw` are stage ID, domain, and index kind. `identity` stores object, original status failing index, typed subject index (or `NM_INVALID_INDEX`), and private trial index. `payload.x` is a presence/completeness bitset: bit 0 status/microstep; bit 1 valid tet/material/nodes; bit 2 node-only witness; bit 3 F high; bit 4 rate; bit 5 accepted material history; bit 6 line-search alpha; bit 7 complete bounded parameter slice. These bits indicate captured fields, not their numerical validity. `payload.yzw` retain material index, state count, and parameter count.

Stage IDs are 10 kinematic target, 20 first physical internal-force evaluation, 30 residual-Armijo private geometry, 31 residual-Armijo private internal force, 40 capsule/box private geometry, 41 capsule/box private internal force, 50 later physical internal-force evaluation, and 60 global FEM validation. FD-only diagnostic evaluations are intentionally not observed. The runtime JSON records the source/world/device fingerprints and SHA-256 of the raw payload. The raw ABI is the C/Metal `NMFEMStatus5CaptureGPU` struct; consumers must validate `record_bytes`, `payload_bytes`, root, environment count, and payload hash before decoding.

## Validation performed

- C++ ABI/slot contract check: `tests/status5_capture_contract_check.cpp`.
- Static source contract check: `tests/status5_capture_source_contract_check.py`.
- Objective-C++ syntax-only check for the overlay runtime.
- Full `matter_kernels.metal` compile and metallib link. The metallib exports `numi_matter_metal::nm_fem_status5_capture`.
- No native transaction was run by this package. It is instrumentation, not evidence identifying source392's first failure until a source-bound native capture is produced and validated.

The Metal compile emits two warnings in the pre-existing FGMRES kernels (`sharedValid` and `refinementValid` potentially uninitialized); the observer adds no warning. The exact source, patch, test, and artifact hashes and reproducible commands are in `manifest.json`.
