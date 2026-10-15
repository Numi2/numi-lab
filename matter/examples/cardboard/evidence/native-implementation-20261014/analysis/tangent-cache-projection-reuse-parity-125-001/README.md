# Source125 tangent-cache projection-reuse parity overlay

This is an external, diagnostic-only source overlay based on the frozen
source-status5-vtccd-diagnostics-125 snapshot. It adds a sibling kernel that
projects the accurate generic 15-state bilinear material once at tangent-cache
column zero, then reuses that exact projected high/low state for columns 1–8
within the same tetrahedron invocation. The original production cache kernel
is preserved byte-for-byte. The optional evaluator hook is only exercised by
the new sibling kernel and is admitted only for the accurate compensated
15-state generic policy with no projection-diagnostic outputs requested.
The temporary projection buffers use NM_MAX_MATERIAL_STATE words (32) because
the evaluator initializes/returns that full capacity.

The checker binds run360's exact package, candidate-node file, accepted-state
sidecar, operator capture, conditioning manifest, and run manifest by SHA-256.
It also verifies all candidate-node words against the capture's serialized
center records, verifies env0 accepted history against its sidecar, checks
capture/package parameter words, then dispatches the existing cache kernel and
the sibling reuse kernel against the same two-environment package, candidates,
history, frames, rates, and timestep. It compares cache-valid bits and every
row-major 9x9 entry for each valid tetrahedron. It then applies both caches
through the production nm_fem_apply_operator_elements on (a) the captured
Newton correction direction and (b) a fixed-seed independent direction.
Native parity requires exact valid-bit, matrix-word, and action-word equality;
the output JSON is written before a mismatch causes a nonzero exit.

The run360 capture was produced by source101. It is being used here as a
source-bound exact input fixture for kernels compiled from source125; the
captured historical tangent matrices are reported as a comparison only, not a
gate. The actual parity gate is old-kernel versus sibling-kernel output built
from the same source125 metallib. This does not establish runtime speedup,
full-solve parity, physical validation, or production qualification. No GPU
dispatch has been run by this overlay owner.

## Files

- prototype/common.patch: adds the default-null, narrow projection-reuse
  hook to evaluateAlgorithmicStressTangent.
- prototype/fem.patch: adds the sibling diagnostic cache kernel and leaves
  the original kernel unchanged.
- prototype/CMakeLists.patch: adds the standalone native checker target.
- prototype/matter/tools/cardboard_tangent_cache_projection_reuse_check.mm:
  source-bound exact-input cache/action comparator.
- prototype/matter/src/matter_kernels.metal: unchanged source125 kernel
  translation unit copied for the isolated metallib compile.
- validation/check_capture_admission.py: CPU-only exact capture/hash and
  input-word checks.
- validation/check_overlay_contract.py: checks source preservation, buffer
  capacity, kernel symbol emission, and patch-local expectations.
- validation/compile_metallib.sh: reproduces the isolated Metal compile.
- validation/metallib/NumiMatter.metallib: source125-derived overlay
  metallib, compiled successfully.
- validation/capture-admission.json, overlay-contract.json, and logs:
  retained CPU and compile receipts.

## Reproduction

From the implementation root, first run:

    python3 analysis/tangent-cache-projection-reuse-parity-125-001/validation/check_capture_admission.py
    python3 analysis/tangent-cache-projection-reuse-parity-125-001/validation/check_overlay_contract.py

To compile the overlay Metal library:

    analysis/tangent-cache-projection-reuse-parity-125-001/validation/compile_metallib.sh

For native parity, integrate the two Metal patches and CMake patch into a
fresh source125 copy, add the checker source, configure/build
numi-matter-cardboard-tangent-cache-projection-reuse-check, then run:

    numi-matter-cardboard-tangent-cache-projection-reuse-check --cpu-admission /Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360

    numi-matter-cardboard-tangent-cache-projection-reuse-check --native /Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360 /absolute/path/to/native-output

The --native branch uses the checker target's source-bound metallib and
requires a physical Apple GPU. It performs zero physical or solver steps.
Keep it on the parent-managed serialized GPU queue. Do not use the historical
capture matrix comparison by itself as a pass criterion.
