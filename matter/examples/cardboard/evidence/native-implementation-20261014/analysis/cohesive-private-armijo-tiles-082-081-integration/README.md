# Ordered tiled cohesive private Armijo

This external integration overlay combines the cohesive tiling patch with
frozen source `source-portable-bct-binding-tests-081`. It does not modify the
shared checkout. The three production/fixture file deltas were authored
against source 076, then applied to 081 with no rejected hunks.

## Source binding

Base: `/Users/home/cardboard-implementation-20261014/snapshots/source-portable-bct-binding-tests-081`.

Base file SHA-256 values are in `base-081-sha256.txt`. The production patch
leaves the shared ABI unchanged. The compiler-measured sizes are 384 bytes
per `NMCohesiveFaceGPU` and 96 bytes per `NMCohesiveFaceVectorGPU`: legacy
private clone-plus-force storage is 480 bytes per face/environment. Tiled
storage retains only force records and has an 8 MiB cap including all
environments. There is an additional fail-closed check that face indices can
be encoded as `faceIndex * 6 + localSlot` without uint32 overflow.

## Runtime and transaction behavior

The opt-in environment variable
`NM_FEM_FULL_CONTACT_TILED_COHESIVE=1` is valid only with the existing
full-contact residual Armijo selector. It rejects conflicting selectors and
cohesive scratch aliasing. It evaluates every cohesive face in ascending
face-index pages. Trial evaluation reads accepted history and candidate
activation references, but does not write accepted face history. The ordered
per-node gather consumes the existing canonical incidence list and applies
the summed cohesive force once at the legacy residual/reaction point. No
atomic accumulation or face-state clone is introduced. Transactional commit
and rollback remain the existing runtime path.

`NM_FEM_FULL_CONTACT_TILED_COHESIVE_PAGE_FACES` is an optional positive
decimal page-size override, accepted only while the tiled mode is enabled and
only if the complete environment-strided force page fits 8 MiB, Metal's
buffer bound, and dispatch limits. The native fixture sets it to 1 to force
three ordered pages.

## Allocation correction

Final source review found that `pageElements` already includes the environment factor. The scratch allocator now receives that checked total directly, instead of multiplying by environments a second time. The allocation is also checked against the exact `pageBytes` value used by the 8 MiB/device-bound admission. This prevents a multi-environment request from allocating beyond the admitted page size; the helper test still verifies the environment-strided byte count.

## Checks completed without GPU execution

The portable CPU layout/gather check passed 61 cases. It covers one or more
environments, the 260,835-face planned input, checked byte/dispatch/encoding
limits, exact 384+96 accounting, no truncation, canonical incidence ordering,
and ordered FP32 gather parity including duplicate incidence contributions
and cancellation.

Strict Objective-C++ syntax checks passed for the full-contact checker and
runtime against this merged source-081 overlay. The checks used the platform
SDK, `-std=c++23`, and `-Wall -Wextra -Wpedantic -Werror`. The CPU helper used
`-std=c++20` with the same warning policy. These are compile/syntax results;
the Metal kernels and native parity fixture still require the integration
owner's shader build and GPU run.

The prepared native executable remains
`numi-matter-cardboard-full-contact-armijo-check`; after integrating the
overlay, run it with the existing bulk `.nmatter` argument and
`--cohesive-tiled-parity`. The integration owner must register the helper
CPU target and CTest option. The native fixture checks a zero-face fail-closed
request, legacy-versus-tiled one-face and three-face cases, accepted mortar
activation, accepted compression/closure, nonuniform opening and shear,
rejected post-commit rollback, and bitwise snapshot parity across nodes,
material/cohesive history, contact/rigid state, statuses, and certificates.
It explicitly reports the executed native FEM transaction count and labels
the law synthetic and physical validation false.

No native GPU result is claimed. Full shipping-scale admission remains
unqualified.
