# Accepted coupon checkpoint/resume overlay (source157)

This isolated, default-off CLI feature saves a native `RuntimeStateSnapshot` only after a two-environment corrugated-strip control has been accepted by both environments. `--coupon-checkpoint-after-control N` uses a zero-based control index, writes the accepted checkpoint plus metadata/source-prefix manifests, and stops immediately with `accepted_checkpoint_prefix_saved`. The result is an intentionally incomplete protocol prefix.

To continue, run the same physical/solver CLI recipe into a new empty output directory, passing both `--coupon-resume-checkpoint` and `--coupon-resume-metadata` from the saved run. The driver compiles and initializes the requested world, checks the current runtime's source-physics and device-program fingerprints, hashes the `NM_*` and `NUMI_MATTER_*` environment, compares the exact protocol manifest, verifies checkpoint and prefix hashes/shape/certificates, then calls `Runtime::restore`. It begins at the next absolute scheduled control index. It does not rebuild state from OBJ or JSON.

The source-prefix manifest binds every regular output file present at the accepted boundary. `observations.csv` is checked at its recorded exact size and hash; the source run stops at the checkpoint, so it must not be appended or edited. The resume run is a separate output segment. Its report explicitly says that trajectory parity and full-protocol completion are unestablished.

The feature rejects shipping, pin, blank/assembly, crease, compile-only, and non-two-environment runs. Checkpoint creation and resume cannot be requested together. A requested checkpoint control must leave at least one protocol control to continue. Invalid, incomplete, mismatched, or truncated data fails closed.

## Validation

`validation/run_validation.py` reruns the standalone C++ contract test and source157 compile-command-based Objective-C++ syntax checks for both ordinary coupon and box-native macro variants. The contract covers strict metadata/prefix parsing, checksums and path/cursor failures, runtime fingerprint separation, runtime-environment canonicalization, request admission, and stop-after-checkpoint boundary behavior.

The overlay has **not** been fully built, run on Metal, or compared against an uninterrupted native trajectory. Resume parity remains a required future native check before treating it as an exact continuation.

## Bound base

- Snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157`
- `matter/tools/cardboard_probe.mm` SHA-256: `8bea98e513afca4f283e61a25aed25e8c55f5c8bd628a720654e9253fc10f3cd`
- `matter/tools/cardboard_shipping_snapshot.hpp` SHA-256: `987eade18437c57a5e5ea364065ab7926c8911281d0b966034a44279b4f48475`
- `matter/tools/cardboard_shipping_driver.hpp` SHA-256: `81844531890af9c6822ad9c9de1ae7766a514f99fe14a8ec4459dbd596afeb2d5`
