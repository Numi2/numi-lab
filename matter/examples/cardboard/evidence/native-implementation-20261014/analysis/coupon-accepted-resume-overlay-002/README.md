# Accepted coupon checkpoint/resume overlay (source157)

This isolated, default-off CLI feature saves a native `RuntimeStateSnapshot` only after a two-environment corrugated-strip control has been accepted by both environments. `--coupon-checkpoint-after-control N` uses a zero-based control index, writes the accepted checkpoint plus metadata/source-prefix manifests, and stops immediately with `accepted_checkpoint_prefix_saved`. The result is an intentionally incomplete protocol prefix.

To continue, run the same physical/solver CLI recipe through the source-bound diagnostic runner into a new empty output directory, passing both `--coupon-resume-checkpoint` and `--coupon-resume-metadata` from the saved run. The driver compiles and initializes the requested world, checks the current runtime's source-physics and device-program fingerprints, compares the exact protocol manifest, verifies checkpoint and prefix hashes/shape/certificates, then calls `Runtime::restore`. It begins at the next absolute scheduled control index. It does not rebuild state from OBJ or JSON.

The v2 resume metadata separates the strict non-output `NM_*` and `NUMI_MATTER_*` settings digest from an exact absent/set map for the seven source161 diagnostic output-path keys. Only those registered paths may change between source and resumed case. If they change, the driver requires the source case's runner-produced `prediction.json` and `native-environment.json`, verifies their hashes, schemas, contract digest, exact path values, and case-local capture-prefix containment, then validates the same runner contract for the resumed case. It consumes those runner-created paths as recorded; the probe does not generate or relocate capture paths. Other native settings, including selectors and numeric policy controls, remain in the strict digest. Explicit non-path settings in the runner contracts are also compared across segments. Arbitrary or unregistered `NM_*` path-like variables remain strict settings.

The path-key registry is bound to source161-002, patch SHA-256 `154c1ce851219615237b6d63c9bf9d543c6750fd9164c42e370932a916150789`. A source-bound runner is required only when the source checkpoint has active registered output paths; ordinary runs without such paths keep all path entries absent.

The source-prefix manifest binds every regular output file listed at the accepted boundary. Each listed file is checked at its recorded exact size and hash. An additional regular file added to the source output directory after the checkpoint is not currently rejected as an unexpected extra; keep the source run closed and unchanged. The resume run is a separate output segment. Its report explicitly says that trajectory parity and full-protocol completion are unestablished.

The feature rejects shipping, pin, blank/assembly, crease, compile-only, and non-two-environment runs. Checkpoint creation and resume cannot be requested together. A requested checkpoint control must leave at least one protocol control to continue. Invalid, incomplete, mismatched, or truncated data fails closed.

## Validation

`validation/run_validation.py` reruns the standalone C++ contract test, checks the seven-key and 49-key registries against the hashed source161-002 runner, and performs source157 compile-command-based Objective-C++ syntax checks for both ordinary coupon and box-native macro variants. The contract covers strict metadata/prefix parsing, checksums and path/cursor failures, runner proof/path relocation rules, non-output environment canonicalization, request admission, and stop-after-checkpoint boundary behavior.

The overlay has **not** been fully built, run on Metal, or compared against an uninterrupted native trajectory. Resume parity remains a required future native check before treating it as an exact continuation.

## Bound base

- Snapshot: `/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157`
- `matter/tools/cardboard_probe.mm` SHA-256: `8bea98e513afca4f283e61a25aed25e8c55f5c8bd628a720654e9253fc10f3cd`
- `matter/tools/cardboard_shipping_snapshot.hpp` SHA-256: `987eade18437c57a5e5ea364065ab7926c8911281d0b966034a44279b4f48475`
- `matter/tools/cardboard_shipping_driver.hpp` SHA-256: `81844531890af9c6822ad9c9de1ae7766a514f99fe14a8ec4459dbd596afeb2d5`

## Seal history

The first pre-seal manifest and patch are retained as `overlay-manifest.pre-seal.json` and `overlay.patch.pre-seal`. That draft manifest described earlier candidate bytes than the final validated files. The v2 manifest binds the current candidate, regenerated patch, and final validation outputs; the retained pre-seal artifacts are historical only.
