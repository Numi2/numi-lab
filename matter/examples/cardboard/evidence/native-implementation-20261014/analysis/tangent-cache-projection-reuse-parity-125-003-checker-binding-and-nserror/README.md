# Checker correction 003: fixture hashes and nil-safe Metal errors

This checker-only correction is based on frozen source132 at
`/Users/home/cardboard-implementation-20261014/snapshots/source-cache-projection-reuse-capture419-132`.
It does not modify the source132 snapshot, production runtime, Metal kernels,
CMake, or any native run input.

## Corrections

The audit checks all nine C++ fixture digests for each bound run against the
actual capture, compiled package, candidate nodes, accepted-state sidecar, run
manifest, conditioning manifest, source-build record, receipt, and result. It
found one transcription defect in `kFixture360`: the compiled package digest
omitted `b94`. The corrected literal matches the file SHA-256. The other 17
C++ digest literals also match their respective files. `validateFixtureHashes`
rejects null, non-64-character, and non-hex digests; CPU admission exercises
those cases and then hashes every bound input.

`MetalRun::make` previously constructed its error string before checking
whether pipeline creation succeeded. A nil `NSError.localizedDescription`
produced a nil UTF8 pointer and caused the retained run428 segmentation fault.
The checker now constructs the description only on failure and uses a stable
fallback when Metal provides no error object. The original crash report is
copied and hash-bound below; it is not overwritten.

## Validation

- Exact C++-to-file fixture hash audit: 18/18 pass.
- Strict Objective-C++ syntax with `-Werror -Wall -Wextra -Wpedantic`: pass.
- CPU admission for run360 and run419 using source132 runtime/compiler libraries: pass; both report 3,840 capture rows, 1,764 node rows, one object, 59 parameters, and zero physical steps.
- Patch dry-run against the exact source132 checker: pass.
- Native parity was not run here. Native427’s package-hash rejection and native428’s SIGSEGV are preserved as failed preflight evidence. The copied 428 `.ips` is bound by SHA-256 in `source-overlay.json`.

Rebuild the existing checker target after applying `prototype/checker-source132-003.patch`, then run:

```sh
numi-matter-cardboard-tangent-cache-projection-reuse-check --cpu-admission /Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360
numi-matter-cardboard-tangent-cache-projection-reuse-check --cpu-admission-419 /Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419
```

Native runs must remain on the parent-managed GPU queue. The fix resolves only
checker admission and the successful pipeline-construction crash; it does not
establish cache parity or performance.
