# Source125 projection-reuse checker extension: exact capture 419

This is a separate checker-only follow-up to sealed package `tangent-cache-projection-reuse-parity-125-001`. It leaves that original package and its checker unchanged. It extends the same parity checker to accept either the exact run360 fixture or exact run419 fixture, while correcting the package-layout expectation exposed by source130 native preflight 423.

The failed 423 check expected 84 parameters. Both compiled package files are version 19, one-object packages with 59 parameters: package section ID 10 has count 1 and element size 176; parameter section ID 6 has count 59 and element size 16. The checker now requires those observed counts. It also binds the run’s source-build JSON, source snapshot path, execution receipt, native result, compiled package, capture, run manifest, candidate-node file, accepted-history sidecar, and conditioning manifest. The dynamic world fingerprint is written to output, including for run419.

## Fixture coverage

The CPU audit in `validation/fixture-coverage.json` recomputes both input admissions and records the exact coverage. Both captures contain 2 environments, 882 nodes and 1,920 tetrahedra per environment, 1,764 node rows, 3,840 tetrahedron rows, and 3,840 valid cached tangent entries. Candidate node data matches every serialized capture node word, and environment 0 accepted state matches the bent-state sidecar bit-for-bit.

For run360 (Newton 23), the active environment has 1,152 15-state kraft liner tets, 576 15-state NSSC medium tets and 192 stateless elastic starch tets. At the declared observable cutoff `1e-12`, accepted accumulated plastic multiplier is nonzero in 110 liner and 365 medium rows; the captured center projection has a positive multiplier increment above the cutoff in 278 liner and 255 medium rows. The remaining rows are reported as having no increment above that cutoff; the capture does not provide a separate constitutive branch label. All 576 medium frames and 192 starch frames are nonidentity, while liner frames are identity. Fdot is nonzero in all active-environment rows.

Run419 (Newton 48) has the same material/tet counts. Above the same cutoff, accepted multiplier is nonzero in 77 liner and 276 medium rows; positive center-projection increments occur in 153 liner and 356 medium rows. The same frame pattern is present; Fdot is nonzero in all active-environment rows.

Thus the captured inputs include stateless elastic starch, paper tets with and without a measured positive plastic-multiplier increment, nonzero accepted plastic history, and rotated material frames. These counts are state-threshold observations, not a per-row branch certificate. The captures pass nonzero Fdot and the `deltaFdot = deltaF/dt` direction contract, but none of the three authored constitutive assets references `Fdot` in its equation text. They do not cover a rate-dependent constitutive law or multiple physical-rate cases.

The native checker compares the original cache-building kernel with the projection-reuse sibling from the same source125 metallib on the same package, candidates, accepted history, parameters, frames, rates and timestep. The historical captured tangent matrices are reported only as a secondary comparison; they are not the parity gate. No physical validation, full-solve parity, performance improvement, or production qualification is claimed.

## Validation

CPU-only checks completed:

- Python admission/coverage audit for both fixed runs: pass; full hashes, package section counts, all candidate node words, and all environment-0 accepted-state rows checked.
- Objective-C++ source syntax: pass with `-Werror -Wall -Wextra -Wpedantic` against frozen source125 headers.
- Checker diff dry-run against frozen source130 checker: pass.
- Native dispatch: not run by this package owner; root owns the serialized GPU queue. The earlier 423 failed preflight is retained and is not a parity result.

Apply `prototype/checker-capture419.patch` to source130 (its base checker SHA is listed in `source-overlay.json`), rebuild the existing checker target, and run CPU admission first:

```sh
numi-matter-cardboard-tangent-cache-projection-reuse-check --cpu-admission /Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360
numi-matter-cardboard-tangent-cache-projection-reuse-check --cpu-admission-419 /Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419
```

The corresponding `--native` and `--native-419` commands perform zero physical steps but require the source-bound physical Apple GPU; run them only through the parent-managed queue.
