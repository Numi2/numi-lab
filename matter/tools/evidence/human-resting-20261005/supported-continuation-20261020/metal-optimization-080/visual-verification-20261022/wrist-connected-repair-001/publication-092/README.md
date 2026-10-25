# Publication 092 — joint interface construction audit

This package records a bounded common-field gauge adjustment to connected skin/fat interface construction. It is construction evidence only. It does not admit the mesh to physical simulation, establish material-mass transfer, or qualify a native runtime.

The full emitted-F32 interface audit in attempt 020 passed its surface gates: 624,902 facets (384,796 skin and 240,106 fat), zero exact global self-intersections, zero nonpositive/nonfinite facet-quality values, and zero facets below q=1e-5. Minimum facet q was 1.0236998195739106e-5 for skin and 1.2189648262852507e-5 for fat. The audited surfaces came from the field candidate in attempt 019.

The common gauge changes 276 source-tet field rows. Its recorded sign/zero-pattern proof leaves the outer zero set unchanged. Nodewise skin/fat thickness residuals are at most 8.674e-19 m. Maximum emitted cutpoint movement was 0.580305 mm, or 23.5476% of local hskin; this is a construction perturbation, not a biological tolerance or accepted material change. Exact local skin/fat volume deltas are in the attempt 019 report. The independent field-only receipt confirms the seven intended field rows changed, outer-zero rows remained bitwise unchanged, and both thickness residuals stayed within 8.674e-19 m; it makes no geometry or mass claim.

Attempt 015 is the earlier full baseline: one skin interface facet was below q=1e-5, so it failed that surface gate. Attempt 017 is retained as a rejected local candidate with one facet below the same floor. Attempts 016 and 018 stopped before writing a candidate/report because their runners rejected measured cutpoint motion above the then-configured 10% construction ceiling. Their separate status receipts are post-run summaries, not as-run reports. Attempt 019 repeats the +105 um target after a 25% construction ceiling was authorized and records the measured motion and gate results.

**Source tetrahedra are a separate gate.** The foundation used by these interface audits has 461,669 tetrahedra, including 191 below source-cell q=1e-5 and minimum source-cell q 3.316e-6, per the separately retained run-013 source quality report. P092 includes no tetrahedral quality remesh. Passing interface-facet q/self-intersection checks does not repair or waive those source-cell findings.

The foundation checkpoint and registered baseline fields are bundled. The large original joint-facet input is pinned by SHA-256 in evidence/source-pins.json but not duplicated to keep this package below 60 MB. The copied runners retain as-run source; their historical absolute input paths are not rewritten for standalone replay.

Base: origin/main 88b7ee5cd3a41c9ff3d160f4799db7e47ca7d278 (tree 26014a4161e574ab66a8f4f9eb2abbe2854f8ba6). No canonical source changes, commit, or push are included. Geometry admission is false; native admission is false; accepted physical steps: 0.
