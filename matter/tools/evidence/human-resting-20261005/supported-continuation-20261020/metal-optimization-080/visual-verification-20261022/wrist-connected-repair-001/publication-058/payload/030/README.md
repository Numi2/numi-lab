# Conforming D37 skin/fat prism candidate 030

This private candidate changes 16 tetrahedra around the left palm while preserving every O/I/D coordinate and boundary triangle. The existing unpaired volume owner validates exact-overlap ancestry, positive emitted Float32 cells, material boundaries, and conserved skin/fat debit. Cell mixtures use raw overlap divided by new-child volume; original parent-volume fractions are retained separately.

The app retains the original ObjectSource descriptor and physical owners, regenerates per-cell source labels through the baseline compiler, reuses existing regional basis IDs, and applies the corrected distinct-region capacity check. All preparation and native failures are retained.

Source preflight 006 passes. All 7,233 emitted node positions are bitwise identical to 024, total skin/fat mass differs by 7.28e-12 kg, and mass outside the 14 affected nodes is unchanged. Local nodal mass redistribution reaches 30.76%; this is a changed discretization, not equivalent-algorithm parity.

Native 002 requested 1 ms and rejected its first physical step after 90.06 s. The former first failing palm cell is no longer first; failure is now fat global tet 19641 at Newton 8, trial J 2.50019526/current J 2.49996257. Initial alpha improves 0.125→1 and failed nonlinear residual 0.00459133→0.000608571, but neither observation establishes convergence or speedup. Zero accepted steps means no sustained-throughput claim.

The source-only contact capture 029 retains 138 nearest-bank inside-solid witnesses. This candidate does not fix those penetrations, qualify anatomy, demonstrate joint motion, or close the full Human deliverable.

See native-comparison-analysis.json and source-parity-analysis-006.json for exact pins and numerical details. Use retained command JSONs; builds and measurements belong to SSH Numi.
