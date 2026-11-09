# Retained 64-division evidence and 128-division continuation

The temporally qualified 64-division result was produced on the separate
`codex/matter-midpoint-reference-20261108` research branch in commit
`d49466d68`. Its completed coarse and ordered-parallel fine traces, execution
records, analysis, and report are reproduced byte-for-byte in this branch so
that the follow-up can be independently evaluated. The `h025` not-run
sentinel remains distinct from the actual `parallel-h025` result. Running
`tests/contact/spatial_n64_analysis.py` against that fine result reproduced
the retained analysis byte-for-byte. Both temporal comparisons pass, while
the 32-to-64 spatial comparison fails all four prescribed gates.

`tests/contact/spatial_n128_analysis.py` checks the 128-division execution
records and parity-checked binary identity, mesh sizes, weight sum, mass,
clearance, and trace shape before reporting separate temporal and spatial
comparisons. The 64-to-128 result is interpretable only if both meshes pass
their own temporal gates. It is still a CPU reference for this fixture, not a
native coupled rollout or a benchmark. The two n=128 results are pending.

The pinned 32-division serial, 64-division serial and ordered-parallel, and
128-division parallel source blobs were checked explicitly. Their mesh and
nodal area construction, pressure and potential, and parameter declarations
match: density 1100 kg/m³, Lamé parameters 2000 and 5000 Pa, activation
width 0.0001 m, and area-law coefficient 1e16 N/m⁵. The two implementations
assemble forces differently; the existing ordered-parallel parity evidence
remains necessary. `reference-law-provenance.json` records the blob hashes and
`tests/contact/check_reference_law_provenance.py` reproduces the source check.
The retained source and binary hashes document the historical build inputs;
they do not independently attest the historical compilation process.

The analysis now reconstructs the two reference triangles in every bottom
surface cell and checks each nodal weight, the zero off-surface weights and
their total, rather than relying on the total alone. A sum-preserving
redistribution, an off-surface transfer and a nonfinite value are each rejected
by `tests/contact/reference_area_verification_test.py`. Recomputing the
completed 32-to-64 analysis with the stricter check reproduced its previous
result byte-for-byte. This strengthens input verification; it does not turn
the failed spatial gates into a pass or supply the pending 128 result.
