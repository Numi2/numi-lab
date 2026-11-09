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
