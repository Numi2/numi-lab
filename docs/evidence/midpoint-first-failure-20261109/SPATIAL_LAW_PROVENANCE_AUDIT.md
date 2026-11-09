# Spatial reference law and provenance audit

The recorded source hashes for the n=32 serial, n=64 serial, n=64 ordered
parallel, and n=128 parallel reference revisions match the corresponding
retained manifests. Their pressure and energy functions match after whitespace
normalization. The audit also checks the recorded source expressions for the
same 100 µm activation width, contact coefficient `1e16`, material constants
`mu=2000 Pa` and `lambda=5000 Pa`, density 1100 kg/m³, cube dimensions, initial
200 µm gap, gravity, and triangle-based area weights. This checks source
identity and selected physical inputs; it is not a proof of all numerical
implementation equivalence or a spatial accuracy result. The separate
reference analyses check the complete weight distribution and total area.

The n=128 coarse execution records the binary hash and it matches the manifest.
The historical n=32 and n=64 execution records used in this audit do not
record an executable hash or command, so a manifest and small-mesh parity
results do not independently bind the executable to those runs. Keep that
provenance limitation with the comparisons; do not upgrade it to a direct
binary-binding claim. The running n=128 fine result must still pass its own
temporal gates before interpreting the n=64-to-128 comparison.

Reproduce `spatial-law-source-audit-001.json` with
`python3 tests/contact/audit_spatial_law_provenance.py`. The accompanying test
checks the retained hashes and verifies that a changed law source is rejected.
