# Constrained finger repair and registration ownership

The old final074 source candidate removed intersections but compressed one PIP surface direction to 0.0666 of reference; it is superseded and remains unadmitted. Broader smooth blending also failed. The new bounded solve preserves the registered neutral surface, limits surface deformation, and changes 28 right-middle-finger weight rows relative to the restored source weights.

Candidate SHA-256: 6e2a95abba679c9e1a1e7be227d3a947899a1b31d73a31c0ce3a838e7641f592.
All eleven retained neutral/individual-finger detailed-surface poses have zero measured self-intersections. At the tested middle PIP bend, minimum/maximum principal surface stretches are 0.3878/1.6577 and minimum area ratio is 0.4224. The declared engineering gates are 0.35 minimum stretch, 2.4 maximum stretch and 0.25 minimum area ratio; these are numerical distortion limits, not measured biological limits. Neutral geometry is preserved within 37.91 nm; topology, stored normals and vertex records outside the 28 rows remain exact relative to the restored candidate.

The updated NHSKIN preparation owner rejects source-hash and registration-fingerprint mismatches, in addition to the existing binding checks. Eleven regression tests pass. replay.py reproduces all three constrained repair steps byte-for-byte through that owner using the recorded external inputs and per-step weight parameters. The reconstructed .nhskin is retained locally, not duplicated in this source commit.

This is source preparation and geometry evidence, not physical admission or complete joint range-of-motion qualification. Shoulder/elbow folds and ankle tissue-to-bone crossings remain open and their failed candidates are retained separately. D24 regenerates coarse donor frames and transfer data from this source before native validation; no default scene promotion is implied here. Existing mixed-source anatomy remains a reference assembly.
