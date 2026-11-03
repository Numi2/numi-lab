# Barrier-aware release placement and reference-axis certification

The default behavior is unchanged. Opt-in `NUMI_HUMAN_FEM_SUPPORT_BARRIER_RELEASE=1` positions the existing fixed flat support plane once, so the emitted Float32 outer skin begins at the existing IPC activation distance. It does not change the pinned anatomical frames, topology, tissue laws, mass, contact coverage, or per-step force ownership.

The first root-translation experiment correctly failed anatomical frame registration. Its source pins and failure are retained. The replacement moves only the physical support plane by -9.95956997940084 micrometres. Compiled minimum clearance is 10.0000002163 micrometres. This is an unbalanced release initialization, not a qualified supported equilibrium.

The opt-in reference-width path now tries additional edge/face separating axes when closest-feature and face-normal proposals cannot certify a sufficient gap. Every proposal uses the existing compensated projection certificate. Hard collision clearance, CCD and final acceptance are unchanged.

## Verification

- Whole-body alignment/release unit check passed, including Float32 emission, invalid inputs and failure preservation.
- Native FGMRES contract passed 81,642 checks on Apple M6. This is numerical kernel coverage, not full-Human qualification.
- Matched full-scene initial contact residual fell from 0.1110717622 to 0.0003516103 (99.6834%). Both changes were tested together; their individual gains are not established.
- Full Human still accepted zero physical steps. It reached Newton iteration 6 and rejected skin/fat cell 717649 at J=2.50083375, above the unchanged 2.5 limit. No long-run or performance qualification is claimed.
- A preceding trial stopped at the unchanged 12 GiB disk guard. SHA-256 verified deduplication recovered 11.7523 GiB while retaining completed evidence paths and bytes.

Exact source/binary/library pins, plans, logs, resource samples and failed attempts are retained alongside this note. Unrelated execution-checkout changes are not part of this publication.
