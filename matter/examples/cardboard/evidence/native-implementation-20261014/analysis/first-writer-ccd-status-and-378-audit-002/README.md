# Private CCD status selection and 378 source-variant audit (corrected v2)

This version preserves the strict 362/369/371 result as `FAIL_UNEXPECTED_STATUS` and corrects v1's over-specific callsite attribution. Version 001 remains unchanged as the superseded, incorrect attribution record.

At root 1201, Newton 7, all three failed runs had the same source physics fingerprint and byte-identical 145-array center capture. The 362/369 trial status was code 6, candidate slot 283, failing index 283, diagnostics `[92,127,0,0]`. Run 371 was code 6, candidate slot 124, failing index 124, diagnostics `[24,127,1,0]`.

The retained center captures provide the stable primitive-kind/topology mapping used to disambiguate these tuples (the private trial failure set itself is not serialized). Their `deformable_contact_candidates[283]` is `(92,127,283,active)` and `[124]` is `(24,127,124,active)`. Primitive identity kind is 0 for both endpoints of both pairs; `NMContinuumSurfaceKind` defines 0 as triangle. The left primitive's captured `cohesive_surface_face_ranges` entry has count zero in each pair. The source helper returns an empty mixed-cohesive patch set for a zero-count range, so both candidates use the generic triangle-triangle fallback. Their diagnostics encode the edge indices passed to `edgeEdgeCCD`: edge pair `(0,0)` for 92/127 and `(1,0)` for 24/127. Neither tuple is point-point, vertex-triangle, nor mortar-complement failure. Candidate records, primitive kinds, and range counts are decoded from the capture using the exact little-endian shared ABI sizes and checked against the reported diagnostics.

In source 106, `recordFailure` in `common.metalinc:294-318` compare-exchanges the environment status from success; the winning thread writes the remaining status fields. The private `nm_contact_narrowphase_deformable` kernel dispatches one invocation per candidate slot (`contact.metalinc:3675-3704`, dispatched from `runtime.mm:14956-14980`). Each generic edge-edge uncertified invocation calls the same first-writer status recorder at `contact.metalinc:3991-4003`. Different reported winners under identical captured center inputs are consistent with competing private CCD failures and arrival-order selection. The failed private candidate/failure set was not captured, so this does not prove which other candidates failed in either run.

Run 378 is a separate source variant: source 107 changes the edge-edge distance iteration count from 17 to 33. Its trajectory diverges by Newton 5, and 22 of 145 center arrays differ from 371. It is not a same-input status-selection replicate. The strict status failure remains preserved; no parity normalization or physical pass is claimed.

## Reproduce

```sh
python3 audit.py
```

The script validates source hashes and native receipts, strictly decodes the candidate and primitive records, checks the incidence count that excludes the mortar path, compares center-capture payloads, and writes `report.json`. It does not infer an uncaptured private failure set.

`manifest.json` binds the complete package and all input captures, logs, and source files. The superseded v1 audit remains separately available at `../first-writer-ccd-status-and-378-audit-001/`.
