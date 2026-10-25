# Same-parent surface flip proposal (349)

This isolated helper creates a topology-only proposal on the 278 refined outer
surface. It never moves points and does not run the exact self, typed-bank,
fixed-reference-area, source-cap, or native audits.

The reusable API is `propose_flip_batch(candidate_arrays, target_face_rows,
max_flips=128, min_angle_gain_deg=1e-8, min_normal_dot=0.0)`. It considers
only interior edge pairs touching the supplied target rows and sharing the
same `refinement_face_parent_face_row`. It requires opposite directed edge
use, strict convexity in current and parent-barycentric charts, a consistent
037 reference reconstruction, matching current/reference orientation,
complete source/subface ancestry, and a strictly better pair minimum angle.
Selected flips are deterministic and vertex-disjoint.

For accepted pairs, the helper leaves all node coordinate and anchor arrays
unchanged; updates `source_O_faces` and
`area_reference_037_source_O_faces` together; updates per-face
`refinement_face_parent_barycentric`; unions source-face and fine-subface
ancestry; preserves immediate-parent boundary rings, the global boundary,
edge incidences, and directed winding; and records face-area changes for
downstream fixed-037 checks. Ambiguous parent metadata is set to -1.

CLI:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python same_parent_flips_349.py \
  --candidate /path/to/candidate.npz \
  --expected-sha256 EXPECTED_SHA256 \
  --target-face-rows-json /path/to/target-rows.json \
  --output-dir /path/to/new-run \
  --max-flips 128
```

The target JSON is a list of exact integer face rows, or an object containing
`face_rows`. The output is a candidate NPZ and a proposal report. A successful
proposal is not geometry-admitted; full exact F32 self-intersection, matched
037 area, anchor, source-cap, all typed-bank, and other required checks remain
mandatory.

Validation performed here: 11 focused unit tests pass. A schema-only preflight
against 278 candidate SHA
`c2a076267c2a8261a5f1a51b742a0b4e44471d31b200f5b809b8c98d6205348f` passes
for 20,889 points / 41,733 faces, and target rows 15082 and 32458 are in range.
No real flip batch, geometry audit, or native run was launched.
