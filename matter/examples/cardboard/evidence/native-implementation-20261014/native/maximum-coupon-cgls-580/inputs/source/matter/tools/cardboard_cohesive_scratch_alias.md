# Full-contact cohesive scratch alias

`NM_FEM_FULL_CONTACT_REUSE_COHESIVE_SCRATCH=1` is an opt-in memory-layout
experiment for the exclusive `NM_FEM_FULL_CONTACT_RESIDUAL_ARMIJO_ROOT` path.
It aliases the private line-search face and force arrays to
`cohesiveFacesCandidate` and `cohesiveFaceForces`. The default remains separate
private buffers. Invalid values, non-exclusive Armijo selectors, missing
buffers, or a failed exact-pointer check reject initialization.

The admission envelope is unchanged: at most 4,096 cohesive faces, at most
four environments, and no more than 8 MiB for the face-plus-force arena. The
existing full-contact mixed-face and static/kinematic-object restrictions also
remain in force. This alias does not raise a face, operator, topology, or
source-scope limit.

The reuse is safe only within that admitted transaction path:

1. `nm_fem_cohesive_face_forces` reads irreversible quadrature history from
   `cohesiveFacesAccepted`. For a mixed face, it reads the candidate face only
   as the prepared activation/reference geometry, then overwrites its derived
   point history, work, and summary state. It zeroes each force record before
   evaluating that face, including inactive faces.
2. The ordinary path copies the prepared candidate face to a private trial
   face before each private residual evaluation. The alias path skips that
   copy because source and destination would be the same buffer. Each force
   evaluation still starts irreversible evolution from accepted history; no
   previous trial history is used as input.
3. The force array is consumed by the same trial residual gather after the
   force kernel. The next trial fully rebuilds it. After line search, the
   accepted-endpoint production residual rebuilds production face/force
   outputs before gathering them. On successful completion,
   `nm_fem_commit_cohesive_faces` publishes candidate faces; after a failed
   transaction it restores candidate faces from accepted faces.
4. Full-contact admission excludes mutable topology, so no face insertion,
   removal, or index remapping can invalidate either alias during the
   transaction. Accepted histories remain in their separate accepted buffer.

The diagnostic line
`fem_full_contact_residual_armijo_enabled` reports
`cohesive_scratch_layout` as either `private` or
`production_candidate_force_alias`. The native fixture option
`--cohesive-scratch-alias-parity` compares activation, loaded, and rejected
transactions against the default path. It requires `--diagnostic-state-output`
so both complete traces and state snapshots are retained. The checker compares
accepted arrays bit-for-bit and requires the full traces to be identical after
normalizing only this layout label, including selected line-search values and
certificate/rollback telemetry.

This is a scratch-lifetime and transaction-parity experiment. It does not
qualify corrugated-board material behavior, shipping-scale memory use, or
physical contact. A source-bound native run is required before relying on the
opt-in path; the Metal cohesive law itself is unchanged.
