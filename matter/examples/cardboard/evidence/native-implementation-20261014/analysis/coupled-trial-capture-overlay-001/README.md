# Coupled private Armijo trial capture overlay

This external overlay adds an opt-in, read-only capture around the existing
full-contact FEM/rigid/cohesive private Armijo trial. It is based on the frozen
source-089 `matter/src/runtime.mm`; it does not change shaders, solver policy,
contact law, accepted history, or physical acceptance.

The selector is `NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR=root:newton:trial:phase`
and the output stem is `NM_FEM_COUPLED_TRIAL_CAPTURE_PATH`. `root`, `newton`,
and `trial` are zero-based unsigned integers. `phase` is `pre`, `post`, or
`both`. The output stem must not already have `.coupled.bin` or `.coupled.json`
files; both are created exclusively. A simultaneous center/F-D capture is
rejected so the fixed readback budget cannot be double-booked.

The payload is little-endian raw buffer data. The JSON manifest uses schema
`numi.matter.coupled_armijo_trial_capture.v1`, contains one entry per captured
array with phase, dtype, count, offset, byte count, and SHA-256, and hashes the
full payload. The single snapshot buffer is capped at 32 MiB and the array
table at 192 entries. Missing optional buffers are listed by phase and name;
missing required buffers, bad extents, device mismatch, capacity overflow, or
an incomplete selected sequence produce a skip record and never reject or
accept a solver trial. No command buffer wait is added.

The `pre` context is copied after the private FEM candidate and private rigid
candidate have been formed, but before private internal force or deformable
contact sweep validation. In the later contact-rebuild helper,
`nm_contact_validate_armijo_deformable_sweep` reads the Newton `femCandidate`
and trial `femArmijoCandidate` buffers at indices 5 and 6; endpoint
`nm_contact_narrowphase_deformable` reads the trial-node argument at index 4,
the supplied frozen deformable history at index 10, and cohesive faces at
index 15. The `post` context is copied only after those calls, compaction,
incidence construction, and deformable residual accumulation. No separate TOI
scalar is exported, so consumers must use the captured status and contact
records instead of inferring a TOI from endpoint geometry. The residual stages
capture the already assembled private RHS after
bulk mechanics, cohesive forces, rigid contact, deformable contact, and the
complete residual immediately before Armijo measurement. The `post` context
contains the rebuilt trial geometry and contact workspaces before free-body
residual terms. The final metrics/status/line-search buffers are copied before
the decision kernel. The JSON names the producer ordering and the current
producer conventions for the deformable-contact sweep and endpoint
narrowphase; there is no independent geometry generation counter in this
overlay.

The capture includes accepted and candidate FEM nodes/tetrahedra, accepted and
production-candidate material-history buffers, rigid state/body data and
generalized candidate/output, borrowed rigid q/v and external-wrench inputs
(when their authored counts are nonzero), contact/cohesive/deformable-contact
records and histories, trial workspaces,
element/cohesive force records, status/metric/line-search data, and the
per-stage residual vectors. It does not include an internal material
projected-stress/state buffer because the production private-force path does
not expose one. The capture is an observational replay input, not a checkpoint
and not evidence that a rejected trial was accepted. Native/on-off history
parity and any independent residual replay still have to be measured by the
caller against a source- and executable-bound run.

## Verification

`check_trial_capture_overlay.py` performs source-bound static contract checks
against the exact frozen-089 base and overlay. The Objective-C++ syntax check
uses the flags and include roots recorded in
`build-cardboard-provenance-089/matter/CMakeFiles/numi_matter_runtime_objects.dir/flags.make`.
No GPU run is part of this overlay package.
