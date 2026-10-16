# Deformable contact Newton capture

Set NM_FEM_CONTACT_TRACE_ROOT and NM_FEM_CONTACT_TRACE_ITERATION to capture
one selected FEM Newton iteration. The runtime extends the existing
deformable-limit trace record with the exact active NMDeformableContactGPU
and emits one fem_deformable_contact_capture= JSONL event after the
deformable limiter. It includes active counts, per-environment truncation,
candidate node state, the FGMRES direction, active contact slot identities,
and contact records. It is transaction-local diagnostics, not checkpoint or
accepted-state data. The trace is capped at 128 active records per
environment and 4 MiB of native readback; any incomplete coverage is marked.
For the selected capture only, a diagnostic Metal kernel also calls the same
`deformableExactVertexTriangleDirectionalForce` helper used by the production
FGMRES gather. It exports the helper's action on the four source support nodes
in `contact.nodes[0..3]` order and one support marker per node. The action uses
`femSolution` as the direction and is labeled as the native FGMRES operator
action. The capture also records the per-object timestep passed to the helper,
which can differ from the owner microstep for multirate objects.
Unsupported contact/material branches export zero vectors with zero markers,
which the replay treats as unavailable rather than as measured zeros. This
read-only dispatch writes only the trace arena.

Decode and replay a sealed native log with:

    python3 matter/tools/cardboard_deformable_contact_derivative_replay.py \
      --capture /path/to/stderr.log --root 0 --iteration 42

The replay rebuilds the contact residual from captured geometry and compares
centered FP64 and FP32-emulated directional derivatives with the captured
frozen contact block. For unmapped frictionless vertex-triangle contacts on a
strictly smooth branch, it also computes the exact derivative of the
re-evaluated residual, including closest-point normal/barycentric motion and
the inertial barrier stiffness response `k = 10/(I dt^2 h^2)` with
`I = sum(w_i^2 invMass_i)`. It verifies this expression against independently
recomputed plus/minus residuals, including their recomputed `I` and `k`.
Other branches retain the radial-block comparison only. The report records
closest-feature, mollifier, stiffness, and barrier branches at the center and
on both sides, plus an FP32-rounded position-perturbation screen over moving
support nodes. It does not re-execute Metal contact generation or the exact
native line-search path. Friction, mortar mappings, impact stiffness, barrier
floor, uncaptured grid-node contacts, branch-near geometry, and invalid
mappings remain unsupported or branch-limited; they are not silently omitted.
When the native exact-action payload is present and all four support markers
are set, the report additionally compares those four Metal vectors directly
with the independent CPU formula evaluated on the captured candidate state
and direction. It reports relative error and cosine without an acceptance
threshold. Partial Metal support or a CPU branch the replay does not model is
reported as unsupported. Older captures without the payload remain readable
and are labeled `not_captured`.

The current Metal gather change applies that exact directional action only
for the same unmapped, frictionless, unit-mollifier, smooth-feature,
above-floor, strictly inertial branch. Every other contact retains the
historical radial block. That source edit still needs a source-bound native
build and rerun before it can be treated as qualified solver behavior.

This is a local operator diagnostic. It is not evidence that the global
nonlinear solve converged, that the CCD path is admissible, or that the
cardboard model is physically validated. Portable tests cover synthetic
point-point replay plus the exact vertex-triangle derivative on face, edge,
and vertex interiors, owner-timestep scaling, inertial-response variation,
and fail-closed feature transitions:

    PYTHONPATH=matter/tools python3 -m unittest \
      matter/tools/test_cardboard_deformable_contact_derivative_replay.py -v
