# Existing Metal viewer geometry replay

The existing skin replay executable now also consumes an optional complete
anatomy mapping exported once by the native resting viewer. It dispatches the
same hash-pinned Metal deformation and normal kernels. It does not advance
physics, physiology, accepted-state history, or control.

Set NUMI_HUMAN_RESTING_CAPTURE_REPLAY_MAPPING=1 for a selected accepted geometry
capture. The existing v1 receipt adds anatomy_source_mapping with exact
uploaded maps, complete positive influences, seed vertices, indices, cardiac
maps and normal incidence tables. Existing skin-only receipts still work.

Build the existing target numi_human_resting_skin_replay. Its five arguments
remain SOURCE_RECEIPT SOURCE_SHA256 POSE_RECEIPT POSE_SHA256 OUTPUT_DIRECTORY.
For same-frame verification, pass the same newly captured receipt twice.
Cardiac replay inputs must be pinned; the native viewer retains the active
common coordinates and explicitly inactive legacy chamber inputs. Retained
trace-derived inputs require a separate explicit derivation record and are
counterfactual geometry inputs, never raw GPU captures.

Measured on SSH Numi, Apple M6, macOS: the final existing executable reproduced
all 2,915,743 complete 80-byte vertex records exactly, including normals and
tangents, from the native selected frame. Position/normal mismatches: zero.
The old skin-only interface also reproduced positions/normals/tangents exactly
for 61,950 vertices. The opt-in native capture kept geometry, accepted body and
respiration state, and the coupled physiology CSV byte-exact against the
prior native capture. The measured geometry dispatch was 0.0146 seconds;
this is not a simulation real-time factor.

The captured native scene ran only 0.002 simulated seconds. These are
instrument/parity results, not anatomy admission, retained-motion clearance,
sustained physiology qualification, rollback qualification, or clinical
validation. Source/binary/runtime identities and exact build commands are
retained in the adjoining reports. The viewer used its pinned retained runtime
ABI, shader and shared libraries; a full source-head runtime rebuild is not
claimed. Direct owning native builds passed with warnings treated as errors;
the complete repository CMake build was not rerun.
