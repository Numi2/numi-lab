# FEFCO 0201 native boundary-action controller

This helper turns the authored flat blank in `cardboard_box_blank.hpp` into
tool and boundary commands. It does not impose a folded bulk mesh pose. The
fold planner reads accepted FEM snapshots, fits wall/flap rotation from
interior nodes, and gives only selected boundary nodes temporary kinematic
grips. The tetrahedral interior remains part of the native FEM solve.

`makeBoxScoreTooling` creates a fixed-orientation die with twelve capsule
segments, one for each authored score-line ID, on one prescribed-translation
body. It also creates a separate support anvil. The die can be lowered to
first contact and retracted. Current paper material inputs have no
crease-crush or permanent-hinge law, so this contact does **not** simulate a
scored or weakened crease and must not be counted as irreversible scoring.

The native driver clamps exposed boundary nodes on the manufacturer-tab body
at the initial accepted pose. This ideal tab fixture supplies a reference
frame for the first signed wall turn; it does not move interior nodes or
prescribe a folded pose. Flap turns are measured against the accepted
action-start state. Wall turns are measured as the signed difference
between each wall's accepted interior material-axis fit and the adjacent
tab/wall fit. Targets are -90 degrees for wall 0 relative to the tab, then
+90 degrees for each following wall. A rigid rotation of the entire flat
blank therefore cannot satisfy the first wall target.

`makeBoxFeatureGrips` resolves the four body-wall distal edges and eight flap
free edges into stable FEM node IDs. For wall and flap actions, the planner
returns target positions for a boundary increment about the measured accepted
hinge frame, tagged `w=3` for acquire/update. Completed wall clamps remain
active as external supports through later tube and seam actions. Flap clamps
release with `w=2` when their action is complete. Commands go through
`EncodeRequest::femKinematicTargets`, and the next action uses only an
accepted snapshot. A rejected step remains a failed attempt; the next plan
must use the restored accepted snapshot.

Angle completion has an additional accepted-shape gate. Wall fits include the
out-of-plane residual of interior 3-D nodes. Both the active wall and its
adjacent tab/wall fit are required independently; the diagnostic uses the
larger RMS residual and lower coverage, so a larger panel cannot mask a poor
fit on a smaller parent tab. Flap fits compare interior nodes at the
accepted action-start and current states under the measured rigid rotation
around the current score-line axis. Each fit reports an RMS residual and the
fraction of sampled nodes within its residual bound. Completion requires RMS
at or below `0.5 * boardCaliper` and at least 95% coverage. These are explicit
engineering diagnostics chosen for this controller, not published corrugated
board tolerances. The native probe exposes `--box-shape-residual-calipers`
(default `0.5`) and `--box-shape-coverage-min` (default `0.95`); the bound in
metres is the first value times the authored board caliper. If the mean angle is at
target but the shape gate fails, the plan is invalid with no grip targets; the
native driver stops and records `shape_fit_rejected` rather than advancing or
changing interior coordinates. `box-actions.csv` records residual, bound,
coverage, whether the shape is within the configured bound, and whether a
target-angle completion was rejected. A moving action can show
`shape_fit_within_bound=0` while it is still progressing; the gate is applied
when the target angle is reached.

In fixture-held joint mode, the driver verifies all four completed wall edge
clamps before releasing the tab-body clamp, then acquires the seam fixture.
The tab's free edge is part of the initial anchor, so it must be released
before that fixture moves the edge. This sequence is explicitly an external
fixture trial: it is not physical assembly without fixtures, does not seal
the seam, and has no calibrated adhesive law.

The die's first-contact center height is `boardCaliper + punchRadius`; the
park position is supplied by the caller. The press/retract path is a tool
trajectory, while the missing crease constitutive law remains explicit.

The 15-action program preserves the blank schedule: four wall turns, the
manufacturer-joint action, eight flap turns, and two end-seal actions. The
joint and end-seal actions require adhesive/cohesive behavior in the source
description. Without fixture mode the driver stops at that missing law;
fixture mode records externally held or clamped, unbonded states.
`measureBoxClosureGapDiagnostics` reports nearest-node distances for the tab
seam and opposing flap pairs. The 0.5-degree default is a command-completion
tolerance, not a published manufacturing acceptance limit. Gap outputs are
resolution-dependent geometry diagnostics, not continuous surface gaps,
contact-force acceptance, or bond tests.

The CPU check covers the twelve die IDs, support/tool body identities, all
fifteen action identities, wall/flap target increments, completed wall-clamp
retention, fail-closed bond actions, and seam/flap gap diagnostics. It also
checks that a proper 90-degree rigid rotation of the whole blank leaves the
relative wall turn at zero and does not complete the first fold, and that tab
anchors are selected from exposed boundary nodes only. Target-angle
counterexamples buckle interior nodes out of plane while preserving the fitted
mean angle. One warps half of the active wall; another warps 10% of the smaller
parent tab while the active wall remains good. The latter also verifies that a
sample-weighted aggregate would pass the configured bounds, but the
worst-required-panel gate rejects completion without producing interior targets.
A matching flap counterexample alters
accepted flap interior points along the hinge axis while preserving its measured
mean turn; the local rigid-motion shape gate rejects it as well. These CPU
geometry checks do not validate physical paperboard behavior. The Metal fixture adds
temporary-grip snapshot restore, injected rollback, episode reset,
invalid-tag, and exact authored-rest checks. It still requires a native Apple
GPU run before runtime behavior is confirmed.

## Shipping-size protocol extension

`cardboard_shipping_protocol.hpp` adds a separate 300 × 200 × 200 mm internal
FEFCO 0201 target and an accepted-coupon-derived dieline. It does not change
the mini-box fixture, its controller schedule, or its output. The shipping
controller sequence is available as `makeShippingAssemblyProgram`; it names
wall turns, accepted manufacturer-lap activation, minor/major end-flap turns,
accepted flap-overlap activations, fixture release, and a settled checkpoint.
Every stage remains `executed=false` until a native driver records it. See
[`cardboard_shipping_protocol.md`](cardboard_shipping_protocol.md) for the
coupon, HMA, compression/drop, qualification, and mesh-resolution limits.
