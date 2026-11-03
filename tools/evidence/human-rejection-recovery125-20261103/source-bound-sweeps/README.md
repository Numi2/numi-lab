# Compensated source-bound contact sweeps

The full coupled Human advances from two to eight consecutive accepted 15.625 us
steps (125 us total). The next step exhausts the experimental four-Newton budget
and fails unchanged nonlinear/vascular residual gates. This is an unbalanced
release transient, not a supported-rest, motion, physiology-duration, or
performance qualification. The 0.1x requirement remains unmet.

## Fixed causes

- Root 2's FGMRES solve passes true-residual certification. Three duplicate rows
  then select the same sub-nanometre VT pair as a time-zero CCD hit. Independent
  Float64 inspection finds 0.718677 -> 0.716696 nm plane distance: positive,
  slightly closing motion, not an intersection.
- For opt-in reference self-contacts only, a dyadic numerical margin is certified
  against immutable rest geometry (0.5 nm for this source pair). It never derives
  from the previous accepted state. Source intersections cannot acquire a
  positive margin. Rest-margin selection is independent of current orientation.
  Ordinary/nonreference contact policy stays intact. This explicitly admits
  positive source gaps smaller than the ordinary 1 nm numerical margin.
- Whole-sweep proofs use compensated candidate axes and outward-rounded
  projection bounds at both endpoints. A prefix stops before a proven margin
  violation. Fixed-axis affine projection bounds certify the entire sweep.
- A separate production VT query was uncertified during contact rebuilding.
  The narrowphase now tries a compensated full-activation-thickness separating
  proof only after legacy CCD fails. It preserves all legacy certified results.
  It cannot remove a force-bearing row: a full-sweep distance beyond the entire
  activation band is required. Unknown cases retain failure.

Material equations, mass, volume ownership, anatomical geometry, forces,
nonlinear tolerances, true-residual checks and accepted-state publication are
unchanged. Exact source pins, failed builds, failed native trials and witnesses
are retained. No visual verification was performed.

## Validation

The native Apple GPU contract passes 82,945 checks. Added cases cover captured
picometre closing motion, immutable source margins, crossing prefixes, invalid
source/endpoint rejection, VT/EE/PP activation-band separation, translated
compensated inputs, and output canaries.

The two coupled CSV rows shared with the previous native trial are text
identical. This comparison is limited to those exported coupled fields; a
rejected horizon does not export a full accepted FEM endpoint.

The eight-step run takes 312.369 s including preparation and the rejected
attempt. Stage sums are retained separately and must not be promoted to complete
coupled throughput. Newton-16 qualification is a separate subsequent trial.

## Storage

Verified identical completed outputs were hardlinked after independent SHA-256
checks. 9.177 GiB of allocated duplicate storage was recovered without changing
contents. The cleanup receipt lists every path and digest. Sources and unique
evidence remain retained.
