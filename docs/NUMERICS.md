# Numerical contract

This document defines the numerical behavior that implementation and evidence
must satisfy. Unsupported behavior is recorded in
[CAPABILITIES.md](CAPABILITIES.md), not disguised by fallback logic.

## Coordinates and integration

- All external quantities use SI units.
- Scalar hinge and slide joints use their natural generalized coordinates.
- Ball and free rotations integrate in tangent space and renormalize only as a
  representation operation, not as a constraint substitute.
- Body translation and linear velocity are centre-of-mass quantities.
- Articulation body twists are materialized by a forward-tree recursion from
  the same accepted `q, v` state used by dynamics. Point velocity is
  `v_com + omega x r`; task and sensor code must not infer it by finite
  differencing poses or silently substitute zero body velocity.
- For target frame `T` observed from moving reference frame `R`, relative
  linear velocity is `R_R^T (v_T - v_R - omega_R x (p_T - p_R))` and relative
  angular velocity is `R_R^T (omega_T - omega_R)`. Subtracting world vectors
  without the transport term is not a valid moving-frame derivative.
- For one generalized-velocity coordinate `j`, the world-axis Jacobian at a
  compiled frame origin is
  `J_linear(:,j) = v_j + omega_j x (p_frame - p_body)` and
  `J_angular(:,j) = omega_j`, where `(v_j, omega_j)` is the analytic body
  motion column produced by the articulated operator. TaskIR evaluates this
  from the same accepted `q` used by dynamics. It does not finite-difference
  poses. A coordinate owned by another disconnected articulation contributes
  exactly zero.
- SignalIR exponential tracking is `exp(-((x - target) / width)^2)` with a
  strictly positive compiled width. Safe division clamps only the denominator
  magnitude to an authored positive epsilon while preserving its sign. Bounds
  gates are inclusive. Signal evaluation is a fixed topological sweep per
  environment; there is no data-dependent graph traversal or early exit.
- Temporal microsteps re-evaluate geometry, limit activation, and response
  factors before integration. The command-delay/backlash timeline advances
  once at the control boundary; re-evaluating motor envelope and passive
  friction at every physics microstep remains an open mechanics gate.
- Velocity or effort safety caps are explicit actuator/safety-envelope stages;
  they are not positional constraints.

Native actuator delay uses `ceil(delay / control_period)` whole control steps.
Backlash is a deterministic play operator around the delayed command. Raw motor
effort is limited by the minimum of the authored generalized effort limit and
the torque-speed envelope; passive dry friction is then applied separately so
the sensor contract can distinguish commanded, motor, passive, and generalized
effort. Reset seeds the command timeline from the neutral joint target for
position control, or zero for effort control.

## Task goal time and orientation

Task goals are immutable compiled records. Fixed goals read their authored
pose directly. Episode-sampled goals derive six independent uniform values
from a counter key containing the session seed, environment, episode, stable
goal identity, channel, and goal-purpose domain. Translation offsets are in
world axes. Rotation offsets are principal rotation vectors with norm no larger
than pi and are applied in the base goal's local frame.

Two-pose trajectories use accepted task time
`episode_step * control_period + phase`. Position interpolation is linear;
orientation interpolation is normalized shortest-arc quaternion slerp. Clamp,
loop, and ping-pong playback share these equations. Physics rejection does not
advance accepted episode time, and reset increments the episode key, so no
parallel mutable goal state or rollback copy is allowed. Multi-knot splines are
not yet part of the production goal operator.

## NumiSolver

`NumiSolver` is the only public constrained-solver identity. Internal topology
selection is not a user-visible solver mode:

- compact islands use reduced-primal semismooth Newton and deterministic block
  factorization;
- large or rod-rich islands use matrix-free semismooth Newton with
  preconditioned CG;
- training uses fixed iteration budgets;
- tighter quality settings use the same equations and operators with larger
  budgets and stricter tolerances.

Contacts, scalar/cone limits, equality constraints, tendons, gears,
attachments, rolling/torsional friction, and rod endpoints participate in one
island graph and one residual. Warm starts are used only when their initial
merit is better than the zero-impulse state.

The implementation must not expose PGS, TGS, Wave, or a separate quality
solver as interchangeable product choices. Temporal microsteps alone do not
constitute a TGS implementation.

## ConstraintIR

Each block has variable `rowOffset/rowCount` and
`endpointOffset/endpointCount`. Common widths 1, 3, and 6 may use specialized
kernels, with a bounded generic path for other supported widths.

Scalar limits compile as stable paired candidates:

- lower gap `q - lower`, Jacobian sign `+1`, impulse `[0,+inf)`;
- upper gap `upper - q`, Jacobian sign `-1`, impulse `[0,+inf)`.

Activation uses current or predicted gap, configured slop, and bounded recovery
velocity. Lower and upper warm impulses are independent and decay when
inactive. A post-integration verifier may record hypothetical correction but
must never mutate production state.

## Rod operator

A connected rod uses tangent coordinates `[translation(3), twist]` and a
compiler-derived block-banded operator

```text
A = M + h D + h^2 K.
```

Production uses a positive-semidefinite Gauss–Newton material tangent. The FP64
oracle may use an exact Newton tangent. The retained factor is built once per
accepted microstep and reused for free motion, constraint response, and
adjoints.

Rod contact response is global. All endpoint impulses are accumulated into one
rod RHS, the band system is solved, and velocities are projected back through
every endpoint Jacobian. Local inverse-node-mass contact approximations are not
permitted in NumiSolver.

## Contact and CCD

- Solver impulses are the force and wrench authority for physical sensors.
- Static/dynamic anisotropic friction uses a declared cone or bounded block.
- Restitution consumes pre-impact normal velocity only for a newly detected
  impact.
- Speculative contact is named speculative contact.
- A path is called CCD only when it advances to impact, solves, and continues
  the remaining step for its declared pair types.

## Residuals and acceptance

Every constrained solve reports normalized primal, dual, cone, and
complementarity residuals plus regularization and iteration counts. Fixed
training budgets do not suppress residual publication.

Release constraint gates are:

- normalized complementarity residual at or below `1e-4`;
- scalar limit violation at or below configured slop plus `1e-5` in joint
  units;
- no debug post-solve correction above numerical round-off;
- active side, impulse sign, and physical outcome agree with the FP64 oracle
  and matched MuJoCo cases within scenario tolerances.

## Differentiation

The production forward solver remains authoritative. Reverse execution does
not unroll iterative solver history. For a converged, stable active set it
linearizes the constraint system and solves the transposed system using the
same compact, banded, or matrix-free response operators.

Forward tapes retain accepted substeps, checkpoints, active stable keys,
contact features, friction regimes, actuator/sensor intermediates, factor
handles, and residuals within a declared bytes-per-environment budget.

Gradients are invalid when relevant discrete behavior changes, including
contact topology, collision feature, stick/slip regime, CCD event, reset,
termination, sleep/wake state, excessive regularization, or unconverged
forward solve. Validity reason bits are part of the result; invalid gradients
are never returned as plausible numbers.

Acceptance is relative error below `1e-3` for smooth contact-free cases and
below `1e-2` for stable fixed-active-set constrained cases.
