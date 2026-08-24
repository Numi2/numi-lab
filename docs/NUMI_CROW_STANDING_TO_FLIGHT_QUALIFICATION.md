# Numi American-crow standing-to-flight qualification

## Scope and evidence boundary

The `birdflow_american_crow_estimated_hybrid` is an estimated hybrid built
from BirdFlow visual/morphometric anchors. It is not a measured crow surface,
inertial identification, CFD solution, wind-tunnel result, tracked flight
trial, hardware flight result, or a claim about American-crow performance.

All results below are native Metal/MLX simulations on an Apple M4 Pro. They
are useful for regression and model-development decisions only. A README GIF
must not represent this work as measured crow flight until the promotion gate
passes and a replay is visually inspected.

## Current implementation

The stage-2 task is a real articulated simulation, not a recorded trajectory
or an injected body force:

- Wing joints remain position-driven flapping joints. Their submitted targets
  are the live stroke amplitude plus a Metal-resident altitude, vertical-rate,
  and yaw-frame-speed trim.
- The articulated tail pitch is trimmed from the same accepted state, with an
  altitude guard; it is not an external aerodynamic correction.
- Stage-2 learner authority is deliberately bounded around that carrier:
  wing and leg residuals are 0.25 of the normalized action span, tail is
  0.10. Later flight bands retain their authored action space.
- The blade-element force uses the current root and hinge state. The remaining
  `unsteadyCoefficients.y = 0.11875` is an estimated fixed stroke-plane
  direction, not a crow measurement.

The current model exposes flap amplitude and tail pitch, but no independently
articulated wing pronation/stroke-plane degree of freedom. That is the key
remaining control-model boundary: a fixed stroke-plane direction plus tail
trim has not produced sustained forward velocity at the 0.35 m/s stage-2
command.

## Fixed-policy brackets

Every row used four environments, 5,000 control steps, band 2 only,
`--no-scheduled-resets`, seed `20260825`, and a state trace. `Contact` means a
non-foot contact termination; `timeout` is the authored horizon, not a physics
failure.

| Configuration | Mean height (m) | Mean tilt (rad) | World-X final / peak (m) | Boundary result |
| --- | ---: | ---: | ---: | --- |
| 0.11875, strong wing/tail trim, zero policy | 1.004 | 0.114 | -5.665 / 0.011 | 8 contacts |
| 0.11875 plus speed-fed mean flap-angle trim, zero policy | 1.009 | 0.139 | -7.813 / 0.035 | 18 contacts, 1 tilt |
| 0.120 local bracket, zero policy | 1.046 | 0.082 | -20.025 / 0.003 | 4 contacts |
| 0.250 under the complete feedback stack, zero policy | 0.298 | 0.469 | 4.554 / 31.230 | 54 contacts |

The response is non-monotonic around the fixed forward-bias coefficient, so
these values do not justify interpolation or a forward-flight claim. The
0.11875 configuration is retained as the current diagnostic bracket because
the 0.250 configuration is materially less safe, not because it qualifies as
flight.

The existing flap hinge was also tested as a bounded speed-fed mean-angle
carrier. It remains an actual ABA position target, but it shares the flap axis
and did not provide independent stroke-plane/pronation authority; the result
increased contact and tilt failures. It is not retained in the default model.

## Training and promotion gate

The most recent bounded-residual run used 128 environments, 128 steps/update,
64 updates, 1,048,576 samples, fixed learning rate `1e-4`, initial log standard
deviation `-2`, and matched 64-environment held-out evaluation. Training
completed with zero failed environment steps and 36,075.8 ms GPU time.

The final held-out candidate was rejected:

| Metric | Incumbent | Candidate |
| --- | ---: | ---: |
| Mean tracking score | 0.5055 | 0.5292 |
| Non-timeout physical failures/environment | 2.0 | 3.0 |
| Mean root height (m) | 1.0327 | 0.5548 |
| Mean tilt (rad) | 0.0918 | 0.2288 |
| World-X final progress (m) | -9.083 | 1.437 |

The candidate missed the authored 0.70 tracking floor, added physical-boundary
terminations, and increased tilt. The protected deployment pack therefore
remained byte-identical to the immutable incumbent
`c789e9d32760796f36d9fecb205691e5e4b10db4c2b44bc67a6b4999dbe0b6b6`.

## Required next evidence

Before publishing a Numi crow flight GIF, add and qualify an independently
articulated wing-pronation/stroke-plane control; repeat fixed-policy response
tests; train from a clean run directory; and promote only a held-out candidate
that reaches tracking >= 0.70 with zero non-timeout physical-boundary failures.
Then capture a deterministic replay and inspect its frames before linking it
from the compact README showcase.
