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

- Each wing is now an explicit two-joint ABA chain: root-connected flap
  shoulder, then a mirrored span-axis pronation joint on the distal lifting
  body. The Metal blade-element kernel composes both accepted coordinates and
  their rates; it does not use pronation as a force-direction parameter.
- Flap targets remain the live stroke amplitude plus a Metal-resident
  altitude, vertical-rate, and yaw-frame-speed trim. Pronation is an
  independently bounded joint-position target. Its current stage-2 baseline
  is a zero-mean, filter-calibrated wingbeat carrier at normalized amplitude
  0.20 and phase 2.62 rad, with an additive 0.25 policy residual.
- The articulated tail pitch is trimmed from the same accepted state, with an
  altitude guard; it is not an external aerodynamic correction.
- Stage-2 learner authority is deliberately bounded around that carrier:
  flap, pronation, and leg residuals are 0.25 of the normalized action span;
  tail is 0.10. Later flight bands retain their authored action space.
- The blade-element force uses the current root and hinge state. The remaining
  `unsteadyCoefficients.y = 0.11875` is an estimated fixed stroke-plane
  direction, not a crow measurement.

The prior one-hinge model had no independently articulated feathering control.
That boundary is removed structurally, but the new estimated two-link model
has not yet produced sustained forward velocity at the 0.35 m/s stage-2
command. Its angle limits, connector inertia, and drive constants are explicit
hybrid-model closures, not crow measurements.

## Articulated-pronation response

The remote Apple M4 Pro response sweep used the real compiled 12-action crow
program, 16 environments, 256 control steps, stage 2 only, the deterministic
`--birdflow-stroke-amplitude 0` carrier, and symmetric static pronation action.
All 16 environments completed without a termination or physics error for every
row. The action is mapped through the two physical position drives; it is not
a recorded wing angle.

| Symmetric pronation action | Mean height (m) | Mean / max tilt (rad) | Final / peak forward progress (m) | Result |
| ---: | ---: | ---: | ---: | --- |
| -1.0 | 0.7603 | 0.01325 / 0.07817 | 0.000 / 0.000 | clean, no forward response |
| -0.5 | 0.7599 | 0.01255 / 0.07213 | 0.000 / 0.000 | clean, no forward response |
| 0.0 | 0.7597 | 0.01185 / 0.06531 | 0.000 / 0.000 | clean baseline |
| +0.5 | 0.7599 | 0.01120 / 0.05860 | 0.000 / 0.000 | clean, lower tilt |
| +1.0 | 0.7604 | 0.01062 / 0.05342 | 0.000 / 0.000 | clean, lower tilt |

This is an actuator-response result, not a flight result. Positive symmetric
pronation lowers tilt over the short fixed probe, but does not establish a
forward trim or justify seeding a fixed pronation carrier. Training therefore
starts from zero pronation with bounded residual authority.

That short response did not persist over the selected task horizon. A second
remote M4 Pro probe held `--birdflow-pronation 1` at stage 2 for 64
environments and 5,000 control steps with scheduled resets disabled. It had
zero failed environment steps and zero non-timeout physical failures, but all
64 environments reached the authored timeout with mean height 0.616 m, mean /
maximum tilt 0.235 / 0.475 rad, tracking 0.503, and zero reported forward
progress. This rejects widening the stage-2 pronation residual from 0.25 to
full authority: a phase-independent feathering offset is not a forward-flight
trim.

An in-phase, zero-mean sinusoidal pronation carrier was also compiled directly
into the two ABA position targets at the same +/-0.075-rad range. Its 16 by
256 stage-2 smoke probe was clean and lowered mean tilt to 0.01113 rad, but a
64-environment, 5,000-step, no-scheduled-reset probe produced 132 non-foot
contact terminations, mean / maximum tilt 0.247 / 0.786 rad, mean height
0.504 m, tracking 0.520, and zero forward progress. No physics-error steps
occurred, but the physical outcome rejects this phase convention. The carrier
was removed before any policy training; a short stable horizon is not enough
to qualify a wingbeat controller.

To identify a replacement without hard-coding another force direction, a
qualification-only host action sweep sent sinusoidal normalized commands
through the same live pronation position drives. The in-phase convention
remained unsafe (23 contact terminations, mean tilt 0.313 rad). The `+pi/2`,
`-pi/2`, and `pi` conditions had zero non-timeout failures but were stationary;
their mean tilt was 0.0652, 0.0649, and 0.0616 rad respectively. The host
action filters a 4.6 Hz command with a 20 ms first-order response. The retained
device carrier therefore uses its equivalent compensated phase 2.62 rad and a
slightly smaller normalized amplitude 0.20. Its direct device-resident
64-environment, 5,000-step probe produced zero physics errors and zero
non-timeout physical failures; all environments reached timeout with mean
height 1.046 m, mean / maximum tilt 0.0618 / 0.1230 rad, tracking 0.4996, and
zero forward progress. This is a safe training baseline only, not a flight
qualification or a claim of a biologically measured feathering phase.

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

The first clean training run after the articulated-pronation import was
`crow-pronation-20260824-v1` at source revision `d8bf334`. It used 128
environments, 128 steps/update, 64 updates, 1,048,576 samples, fixed learning
rate `1e-4`, initial log standard deviation `-2`, and checkpointed revisions
17, 33, 49, and 65. Native Metal/MLX training on the Apple M4 Pro completed
with zero failed environment steps in 64.065 seconds (50,878.4 ms measured GPU
time; 16,367 end-to-end environment steps/s). The final learner batch's 0.941
tracking is an optimization statistic, not a held-out flight result.

The selector evaluated every checkpoint, the final candidate, and the
immutable pre-training incumbent on the same 64-environment, 5,000-step,
no-scheduled-reset rollout with held-out seed `2650443581`. The final candidate
was rejected:

| Metric | Incumbent | Final candidate |
| --- | ---: | ---: |
| Mean tracking score | 0.8360 | 0.8248 |
| Non-timeout physical failures/environment | 0.171875 (11 / 64) | 0.171875 (11 / 64) |
| Mean / maximum tilt (rad) | 0.01877 / 0.12785 | 0.04415 / 0.56926 |
| Mean root height (m) | 0.4095 | 0.4379 |
| World-X final / peak progress (m) | -22.637 / 0.199 | -29.852 / 0.090 |
| Failed environment steps | 0 | 0 |

No checkpoint cleared the zero-physical-boundary-failure gate. The protected
deployment SHA-256 is therefore the incumbent's
`3cb1f9bd88cd5c1538a921b7e1a9a0c7b13aa9d52f513d05203e172ee58215d8`, not the
retained candidate's
`09240a50d578eb9226a313ad2169af3b532aae72f288ff466e2f86e9554cfb99`.

## Aligned-objective extended trial (rejected)

The current velocity gate reports `exp(-squared speed error / 0.25)`, while
the inherited dove training reward used the looser 0.35 width. Commit
`4ac0bcb` changes the crow copy only to the same 0.25 width and adds a native
program-check assertion. It does not lower the 0.70 gate, alter the measured
state, inject a force, or change a termination condition.

Before training, two controller-sign ablations were run on the Apple M4 Pro
at held-out seed `2650443581`, band 2 only, 64 environments, 5,000 steps, and
no scheduled resets. Reversing both wing and tail speed terms retained zero
physics errors but collapsed mean height to 0.398 m, so it was rejected. The
tail-only reversal remained physically clean (mean height 1.029 m, mean/max
tilt 0.0689 / 0.1229 rad) but lowered tracking to 0.4990 and its deterministic
trace still accelerated from reverse motion to +1.30 m/s. That variant was
also rejected; the published live carrier stays unchanged.

Run `crow-pronation-aligned-20260824-v1` trained the aligned objective from a
zero-output actor at source revision `4ac0bcb` with 128 environments, 128
steps/update, 256 updates, fixed learning rate `1e-4`, initial log standard
deviation `-2`, and checkpoint interval 32. It completed 4,194,304 samples
with zero failed environment steps in 192.091 s (139,998.2 ms measured GPU
time; 21,835 end-to-end environment steps/s). Every checkpoint and final
candidate was evaluated on the unchanged held-out 64-environment, 5,000-step,
no-scheduled-reset rollout at seed `2650443581`.

| Held-out policy | Tracking | Mean height (m) | Mean tilt (rad) | Physical failures/environment | World-X final (m) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Protected incumbent | 0.4996 | 1.0462 | 0.0618 | 0 | 23.709 | retained |
| Revision 33 | 0.4996 | 1.0177 | 0.0621 | 0 | 33.105 | reject: tracking below gate |
| Revision 65 (best tracking) | 0.5497 | 0.4997 | 0.3049 | 2.59375 | 18.136 | reject: failures and tracking below gate |
| Final revision 257 | 0.5030 | 1.0170 | 0.4101 | 0 | 125.339 | reject: tracking below gate and tilt regression |

The selector chose the immutable incumbent and did not advance deployment.
The aligned loss therefore remains a source-level experiment, not evidence of
successful crow flight. Its result is especially clear: higher displacement
alone is not commanded-speed tracking. No replay or README GIF is authorized
from this run.

## Required next evidence

The next control experiment must first identify a bounded, long-horizon
speed-control response around the retained safe feathering carrier. It must
distinguish commanded-speed regulation from unbounded forward displacement,
then be tested before another long learner run. Promote only a held-out
candidate that reaches tracking >= 0.70 with zero non-timeout
physical-boundary failures. Then capture a deterministic replay and inspect
its frames before linking it from the compact README showcase. Until then, no
Numi crow flight GIF belongs in the README.
