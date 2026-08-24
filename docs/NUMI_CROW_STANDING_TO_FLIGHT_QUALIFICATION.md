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

- Each wing is now an explicit three-joint ABA chain: root-connected fore-aft
  sweep, a mass-light flap link, then a mirrored span-axis pronation joint on
  the distal lifting body. The Metal blade-element kernel composes all three
  accepted coordinates and their rates; it does not use either sweep or
  pronation as a force-direction parameter.
- Flap targets remain the live stroke amplitude plus a Metal-resident
  altitude, vertical-rate, and yaw-frame-speed trim. Pronation is an
  independently bounded joint-position target. Its current stage-2 baseline
  is a zero-mean, filter-calibrated wingbeat carrier at normalized amplitude
  0.20 and phase 2.62 rad, with an additive 0.25 policy residual.
- The articulated tail pitch is trimmed from the same accepted state, with an
  altitude guard; it is not an external aerodynamic correction.
- Stage-2 learner authority is deliberately bounded around that carrier:
  flap, sweep, pronation, and leg residuals are 0.25 of the normalized action
  span; tail is 0.10. Later flight bands retain their authored action space.
- The blade-element force uses the current root and hinge state. The remaining
  `unsteadyCoefficients.y = 0.11875` is an estimated fixed stroke-plane
  direction, not a crow measurement.

The prior two-joint model had no independently articulated fore-aft sweep.
That boundary is removed structurally, but the new estimated three-link model
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

## Wing-speed response identification (rejected variants)

Commit `08d297d` added a deterministic host-side bilateral wing-pulse probe.
It changes only the normal policy action lanes before they enter the live
articulated-wing Metal controller; it injects no body force and is retained as
a diagnostic, not as a flight policy. At the held-out seed, band 2, one
environment, 5,000 steps, and no scheduled resets, the zero-action baseline
ended at +23.84 m with tracking 0.4995, mean height 1.0466 m, and mean/max
tilt 0.0617 / 0.1206 rad. A late raw -1.0 pulse (steps 3,000--4,000) did alter
the speed trace but reached 0.7246 rad maximum tilt and ended at -7.83 m;
short early raw -0.5 pulses also reached about 0.72 rad tilt. These are not
bounded tracking responses suitable for a training target.

Two closed-loop wing-speed variants were separately tested for 64
environments, 5,000 steps, with the same held-out seed and no scheduled
resets. Halving the existing speed feedback was physically clean but reduced
mean height to 0.8758 m and tracking to 0.5002. A height-gated correction
completed without non-timeout failures, but reduced mean height to 0.8300 m
and tracking to 0.4997 (mean/max tilt 0.0625 / 0.1189 rad). Both changes were
reverted. The retained carrier is therefore the already-qualified
`-0.100 * forwardSpeedError` term, while the next probe may only promote a
response that improves held-out tracking without an attitude regression.

The follow-up fixed-seed pulse sweep confirms why another unconstrained PPO
run is not yet justified. Early raw wing residuals -0.10, -0.20, -0.35, and
-0.425 (steps 1,000--2,000) were all physically clean, with tracking only
0.49946--0.49967; the -0.425 trace increased reverse speed during its window
but returned to the original late +1.14 m/s acceleration. A positive +1.0
residual held from steps 3,000--5,000 was also clean (mean/max tilt 0.0604 /
0.1206 rad) but raised mean height to 1.288 m and reached only 0.49979
tracking. This brackets the existing action carrier as an altitude authority,
not a proven speed regulator.

## Flight-control action-space partition (rejected)

The action-space trial froze only crow action lanes 5--10 (legs) after
stage-2 lift-off; wing lanes 0--1, pronation lanes 2--3, and tail lane 4 kept
their existing bounded residual authority. This was an action-space partition,
not an aerodynamic, reward, termination, or body-force change. It did not
establish that leg exploration caused the prior failure.

The fresh post-partition baseline ran at revision `87bef2f` on Apple M4 Pro:
64 environments, 5,000 steps, band 2 only, seed `2650443581`, and no scheduled
resets. All 64 episodes timed out normally with zero failed physics steps and
zero height/tilt terminations. Mean tracking was 0.49955, mean root height was
1.04621 m, and mean/max tilt was 0.06176 / 0.12276 rad. The zero action result
therefore preserves the previously qualified mechanical carrier; it is a gate
for the constrained-control learner, not a flight promotion.

Run `crow-flight-controls-20260824-v1/train-128x128x256` then trained the
partitioned action space from a zero-output actor at revision `9e0ce56` with
the same 128 environments, 128 steps/update, 256 updates, fixed `1e-4`
learning rate, and initial log standard deviation `-2`. It completed 4,194,304
samples with zero failed environment steps in 199.383 s (147,323.5 ms measured
GPU time; 21,036.4 end-to-end environment steps/s). Each checkpoint and final
candidate was evaluated at the same 64-environment, 5,000-step held-out seed.

| Held-out policy | Tracking | Mean height (m) | Mean tilt (rad) | Physical failures/environment | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Protected incumbent | 0.49955 | 1.04621 | 0.06176 | 0 | retained |
| Revision 65 (best tracking) | 0.51991 | 0.55898 | 0.28704 | 1.98438 | reject: failures, height, and tilt |
| Revision 129 | 0.51284 | 0.54814 | 0.46079 | 0 | reject: tracking, height, and tilt |
| Revision 193 | 0.49641 | 1.33017 | 0.28715 | 9.53125 | reject: failures and tilt |
| Final revision 257 | 0.49801 | 1.22218 | 0.30364 | 10.79688 | reject: failures and tilt |

The selector retained the immutable incumbent. Since the partition does not
meet the held-out flight gate and its causal benefit was not demonstrated, it
was reverted; the next test starts from the qualified all-lane residual
carrier rather than stacking unvalidated restrictions.

## Tracking audit and stroke-plane bracket (rejected)

The published `tracking` outcome is deliberately composite, not world-axis
displacement: the device computes `0.5 * (linearVelocityTracking +
yawVelocityTracking)`. A one-environment, 5,000-step, no-reset state trace
at the protected seed was independently finite-differenced in the root yaw
frame. It estimates mean forward speed of -1.016 m/s against the +0.35 m/s
command. The trace-derived mean linear score was about 0.0108 while its yaw
score was about 0.9947. These reconstructed values are a diagnostic only--the
native evaluator remains authoritative--but they explain why the approximately
0.50 aggregate cannot be promoted by world-X displacement or yaw changes.

The current 0.11875 fixed stroke-plane tilt is an explicit estimated-hybrid
closure: the flapping kernel resolves its unsteady load in the direction
`airframeUp + tilt * airframeForward`. It is not a measured crow incidence.
To test the nearest bounded alternative, `0.125` was run at source revision
`b601de7` with zero actions, 64 environments, 5,000 steps, band 2 only,
seed `2650443581`, and scheduled resets disabled. All 64 episodes timed out
normally with zero failed steps and zero non-timeout physical-boundary
failures. Mean tracking was only 0.50099, mean root height 1.00231 m, and
mean / maximum tilt 0.07381 / 0.13527 rad: stable but not a material tracking
improvement over the 0.11875 incumbent.

The midpoint `0.13125` was then tested as a one-environment pilot under the
same fixed conditions. It produced two non-timeout contact resets, mean root
height 0.46872 m, and mean / maximum tilt 0.26618 / 0.78026 rad. The bracket
is therefore closed and the source restores 0.11875. Neither closure is a
flight calibration or supports training or README media.

## Wing-authority response ladder (rejected feedback variants)

The post-selection system-identification sweep used the restored qualified
carrier, one environment, 5,000 steps, band 2 only, held-out seed
`2650443581`, no scheduled resets, and a state trace for every row. The
bilateral action enters the live flapping-position drives as the existing
0.25-scaled policy residual; it is not an aerodynamic coefficient or body
force. All four symmetric rows completed normally with zero failed environment
steps and zero non-timeout physical-boundary failures.

| Full-horizon bilateral raw residual | Tracking | Mean height (m) | Mean / max tilt (rad) | World-X final (m) |
| ---: | ---: | ---: | ---: | ---: |
| +0.25 | 0.49868 | 1.20470 | 0.06128 / 0.11916 | 22.265 |
| -0.25 | 0.50058 | 0.88857 | 0.06224 / 0.12262 | 24.933 |
| -0.50 | 0.50174 | 0.72817 | 0.06403 / 0.12393 | 27.691 |
| -0.75 | 0.50260 | 0.56542 | 0.06679 / 0.12432 | 32.542 |

The monotone direction is real, but it trades root height for small tracking
movement and never approaches the 0.70 gate. It therefore does not justify
turning a fixed residual into a flight controller.

Commit `da82269` extended the diagnostic so a pulse can select either physical
wing separately. That extension changes host qualification actions only; it
does not alter the solver, robot, or policy action contract. Persistent
left-only `+0.05` and `+0.25` residuals caused 7 and 12 non-foot-contact
terminations respectively (maximum tilt 0.736 and 0.945 rad). In contrast,
a 100-step `+/-0.05` left-wing pulse at step 1,000 completed cleanly, but both
traces returned to the unregulated trajectory (tracking 0.49954 and 0.49949).
Thus differential wing authority exists but no persistent, safe heading trim
has been identified.

The separate wing-sign feedback test kept the tail-speed term and every other
controller term unchanged. A positive wing speed gain of `+0.10` completed
the one-environment horizon cleanly but reduced mean root height to 0.37856 m
with tracking 0.50161. Reducing that gain to `+0.025` still reduced mean
height to 0.62087 m and reached only 0.50077 tracking. Neither pilot merits a
population rollout or PPO run, so both variants were reverted.

## Wingbeat-frequency feedback (rejected)

Frequency is a real flapping-position control direction, so it was tested as
a bounded controller on the same live 4.6 Hz wingbeat clock rather than as an
added aerodynamic force. All probes used one environment, 5,000 steps, band
2, held-out seed `2650443581`, and no scheduled resets. Lengthening the period
on overspeed at a 12% bound was physically clean but produced a high, backward
trajectory (tracking 0.49943, mean height 1.27287 m, final X -9.85 m).
Reversing that sign was also clean but accelerated to final X +47.81 m with
tracking 0.49897 and mean height 0.81204 m. Reducing the first sign to a 2%
gain remained clean but reached only 0.49936 tracking (mean height 1.09570 m,
final X +19.70 m). Neither direction improves the held-out objective, so all
frequency-feedback variants were reverted. These are single-environment
response probes, not population-level qualification.

## Speed-focused stage-2 reward (rejected)

Commit `4bf3f41` tested one causal objective change only: in the crow's
stage-2 band it removed the signed root-height-progress weight and raised the
existing linear-velocity-tracking weight from 2 to 5. The articulated model,
action interface, terminal conditions, command, held-out evaluator, and all
other reward terms were unchanged. This was intended to prevent a learner from
trading commanded speed for a ballistic climb; it was not a physical-model or
force change.

The fresh zero-action reference at that revision ran on the Apple M4 Pro with
64 environments, 5,000 steps, band 2 only, seed `2650443581`, and scheduled
resets disabled. All 64 episodes timed out normally: zero failed environment
steps and zero non-timeout physical-boundary failures. Mean tracking was
0.4995507, mean root height 1.0462137 m, and mean / maximum tilt
0.0617644 / 0.1227599 rad. Its altered mean reward (0.0558224) is not
comparable to previous rewards because the reward weights changed; the
physical baseline did not.

Run `crow-speed-reward-20260824-v1/train-128x128x256` then started from the
same zero-output actor and trained 128 environments for 128 steps/update over
256 updates, chunk 8, fixed learning rate `1e-4`, initial log standard
deviation `-2`, and checkpoint interval 64. The 4,194,304 training samples
completed with zero failed environment steps. The immutable evaluator scored
the incumbent and revisions 65, 129, 193, and 257 with the same 64-environment,
5,000-step, no-scheduled-reset held-out rollout and seed.

| Held-out policy | Tracking | Mean height (m) | Mean tilt (rad) | Physical failures/environment | World-X final (m) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Protected incumbent | 0.49955 | 1.04621 | 0.06176 | 0 | 23.709 | retained |
| Revision 65 | 0.51106 | 0.67049 | 0.08538 | 0 | 44.719 | reject: below gate and tilt regression |
| Revision 129 | 0.59346 | 0.40546 | 0.22865 | 4.00000 | 10.137 | reject: failures and below gate |
| Revision 193 | 0.60883 | 0.39079 | 0.22054 | 5.10938 | 6.331 | reject: failures and below gate |
| Final revision 257 | 0.61144 | 0.53379 | 0.21828 | 5.18750 | 5.608 | reject: failures and below gate |

The selector retained the immutable incumbent and did not advance deployment.
Although tracking improved on the held-out rollout, no checkpoint reached the
predeclared 0.70 threshold; the later checkpoints also introduced physical
boundary terminations and an attitude regression. The stage-2 reward change
was therefore reverted. It supplies no replay or README GIF evidence.

## Extended phase-aware stage-2 learning (rejected)

The unchanged qualified articulated carrier at source revision `1ad06cf` was
given a deliberately longer fixed-budget test before further model changes:
run `crow-phase-aware-stage2-20260824-v1/train-128x128x1024`. It started from
the zero-output actor, used only curriculum band 2, 128 environments, 128
control steps/update, 1,024 updates, chunk 8, fixed learning rate `1e-4`,
initial log standard deviation `-2`, and a checkpoint interval of 128
updates. Native Metal/MLX execution on Apple M4 Pro completed 16,777,216
training samples in 749.342 s, with zero failed environment steps.

The immutable selector evaluated the protected incumbent, every checkpoint,
and the final candidate with 64 environments, 5,000 steps, scheduled resets
disabled, and held-out seed `2650443581`. It retained the incumbent and did
not advance a deployment pack. `World-X` below is a diagnostic position
quantity, not the flight criterion: the native `tracking` score remains the
yaw-frame velocity and yaw-rate measure described above.

| Held-out policy | Tracking | Mean height (m) | Mean tilt (rad) | Physical failures/environment | World-X final (m) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Protected incumbent | 0.49955 | 1.04621 | 0.06176 | 0 | 23.709 | retained |
| Revision 129 | 0.50067 | 1.15568 | 0.19909 | 6.09375 | -10.483 | reject: failures and tilt |
| Revision 257 | 0.50304 | 1.01702 | 0.41014 | 0 | 125.339 | reject: tracking and tilt |
| Revision 385 | 0.49762 | 1.41679 | 0.21658 | 6.64063 | -12.787 | reject: failures and tilt |
| Revision 513 | 0.49794 | 1.66072 | 0.19425 | 4.46875 | -7.687 | reject: failures and tilt |
| Revision 641 | 0.49381 | 1.62733 | 0.22702 | 7.01563 | -15.358 | reject: failures and tilt |
| Revision 769 | 0.50232 | 2.13823 | 0.30065 | 0 | 112.241 | reject: tracking, height, and tilt |
| Revision 897 | 0.50614 | 2.29948 | 0.07804 | 0 | 18.905 | reject: tracking and height |
| Final revision 1,025 | 0.50090 | 2.31089 | 0.08088 | 0.04688 | 11.026 | reject: failure, tracking, and height |

This test is a negative result, not a flight claim. The longest run did expose
policies that produce large raw `World-X` values, but none supplied the
required yaw-frame speed tracking; most also produced a large altitude or
attitude regression, and several crossed a physical boundary. The predeclared
`tracking >= 0.70` plus zero non-timeout physical-boundary failure gate
therefore blocks promotion. Its candidate/deployment packs, state traces, and
selector JSON are retained with the remote run for reproducibility, but no
candidate receives a deterministic replay, GIF, or README showcase entry.

## Articulated tail and wing--tail pulse response (rejected)

Commit `8cf1af9` adds a host-only secondary pulse lane so two disjoint,
already-compiled action groups can be perturbed over one interval. It does not
change the Metal carrier, ABA mechanics, reward, policy ABI, or aerodynamics.
This permits a causal interaction test rather than inferring a coordinated
controller from PPO telemetry.

The first tail step-response set used the qualified source, Apple M4 Pro, 16
environments, 5,000 control steps, band 2 only, one-step submissions, no
scheduled resets, and seed `2650443581`. The two pulse conditions used the
ordinary tail residual lane only from steps 1,000--1,599. Its normalized
amplitude is still bounded by the live 0.10 tail-residual scale. All rows had
zero failed environment steps and zero physical-boundary terminations.

| Condition | Tracking | Mean height (m) | Mean tilt (rad) | World-X final (m) | Mean yaw-frame forward speed during pulse (m/s) | Accepted tail position during pulse (rad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero residual | 0.499551 | 1.046273 | 0.061750 | 23.689 | -1.077856 | 0.066167 |
| Tail `+1.0` | 0.499514 | 1.044719 | 0.062396 | 24.843 | -1.029897 | 0.111258 |
| Tail `-1.0` | 0.499595 | 1.047691 | 0.061098 | 22.297 | -1.153109 | 0.021067 |

The state traces share the exact pre-pulse trajectory. Raising the physically
accepted tail coordinate by about 0.045 rad improves the local yaw-frame
forward response by 0.048 m/s; lowering it worsens it by 0.075 m/s. The effect
returns after the pulse and does not materially improve the held-out aggregate
tracking score, so a tail-only feedback change is not authorized.

The secondary lane then tested a fixed `-0.5` bilateral-wing residual with
and without a simultaneous `+1.0` tail residual over that same interval and
protocol. The wing residual acts through the existing flapping position drives;
the tail remains an articulated pitch joint. These are action response probes,
not learned policies.

| Condition | Tracking | Mean height (m) | Mean / maximum tilt (rad) | World-X final (m) | Mean yaw-frame forward speed during pulse (m/s) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wing `-0.5` | 0.501629 | 0.999438 | 0.083008 / 0.731176 | -12.828 | -1.705678 | reject: severe transient attitude excursion |
| Wing `-0.5` + tail `+1.0` | 0.499527 | 1.006819 | 0.062347 / 0.122496 | 23.996 | -1.156404 | reject: stable but no forward-speed improvement |

The paired response proves that the real tail joint can counter the tested
wing-amplitude transient: its recovery trace returns to -1.028 m/s and the
maximum tilt returns to the baseline envelope. It does not prove a viable
forward-flight controller--the coordinated pulse remains slower than the
baseline during actuation. No source flight controller, policy, replay, GIF,
or README media is promoted from this experiment.

## Articulated-wing-sweep import and response (rejected)

Commit `9ef2f4d` adds an explicit shoulder sweep coordinate ahead of the
existing flap and pronation coordinates; `6e20478` fixes the native C++
compilation of that model. The resolved stage-2 crow has 14 robot bodies, 13
joints, 20 generalized coordinates, 19 generalized velocities, 14 actions,
and 81 actor observations; the ground scene produces a 15-body compiled
model. The Apple M4 Pro native program check passed at `6e20478` with zero
failed probe steps before any rollout was accepted.

The kinematic motivation is deliberately narrower than calibration. A
[pigeon joint-kinematics study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12068018/)
describes sweep/fold with out-of-plane shoulder, elbow, and wrist yaw terms,
which supports representing a separate sweep degree of freedom. It does not
identify American-crow sweep range, phase, mass properties, or aerodynamic
coefficients. Those values remain estimated hybrid closures and this section
does not make a species-performance claim.

The fixed-seed protocol used Apple M4 Pro native Metal, band 2 only, 16
environments, 5,000 control steps, one-step submissions, no scheduled resets,
and seed `2650443581`. The artifact root is
`.numi/runs/crow-articulated-sweep-20260824-v1/`; every condition contains the
exact arguments, revision, hashes, JSON evidence, and a 5,001-row environment
0 state trace. The pulse was steps 1,000--1,599. The stage-2 residual limit
maps a normalized sweep action of `+/-1.0` to approximately `+/-0.05` rad;
the table reports the accepted left/right sweep coordinates from that window,
not a requested or prerecorded angle. `Contact` is the compiled termination
reason 3 and is a non-timeout physical-boundary failure.

| Sweep action | Accepted sweep L / R (rad) | Tracking | Mean height (m) | Mean / max tilt (rad) | Mean yaw-frame forward speed during pulse (m/s) | Contact / timeout terminations | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | +0.00004 / -0.00042 | 0.500851 | 1.046268 | 0.062061 / 0.122280 | -1.039950 | 0 / 16 | clean zero baseline |
| +0.25 | +0.01264 / +0.01214 | 0.500525 | 1.027522 | 0.069052 / 0.142192 | -0.610084 | 0 / 16 | clean but lower held-out tracking |
| -0.0625 | -0.00315 / -0.00358 | 0.503426 | 1.026824 | 0.087096 / 0.758473 | -1.325660 | 16 / 0 | reject: contact failure and attitude excursion |
| -0.125 | -0.00647 / -0.00691 | 0.503345 | 1.030624 | 0.078907 / 0.747134 | -2.355280 | 16 / 0 | reject: contact failure and attitude excursion |
| -0.25 | -0.01265 / -0.01310 | 0.501876 | 1.030189 | 0.080753 / 0.787822 | +0.926167 | 16 / 0 | reject: contact failure despite local response |
| +1.0 | +0.05045 / +0.04997 | 0.507821 | 0.868389 | 0.140844 / 0.859601 | +0.600024 | 24 / 0 | reject: contact failure and loss of height |
| -1.0 | -0.04991 / -0.05062 | 0.504148 | 1.002755 | 0.108644 / 0.972284 | +1.447660 | 44 / 0 | reject: contact failure and severe attitude excursion |

The identical pre-pulse trace, accepted-coordinate sign changes, and response
windows establish that the physical sweep drives are live. They do not supply
an eligible flight controller: the sole clean nonzero condition is slightly
worse than the zero baseline, while every tested negative condition terminates
through contact. Higher reward or local speed in an unsafe row does not
override the predeclared tracking and physical-boundary gate. Sweep therefore
remains a bounded residual action only; it does not receive a new carrier,
PPO seed, deployment pack, replay, GIF, or README entry.

## Phase-aware sweep waveform (rejected)

Commit `8b8d4c0` adds a qualification-only host waveform with an explicit
normalized amplitude and phase. It enters only the two already-compiled sweep
position action lanes. It does not edit the device-resident carrier, the ABA
mechanics, the aerodynamic closure, or any PolicyPack. The companion
[pigeon free-flight study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12087764/)
reports flight-stage-dependent coupled sweep behavior, which motivates testing
a phase grid; it neither calibrates this estimated crow nor authorizes a
pigeon waveform as its controller.

The same M4 Pro, 16-environment, 5,000-step, band-2, fixed-seed, no-reset
protocol tested a zero-mean wave using `--birdflow-stroke-amplitude 0`, which
leaves the live stage-2 flap carrier active while holding its policy residual
at zero. The accepted sweep ranges below come from each full environment-0
state trace. Contact reason 3 is a non-timeout physical failure; reason 4 is
the normal authored timeout.

| Sweep-wave amplitude / phase | Accepted sweep L / R range (rad) | Tracking | Mean height (m) | Mean / max tilt (rad) | Mean yaw-frame forward speed during pulse (m/s) | Contact / timeout terminations | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 / 0 | -0.01024..+0.01085 / -0.01054..+0.00974 | 0.501347 | 1.003638 | 0.063251 / 0.124397 | -0.932510 | 0 / 16 | clean, but below flight gate |
| 0.25 / +pi/2 | -0.04939..+0.04046 / -0.05720..+0.02294 | 0.504432 | 0.971187 | 0.121789 / 0.788345 | +3.038712 | 36 / 0 | reject: contact failure and attitude excursion |
| 0.25 / pi (3.14 rad) | -0.01060..+0.01044 / -0.01200..+0.01007 | 0.504071 | 0.989411 | 0.103923 / 0.758107 | +1.596971 | 19 / 0 | reject: contact failure and attitude excursion |
| 0.25 / -pi/2 | -0.01060..+0.01063 / -0.01035..+0.01011 | 0.501651 | 0.969545 | 0.090300 / 0.152706 | -0.928630 | 0 / 16 | clean, but below flight gate |
| 0.50 / -pi/2 | -0.02870..+0.02123 / -0.05607..+0.03960 | 0.514779 | 0.732963 | 0.219484 / 0.837239 | +0.736750 | 52 / 0 | reject: contact failure, height regression, and attitude excursion |

This phase grid is a negative control result, not a sparse policy search. The
two clean quarter-amplitude phases leave tracking at about 0.501, while every
condition with a larger local response crosses the physical-boundary gate.
Phase-aware sweep therefore remains unpromoted. No learner run, candidate
selection, deterministic replay, GIF, or README media is authorized from this
evidence.

## American-crow kinematics source screening

The most relevant source screened here is the Journal of Experimental Biology
[Corvidae escape-flight study](https://journals.biologists.com/jeb/article/214/3/452/33507/Scaling-of-mechanical-power-output-during-burst).
It reports vertical escape takeoffs from three American crows, captured with
three synchronized 250 Hz high-speed cameras and digitized into calibrated 3D
wing and body marker coordinates. That is valuable evidence that a
same-species kinematic record existed, but it is not an importable Numi data
package as published on the reviewed article page.

The downloadable 112-page source dissertation, [The allometry of bird flight
performance](https://scholarworks.umt.edu/etd/960/), was independently
reviewed on 2026-08-24. Its Appendix III is a one-page species-level table of
mean, standard error, and maximum kinematic summaries--including an American
crow row--for travel angle, wing angular velocity, stroke-plane measures,
wing angle of attack, stroke amplitude, body angle/velocity, and tail values.
It is useful literature context, but it contains no per-frame marker series,
camera calibration, specimen linkage, or registered surface sequence. These
aggregates must not be used to fit a purported measured crow trajectory or
promote an estimated controller.

The reported task is constrained vertical burst flight in a chamber, not the
current unassisted standing-to-forward-flight target. The article's accessible
material supplies methods, aggregate values, figures, and a supplementary
movie; it does not supply the numerical marker-coordinate series, camera
calibration, coordinate transform, bilateral time-varying surfaces, mass/COM/
inertia record, or a reuse license for such a package. Figures and video frames
must not be digitized opportunistically into a purported measured crow model,
and these data cannot be merged with the independent estimated BirdFlow visual
surface as though they came from one specimen.

An eventual data request or acquisition must obtain an explicit license plus
the original timebase, camera calibration and world axes, 3D wing/body marker
coordinates or registered surfaces, side labels, toe-off and wingbeat events,
specimen mass properties, and atmosphere/force measurements. It must then
enter the provenance-locked schema as a new hybrid or same-specimen record;
until then, this lead remains literature evidence rather than a training input.

## Three-joint 14-action stage-2 PPO (rejected)

Run `crow-articulated-sweep-20260824-v1/train-stage2-14action-128x128x1024`
trained the current three-joint-per-wing crow on the Apple M4 Pro at source
revision `d1aca2a`. It used band 2 only, 128 environments, 128 control steps
per update, 1,024 updates, chunk 8, fixed learning rate `1e-4`, learner seed
`2650443581`, initial log standard deviation `-2`, and a checkpoint every 128
updates. The initial 14-action actor was exactly zero-output around the
previously qualified live flap, pronation, tail, and speed/height trim carrier.
It completed 16,777,216 samples with zero failed environment steps in 575.526
seconds of native submission time.

The selector then evaluated the immutable initial incumbent, every checkpoint,
and the final candidate with the same 64-environment, 5,000-step,
no-scheduled-reset rollout at held-out seed `2650443581`. The population did
discover large displacement and lift-off outcomes, but it did not control the
authored body-frame 0.35 m/s target: its velocity tracking remained below the
fixed 0.70 floor and its attitude regressed. `World-X` remains diagnostic only;
the native tracking metric and physical failures decide the result.

| Held-out policy | Tracking | Mean height (m) | Mean tilt (rad) | Physical failures/environment | World-X final (m) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Immutable 14-action incumbent | 0.500856 | 1.046220 | 0.062081 | 0 | 9.898 | retained |
| Revision 129 | 0.516157 | 0.682267 | 0.332499 | 0 | 51.178 | reject: below tracking floor and tilt regression |
| Revision 257 | 0.495837 | 1.142578 | 0.279128 | 9.562500 | -11.261 | reject: physical failures, tracking, and tilt |
| Revision 385 | 0.482384 | 1.791199 | 0.331104 | 7.453125 | -13.154 | reject: physical failures, tracking, and tilt |
| Revision 513 | 0.473239 | 2.359165 | 0.578180 | 0 | 278.671 | reject: tracking and tilt |
| Revision 641 | 0.497192 | 2.241771 | 0.728765 | 0 | 112.285 | reject: tracking and tilt |
| Revision 769 | 0.496461 | 2.198720 | 0.789204 | 0 | 155.865 | reject: tracking and tilt |
| Revision 897 | 0.492188 | 2.247146 | 0.837163 | 0 | 322.051 | reject: tracking and tilt |
| Final revision 1,025 | 0.477147 | 2.365752 | 0.650030 | 0 | 148.775 | reject: tracking and tilt |

The selector retained the initial incumbent, set
`candidate_advanced_deployment` to false, and retained all candidate packs,
checkpoints, state traces, arguments, and hashes under the run root. This is a
negative result about the current residual-control objective, not a flight
claim. In particular, high displacement or height does not satisfy yaw-frame
forward-speed tracking. The rejected candidate does not receive a replay,
GIF, or README showcase entry.

## Coordinated sweep--tail control bracket (rejected)

The 14-action response tool was used after the rejected PPO run to test a
specific coordinated physical control direction before changing another
learner. Its primary lane sends bilateral shoulder-sweep targets; its
non-overlapping secondary lane sends the tail-pitch target. No run changed the
Metal carrier, force model, aerodynamic coefficients, policy ABI, or terminal
conditions. The pulse starts at step 1,000; it therefore observes the same
supported takeoff prefix as the 14-action zero baseline before acting through
the live articulated position drives.

At sweep `+0.25` plus tail `+1.0`, a 600-step 16-environment M4 Pro probe was
clean. A kernel-aligned finite difference of the environment-0 trace measured
mean yaw-frame forward speed of `-0.638374` m/s over the pulse, compared with
`-1.044208` m/s for the identical zero trace. This establishes a real local
coupled response, not a force injection. It did not improve aggregate tracking
(`0.500493` versus `0.500851`) and raised maximum tilt to `0.145055` rad.

The persistent 4,000-step bracket used the same 16 environments, 5,000-step
horizon, no scheduled resets, and seed `2650443581`. Every successful row
ended by normal timeout; `Contact` is compiled non-foot-contact termination
reason 3. `World-X` is diagnostic only and must not be read as forward-flight
success.

| Sweep / tail residual | Tracking | Mean height (m) | Mean / max tilt (rad) | Contact / timeout terminations | World-X final (m) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Zero carrier | 0.500851 | 1.046268 | 0.062061 / 0.122280 | 0 / 16 | 23.709 | retained reference |
| +0.25 / +1.0, 600-step pulse | 0.500493 | 1.025456 | 0.069645 / 0.145055 | 0 / 16 | 20.032 | clean local response; no aggregate gain |
| +0.25 / +1.0 persistent | 0.498346 | 0.903664 | 0.113421 / 0.145055 | 0 / 16 | 67.168 | clean but tracking and attitude regress |
| +0.28125 / +1.0 persistent | 0.498078 | 0.872487 | 0.121802 / 0.151056 | 0 / 16 | 71.237 | clean but tracking and attitude regress |
| +0.3125 / +1.0 persistent | 0.511010 | 0.698529 | 0.208260 / 0.829751 | 30 / 0 | 4.477 | reject: physical boundary failures |
| +0.375 / +1.0 persistent | 0.525286 | 0.586371 | 0.273438 / 0.845913 | 53 / 0 | 15.676 | reject: physical boundary failures |
| +0.50 / +1.0 persistent | 0.536301 | 0.546074 | 0.306798 / 0.869224 | 76 / 0 | 6.916 | reject: physical boundary failures |

The constant coordination law is therefore closed. Below the discontinuous
contact boundary, it worsens the held-out tracking measure; above it, it gains
some tracking only by losing the height/attitude envelope and repeatedly
striking the ground. It must not become a PPO initialization, native carrier,
deployment pack, replay, GIF, or README item. Any successor must be a
separately specified state-gated coupled articulation law and must requalify
from a fresh zero-output baseline under the same gate.

## State-gated sweep--tail co-trim and PPO pilot (rejected)

Source revision `c5e804f` adds a deliberately narrow successor hypothesis: in
stage 2 only, the existing articulated shoulder-sweep and tail position targets
receive the previously clean `+0.28125` / `+1.0` coordination magnitude only
while all of the following live state gates have headroom: yaw-frame forward
speed is below 0.35 m/s, root height is below 0.95 m, tilt is below 0.14 rad,
and vertical speed is below 0.40 m/s. It changes neither the articulated ABA
model nor aerodynamic/force parameters, terminal conditions, action ABI, or
the 0.70 selection floor. The new compiled task fingerprint is
`3198934467138572318`; `metalrobo_run_program_check` passed before evaluation.

The fresh zero-output baseline was evaluated on the Apple M4 Pro with 64
environments, 5,000 steps, no scheduled resets, band 2 only, and held-out seed
`2650443581`. All 64 environments reached normal timeout, with zero
non-timeout physical-boundary failures. Mean tracking was `0.501385`, mean
root height `1.041576` m, mean tilt `0.063515` rad, and maximum tilt
`0.133755` rad. The result is a clean incumbent for the changed task program,
not evidence of controlled flight and not a cross-fingerprint promotion over
the previous carrier.

The predeclared 256-update PPO pilot completed 4,194,304 samples in 167.712
seconds of native submission time, with 128 environments x 128 steps, chunk 8,
fixed learning rate `1e-4`, initial log standard deviation `-2`, learner seed
`2650443581`, and checkpoints every 64 updates. The fixed selector compared the
immutable fresh incumbent and every candidate at 64 environments x 5,000 steps,
without scheduled resets, at the same held-out seed. `Physical failures` is the
selector's mean non-timeout physical-boundary termination rate per environment;
`World-X` is diagnostic only.

| Held-out policy | Tracking | Mean height (m) | Mean tilt (rad) | Physical failures/environment | World-X final (m) | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Immutable fresh incumbent | 0.501385 | 1.041576 | 0.063515 | 0 | 12.056 | retained |
| Revision 65 | 0.517294 | 0.629447 | 0.217466 | 2.000000 | -1.595 | reject: physical failures, tracking, and tilt |
| Revision 129 | 0.500475 | 1.207358 | 0.199190 | 6.000000 | -9.304 | reject: physical failures, tracking, and tilt |
| Revision 193 | 0.500925 | 1.153037 | 0.188046 | 5.671875 | -11.132 | reject: physical failures, tracking, and tilt |
| Final revision 257 | 0.500611 | 1.099808 | 0.119961 | 3.000000 | -15.282 | reject: physical failures and tracking |

The comparison champion is the incumbent and no candidate advanced deployment.
The state-gated co-trim is therefore closed, and its default task-program flag
and Metal target contribution are removed after this record. The retained
candidates, selector JSON, per-policy evidence, state traces, arguments, and
hashes remain in the run root. This is a negative result about the estimated
hybrid and residual-control objective, not flight evidence: no candidate earns
a deterministic replay, GIF, picture, or README showcase entry.

The removal was independently checked on the Apple M4 Pro at source revision
`2eea031` using a fresh 64-environment, 5,000-step, band-2, no-reset,
zero-output rollout at seed `2650443581`. All 64 environments reached normal
timeout with zero failed environment steps. It restored the original incumbent
to the shown precision: tracking `0.500856`, mean root height `1.046220` m,
mean tilt `0.062081` rad, and maximum tilt `0.122280` rad. The reversion is
therefore a physical restoration check, not a new flight qualification.

## Required next evidence

The next research stage is provenance and model identification, not another
hand-tuned carrier: the scalar stroke-plane, constant symmetric-sweep,
phase-aware sweep, and state-gated sweep--tail hypotheses are closed. A new
control hypothesis needs provenance-adequate measured crow kinematics or a
separately specified coupled articulation/model change, declared with a fresh
fixed seed and physical gate before execution. Promote only a held-out
candidate that reaches tracking >= 0.70 with zero non-timeout
physical-boundary failures. Then capture a deterministic replay and inspect
its frames before linking it from the compact README showcase. Until then, no
Numi crow flight GIF belongs in the README.
