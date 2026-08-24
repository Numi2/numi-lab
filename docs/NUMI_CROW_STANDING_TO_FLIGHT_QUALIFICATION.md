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
  span; tail is 0.10. The narrower controller candidate below was rejected by
  its pre-registered full-envelope stress guard. Later flight bands retain
  their authored action space.
- The blade-element force uses the current root and hinge state. The remaining
  `unsteadyCoefficients.y = 0.11875` is an estimated fixed stroke-plane
  direction, not a crow measurement.

The prior two-joint model had no independently articulated fore-aft sweep.
That boundary is removed structurally, but the new estimated three-link model
has not yet produced sustained forward velocity at the 0.35 m/s stage-2
command. Its angle limits, connector inertia, and drive constants are explicit
hybrid-model closures, not crow measurements.

## All-articulated wing-velocity closure (rejected)

The current crow blade-element kernel resolves the section point velocity as
the sum of flap, sweep, and pronation velocities. Until this experiment, its
unsteady-force fraction, tail-wash proxy, and differential wing-energy
bookkeeping used only the flap component. That mismatch became material after
the sweep and pronation joints were added: a physical sweep or pronation rate
could affect quasi-steady flow but not those three rate-sensitive closures.

The only planned model change is to use the already computed complete
relative-wing velocity `flap + sweep + pronation` for those quantities. It
adds no force term, coefficient, action, reward, controller, termination, or
species calibration. It is a generic kinematic-consistency correction, not a
claim that the estimated crow has measured aerodynamic loads. The motivation
is independently checkable: a blade-element analysis of flapping birds
computes each section's velocity from both flapping and morphing motion, and
the bird-flight review describes strips moving from both bird flight velocity
and wing angular velocity ([Morris et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10942624/);
[Tobalske, 2007](https://journals.biologists.com/jeb/article/210/18/3135/17027/Biomechanics-of-bird-flight)).

Before the source change, the exact `54c2eb8` reference used Apple M4 Pro,
64 environments, 5,000 control steps, band 2 only, zero actions, no scheduled
resets, and held-out seed `2650443581`. It completed with zero failed
environment steps and zero non-timeout physical-boundary failures; all 64
environments timed out normally. The mean tracking score was `0.5008565`,
mean root height `1.0462196 m`, and mean / maximum tilt `0.0620805 / 0.1222802
rad`. Its immutable artifact root is
`.numi/runs/crow-articulated-sweep-20260824-v1/all-articulated-velocity-prechange-zero-64x5000/`.

The corrected source must first pass the native program checker, then repeat
that exact zero-action rollout. It is eligible for one protected
128-environment × 128-step × 256-update PPO run only if that rollout has zero
failed environment steps, zero non-timeout physical-boundary failures, mean
root height in `[0.85, 1.30] m`, mean tilt at most `0.15 rad`, and maximum tilt
at most `0.30 rad`. The immutable 64-environment held-out selector then
compares every checkpoint and final candidate to its matched incumbent. A
candidate needs tracking at least `0.70`, zero non-timeout physical-boundary
failures, and no attitude/height regression before any protected deployment,
deterministic replay, GIF, or README media can advance. Otherwise the model
variant is recorded as rejected and the retained source stays unchanged.

At source revision `2874407`, the M4 Pro rebuilt the Metal library and passed
the native program check. The corrected zero-action rollout retained zero
failed environment steps but violated the physical gate: 240 non-foot contact
terminations occurred across 64 environments (no normal timeouts), the mean
tilt was `0.1505679 rad`, and the maximum tilt was `0.7760603 rad`. Mean root
height was `1.0087967 m`; the superficially higher tracking score of
`0.5065342` cannot outweigh the boundary failure. The complete immutable
artifact root is
`.numi/runs/crow-articulated-sweep-20260824-v1/all-articulated-velocity-zero-baseline-64x5000/`.

The variant receives no PPO run, candidate policy, deployment artifact,
replay, GIF, or README media. The source restores the qualified flap-only
rate closure. This negative result does not establish that flap-only is a
complete physical model; it establishes only that this uncalibrated expansion
is unsafe in the present estimated hybrid and cannot be promoted as a crow
flight improvement.

## Articulated wing-wrench placement (pre-registered)

`MRABABodyWrenchGPU` defines force and torque about the receiving body's
center of mass. The existing blade-element kernel integrates each wing's load
about the airframe root and writes that whole wrench directly to the root.
That preserves the net airframe resultant but bypasses the finite articulated
sweep, flap, and pronation drives--they receive no aerodynamic reaction. The
next experiment changes only the wrench reference/recipient: the compiler
resolves the parent and child anchors of the root-most wing joint plus the
distal wing-body COM; Metal reconstructs its current world position and uses
`tau_COM = tau_root - r_root_to_COM x force` before writing the same force and
resultant moment to the wing body. It adds no aerodynamic coefficient, force,
action, controller, reward, termination, or crow-specific calibration.

This is an articulated-mechanics correction, not a claim of measured bird
aerodynamics. The broader model limitation remains explicit: detailed bird
wing kinematics and unsteady profiles require data beyond a generic
blade-element closure ([Tobalske, 2007](https://journals.biologists.com/jeb/article/210/18/3135/17027/Biomechanics-of-bird-flight)).

The pre-change crow reference is the same exact `54c2eb8` run reported above:
64 environments × 5,000 steps, band 2, zero actions, no scheduled resets,
seed `2650443581`, zero failed environment steps/physical-boundary failures,
tracking `0.5008565`, height `1.0462196 m`, and mean / maximum tilt
`0.0620805 / 0.1222802 rad`. The shared Dove guard used that identical
environment/step/seed protocol with zero actions at source `83f624e`; it had
zero failed environment steps, tracking `0.5655592`, height `1.0373152 m`,
and mean / maximum tilt `0.1361817 / 0.2532357 rad`. Its expected 3,840
reason-1 task transitions are a pre-existing task outcome, not solver errors.
The immutable roots are respectively
`.numi/runs/crow-articulated-sweep-20260824-v1/all-articulated-velocity-prechange-zero-64x5000/`
and
`.numi/runs/birdflow-dove-articulated-load-path-20260824-v1/prechange-zero-64x5000/`.

The candidate must pass native program compilation and both 64×5,000
zero-action regressions. Crow must retain zero failed environment steps and
zero non-timeout physical-boundary failures, mean height in `[0.85, 1.30] m`,
mean tilt at most `0.15 rad`, and maximum tilt at most `0.30 rad`. Dove must
retain zero failed environment steps, no new termination reason, tracking at
least `0.55`, height in `[0.90, 1.20] m`, and maximum tilt at most `0.30 rad`.
Only if both gates pass may the crow receive one protected 128×128×256 PPO
trial followed by the existing immutable 64-environment selector. The
unchanged `tracking >= 0.70` plus zero physical-boundary-failure promotion gate
still governs deployment, replay, GIF, and README eligibility.

### Outcome: baselines pass; protected PPO rejected

At source revision `4ee45ae`, the Apple M4 Pro rebuilt the Metal program and
`metalrobo_run_program_check` passed. The fresh crow zero-action guard used 64
environments, 5,000 steps, band 2 only, no scheduled resets, and seed
`2650443581`. All 64 environments reached normal timeout with zero failed
environment steps and zero non-timeout physical-boundary failures. Mean
tracking was `0.5006364`, mean root height `1.0409775 m`, and mean / maximum
tilt `0.0635701 / 0.1247985 rad`. Its immutable root is
`.numi/runs/crow-articulated-wrench-placement-20260824-v1/zero-baseline-64x5000/`.

The matching Dove guard at the same source revision also passed: zero failed
environment steps; its expected 3,840 reason-1 task transitions and no new
termination reason; tracking `0.5652682`; mean root height `1.0363976 m`; and mean /
maximum tilt `0.1355671 / 0.2533256 rad`. Its immutable root is
`.numi/runs/birdflow-dove-articulated-load-path-20260824-v1/wrench-placement-zero-64x5000/`.
These paired zero-output results qualify the mechanics correction for the one
pre-registered PPO experiment; they do not demonstrate trained crow flight.

That PPO trial used 128 environments × 128 steps × 256 updates (4,194,304
samples), chunk 8, fixed learning rate `1e-4`, initial log standard deviation
`-2`, learner seed `2650443581`, and checkpoints every 64 updates. The native
training submission took `192.807 s`. The immutable selector compared every
checkpoint and the final policy against the fresh 64-environment incumbent for
5,000 no-reset band-2 steps at held-out seed `2650443581`. Physical failures
below are non-timeout physical-boundary terminations per environment.

| Held-out policy | Tracking | Mean height (m) | Mean tilt (rad) | Physical failures/environment | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Immutable fresh incumbent | 0.500636 | 1.040978 | 0.063570 | 0 | retained |
| Revision 65 | 0.546855 | 0.575657 | 0.239204 | 2.015625 | reject: physical failures, tracking, and tilt |
| Revision 129 | 0.500126 | 0.962596 | 0.270468 | 10.031250 | reject: physical failures, tracking, and tilt |
| Revision 193 | 0.499649 | 0.800986 | 0.329394 | 12.812500 | reject: physical failures, tracking, height, and tilt |
| Final revision 257 | 0.490202 | 0.608797 | 0.414522 | 14.640625 | reject: physical failures, tracking, height, and tilt |

The selector retained the incumbent and advanced no candidate deployment. The
load-path correction remains a mechanically motivated, zero-output-qualified
change; this PPO result is negative evidence for the present estimated crow
hybrid and residual-control objective. The retained candidates, selector JSON,
per-policy evidence, state traces, arguments, and hashes remain at
`.numi/runs/crow-articulated-wrench-placement-20260824-v1/train-128x128x256/`.
No candidate earns a deterministic replay, Crow GIF, picture, or README entry.

## Published takeoff wingbeat clock (pre-registered)

The imported visual profile uses a 4.6 Hz presentation wingbeat. The primary
[Jackson and Dial (2011) Corvidae experiment](https://journals.biologists.com/jeb/article/214/3/452/33507/Scaling-of-mechanical-power-output-during-burst)
instead reports `6.4 Hz` for American crow maximal takeoff; its flight chamber
uses synchronized high-speed cameras and a forceplate, and the published
results contain three American crows. The work is directly relevant to the
initial standing-to-lift-off regime, but it is vertical escape flight rather
than the current task's forward reference. It supplies neither a reusable
per-frame trajectory nor a wing-angle phase convention, so this experiment
must not derive a phase target, replay, force, mass/inertia, or aerodynamic
coefficient from it.

The new
[`american-crow-numi-hybrid-v2` provenance record](../assets/birdflow/american-crow-numi-hybrid-v2.json)
therefore changes exactly one compiled parameter: the crow task gait period
from `1 / 4.6 s` to `1 / 6.4 s`. The BirdFlow visual surface lock, all body
dimensions and masses, blade-element and unsteady coefficients, action ABI,
rewards, resets, terminations, carrier amplitudes, and the `tracking >= 0.70`
selection floor remain unchanged. `run_program_check` asserts the new clock;
both Dove and Crow programs must compile, but only the Crow task fingerprint
and rollout need a new physical guard.

The prior reference is the `4ee45ae` articulated-wrench zero-action Crow
baseline: 64 environments × 5,000 steps, band 2, no scheduled resets, seed
`2650443581`, zero failed environment steps, zero non-timeout physical-boundary
failures, tracking `0.5006364`, mean root height `1.0409775 m`, and mean /
maximum tilt `0.0635701 / 0.1247985 rad`. Before training, the 6.4 Hz variant
must pass the same M4 Pro zero-action protocol with zero failed environment
steps and zero non-timeout physical-boundary failures, mean height in
`[0.85, 1.30] m`, mean tilt at most `0.15 rad`, and maximum tilt at most
`0.30 rad`. Failure restores the 4.6 Hz source and records a negative model
result; no learner is authorized.

If the guard passes, it receives exactly one 128-environment × 128-step ×
256-update PPO trial (chunk 8, fixed learning rate `1e-4`, initial log standard
deviation `-2`, learner seed `2650443581`, checkpoints every 64 updates). The
immutable selector evaluates every checkpoint and final candidate against the
fresh 64-environment incumbent for 5,000 no-reset band-2 steps at that held-out
seed. A replay, Crow GIF, picture, README entry, or protected deployment still
requires `tracking >= 0.70` and zero non-timeout physical-boundary failures.

### Outcome: rejected by the height gate

Source revision `6b1dcdb` compiled on the Apple M4 Pro and
`metalrobo_run_program_check` passed. The pre-registered zero-action Crow
guard then ran 64 environments × 5,000 steps, band 2, no scheduled resets,
and seed `2650443581` at the published 6.4 Hz clock. All 64 environments
reached normal timeout with zero failed environment steps, zero non-timeout
physical-boundary failures, and mean / maximum tilt `0.0675752 / 0.1075437
rad`. However, mean root height was `1.5651201 m`, exceeding the `1.30 m`
upper gate; tracking was `0.4994149`, mean final forward progress was
`-35.36499 m`, and maximum forward progress was `0 m`. The immutable artifact
root is `.numi/runs/crow-published-takeoff-clock-20260824-v1/zero-baseline-64x5000/`.

The candidate is therefore rejected before PPO. The active source restores the
qualified 4.6 Hz visual-hybrid clock exactly; the v2 provenance record and all
M4 Pro evidence remain retained. This demonstrates that the published maximal
vertical-takeoff frequency is not interchangeable with the present hybrid's
full-horizon forward-flight clock. It does not refute the source experiment or
establish a calibrated Crow flight model. No deployment, replay, GIF, picture,
or README media is authorized from this timing variant.

### Restored v1 verification

After the rollback commit, the Apple M4 Pro rebuilt the live package and
`metalrobo_run_program_check` passed with the restored 4.6 Hz task fingerprint
`4623727717616635550`. The same zero-action, band-2, no-reset 64 × 5,000-step
guard and seed `2650443581` reproduced the qualified v1 metrics exactly:
`0` failed environment steps, `64` normal timeouts, mean / maximum root height
`1.0409775 / 1.0953953 m`, mean / maximum tilt `0.0635701 / 0.1247985 rad`,
mean tracking `0.5006364`, and mean / maximum forward progress
`16.8779269 / 17.9386941 m`. The immutable restored artifact root is
`.numi/runs/crow-published-takeoff-clock-20260824-v1/restored-v1-zero-64x5000/`.
This is rollback evidence only; the 4.6 Hz hybrid still does not meet the
separate held-out PPO selection gate for a flight-policy replay or media.

## Controller-only residual envelope (pre-registered)

The retained 4.6 Hz baseline has zero physical-boundary failures, whereas the
first articulated-load PPO trial first improved tracking to `0.546855` but
already incurred `2.015625` physical failures per environment at revision 65;
later checkpoints further increased tilt and failures. This identifies a
controller-authority problem, not evidence for a new Crow morphology,
aerodynamic coefficient, or target trajectory. In the live stage-2 task,
the previous normalized `0.25` wing residual changes the flapping-amplitude
command by up to `0.125`; the same residual changes the pronation target by
up to `0.075 rad` in addition to its verified carrier. Those excursions can
cancel or overwhelm the qualified state-feedback carrier before PPO has
learned a safe correction.

The candidate therefore changes only the stage-2 residual multipliers in the
Metal action-application kernel: flapping and non-wing residuals `0.25 →
0.10`, tail residual `0.10 → 0.05`, and pronation residual `0.25 → 0.05`.
The resulting maximum residuals are respectively `±0.050` flapping-amplitude
command, `±0.020 rad` sweep, `±0.015 rad` pronation, and half of the previous
tail correction. The task clock, articulated mechanics, Blade-element and
unsteady coefficients, observation/action ABI, curriculum, rewards, resets,
terminations, policy architecture, optimizer, and held-out selector remain
unchanged. This is an exploration-envelope test for an estimated hybrid, not
a biological control claim.

Before PPO, the candidate must build on the Apple M4 Pro and pass
`metalrobo_run_program_check`, then pass two 64-environment × 5,000-step,
band-2, no-scheduled-reset guards at seed `2650443581`: (1) zero actions and
(2) a persisted, deterministic full-envelope Rademacher action stream with
each 14-lane vector held for eight control steps. Both require zero failed
environment steps, zero non-timeout physical-boundary failures, mean height in
`[0.85, 1.30] m`, mean tilt at most `0.15 rad`, and maximum tilt at most
`0.30 rad`. The stress stream is only a controller-safety probe; it is neither
a Crow kinematic target nor training data. Failure restores the qualified
residual multipliers and records a negative result before PPO.

If both guards pass, the candidate receives exactly one matched 128-environment
× 128-step × 256-update PPO run (chunk 8, learning rate `1e-4`, initial log
standard deviation `-2`, learner seed `2650443581`, checkpoints every 64
updates). The existing immutable 64-environment, 5,000-step held-out selector
evaluates every checkpoint and final candidate. Nothing advances to deployment,
replay, GIF, picture, or README unless tracking is at least `0.70` with zero
non-timeout physical-boundary failures.

### Outcome: rejected by full-envelope stress guard

At source revision `0491bf6`, the Apple M4 Pro rebuilt the Metal kernel,
`metalrobo_task_rollout`, `metalrobo_task_train`, and
`metalrobo_run_program_check`; the program check passed. The correctly labelled
zero-action guard then reproduced the active baseline: `0` failed environment
steps, `64` normal timeouts, mean / maximum height `1.0409775 / 1.0953953 m`,
mean / maximum tilt `0.0635701 / 0.1247985 rad`, tracking `0.5006364`, and
mean final forward progress `16.8779269 m`. Its immutable root is
`.numi/runs/crow-residual-envelope-20260824-v1/zero-baseline-64x5000-labeled/`.

The persisted stress stream has 14 action lanes for 64 environments and 5,000
control steps, holds each SplitMix64-keyed Rademacher vector for eight steps,
uses seed `2650443581`, and has SHA-256
`cfb160d076d46fc6efbebd2bb45a3e5d2c864a76095f8266b8231b4e0755102f`.
The full-envelope M4 Pro rollout had `0` failed environment steps but `377`
non-timeout physical-boundary terminations (reason 3), no normal timeouts,
mean height `0.9254082 m`, mean / maximum tilt `0.2148062 / 1.1162999 rad`,
tracking `0.4997821`, and mean final forward progress `-9.3169223 m`. Its
immutable root is
`.numi/runs/crow-residual-envelope-20260824-v1/stress-rademacher-64x5000/`.

The candidate fails the pre-registered physical-boundary and attitude gates
before PPO. The active source therefore restores the earlier residual envelope
(`0.25` flap/non-wing/pronation, `0.10` tail). The v3 provenance lock and both
guard artifacts remain retained as negative controller evidence. No candidate
policy, deployment, replay, GIF, picture, or README entry is authorized.

After rollback source revision `cb0f514`, the Apple M4 Pro regenerated the
Metal library, rebuilt both Swift evaluation/training executables, and passed
`metalrobo_run_program_check`. A fresh restored 64 × 5,000-step zero-action
guard at the same held-out seed reported the original residual label and
reproduced the active baseline exactly: `0` failed environment steps, `64`
normal timeouts, mean / maximum height `1.0409775 / 1.0953953 m`, mean /
maximum tilt `0.0635701 / 0.1247985 rad`, tracking `0.5006364`, and mean /
maximum forward progress `16.8779269 / 17.9386941 m`. Its immutable root is
`.numi/runs/crow-residual-envelope-20260824-v1/restored-v1-zero-64x5000/`.

## Low-exploration PPO trial (pre-registered)

The two controller-envelope experiments are rejected, and the active 4.6 Hz
hybrid and its physical task are now independently requalified. The remaining
direct evidence from the prior matched PPO run is that its initial
`log standard deviation = -2` corresponds to mean action standard deviation
`0.13514`, and the learner accumulated contact terminations before its
held-out candidates failed. The next test changes only the initial exploration
scale to `log standard deviation = -3` (standard deviation about `0.04979`).
It does not alter the model, action bounds, controller, reward, curriculum,
physics, optimizer, learning rate, seed, policy architecture, checkpoint
schedule, or selector. This is an RL stability experiment, not a claim about
Crow control or movement.

The one authorized run starts a new zero-actor-output Crow policy at source
`73bad6b`, using 128 environments × 128 steps × 256 updates, chunk 8, fixed
learning rate `1e-4`, learner seed `2650443581`, and checkpoints every 64
updates. The immutable selector uses 64 environments, 5,000 no-reset band-2
steps, and held-out seed `2650443581` to compare every checkpoint and final
candidate against the fresh incumbent. A candidate must reach tracking at
least `0.70`, have zero non-timeout physical-boundary failures, and not regress
the height/attitude envelope before any deployment, replay, GIF, picture, or
README entry. Otherwise the incumbent remains active and the result is
negative evidence; no follow-on hyperparameter sweep is authorized by this
pre-registration.

### Outcome: incumbent retained; low exploration did not clear the gate

The first wrapper invocation was a configuration-only failure: it omitted the
explicit checkpoint directory required by the checkpoint interval, and stopped
before launching the learner. The reissued command added that path without
changing the pre-registered training parameters. Its source revision was
`713ea69`; the executable Crow code remained the independently requalified
`73bad6b` code. The remote Apple M4 Pro completed all 256 updates
(`4,194,304` samples) in `213.7332 s`, reporting `160,336.31` GPU ms and
`19,624.02` end-to-end environment steps/s. Training reported zero failed
environment steps; that is a trainer-health observation, not a flight result.

The held-out selector evaluated each persisted checkpoint and the final
candidate over the pre-registered 64-environment, 5,000-step condition. The
figures below are mean tracking, mean root height, mean tilt, and non-timeout
physical-boundary terminations per evaluation environment. None met the
tracking threshold or zero-physical-failure requirement, so each was rejected
and the incumbent was retained.

| Policy | Tracking | Height (m) | Tilt (rad) | Physical terminations / env | Selector result |
| --- | ---: | ---: | ---: | ---: | --- |
| Incumbent | 0.500636 | 1.040978 | 0.063570 | 0.000000 | retained |
| Update 65 | 0.507202 | 1.037329 | 0.135742 | 2.812500 | rejected |
| Update 129 | 0.560095 | 0.545074 | 0.240295 | 3.000000 | rejected |
| Update 193 | 0.579314 | 0.563368 | 0.235984 | 2.984375 | rejected |
| Update 257 (final) | 0.502275 | 1.062237 | 0.156605 | 4.609375 | rejected |

The immutable record is
`.numi/runs/crow-low-exploration-20260824-v1/train-128x128x256-reissued/`,
including the initial/final policies, all four checkpoints, 64-environment
state traces, evidence JSON, selector decision, source revision, runtime hash,
and artifact checksums. The selector records `selected = incumbent`,
`comparison_champion = incumbent`, and
`candidate_advanced_deployment = false`. Consequently there is no accepted
Crow policy, replay, GIF, picture, or README media entry from this experiment.

## Stage-2 leg-authority transport probe (pre-registered)

The current band-2 carrier begins its resolved wing actuation with all six leg
residual actions at zero. This does not recreate an American-crow takeoff: the
available Corvidae study reports a pre-lift-off counter-movement and partial
first downstroke before toe-off, but does not publish the per-frame leg and
wing data needed to reconstruct one. It does identify a causal gap worth
testing: whether the existing, physical leg-action path can transmit a small
support-to-liftoff perturbation through the estimated hybrid without a solver
error or contact failure. The literature is rationale only, not an input
trajectory or parameter fit.

The one authorized test uses the already compiled
`--birdflow-ground-gait-probe` action stream at source `b365888`. It sends the
existing symmetric, zero-mean, 0.50-s alternating leg residual through action
lanes 7--12; all wing, sweep, pronation, and tail policy lanes remain zero,
while the unchanged Metal-resident stage-2 wing, pronation, and tail carriers
remain live. This is a leg-actuator transport control, not a learned gait, a
counter-movement reconstruction, or a policy-training run.

It will run once on the remote Apple M4 Pro: 64 environments × 5,000 control
steps, band 2 only, no scheduled resets, seed `2650443581`, with environment-0
state trace. It is compared with the requalified zero-action incumbent. The
probe is considered physically bounded only if it has zero failed environment
steps and zero non-timeout physical-boundary terminations, mean root height in
`[0.85, 1.30] m`, mean tilt at most `0.15 rad`, and maximum tilt at most
`0.20 rad`. Those checks determine only whether the action path remains safe
enough for further system identification. They cannot select a policy or
authorize a replay, GIF, picture, README entry, claim of American-crow
kinematics, or a follow-on PPO sweep.

### Outcome: physically bounded, but no useful transport signal

The initial capture invocation stopped during argument validation because a
state trace requires `--chunk 1`; it executed no physics step. The reissued
run changed only that capture setting. On the Apple M4 Pro it completed the
pre-registered 64 × 5,000 condition in `16.1333 s` (`13,959.49` GPU ms;
`19,834.75` environment control steps/s). All 64 environments reached the
normal authored timeout (reason 4), with zero failed environment steps and
zero non-timeout physical-boundary terminations.

| Action source | Tracking | Height (m) | Mean / max tilt (rad) | Final forward progress (m) | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Requalified zero-action incumbent | 0.500636 | 1.040978 | 0.063570 / 0.124799 | 16.877927 | reference |
| Existing low-amplitude leg probe | 0.500605 | 1.041254 | 0.063502 / 0.124808 | 16.684748 | bounded, no tracking gain |

The existing support-gait perturbation is therefore safe at this amplitude but
does not show a useful standing-to-liftoff transport effect. It is not a
counter-movement reconstruction, does not validate a stronger leg sequence,
and does not authorize policy training. The immutable reissued record is
`.numi/runs/crow-leg-authority-transport-20260824-v1/ground-gait-probe-64x5000-reissued/`;
it contains the evidence JSON, environment-0 trace, runtime/source hashes, and
artifact checksums. No Crow policy, replay, GIF, picture, or README media was
created. A valid next controller experiment requires the licensed event- and
kinematics-resolved Crow data specified in the data-intake record, rather than
inventing a higher-amplitude biological trajectory from aggregate literature.

## Curriculum-protected standing-to-flight learning (pre-registered)

The preceding controls rule out treating the existing small ground-gait probe
as a Crow takeoff target. They do not rule out learning the already-authored
ground-support and lift-off task as a curriculum. Its first three physical
bands are passive bilateral support (0), carrier-supported walking (1), and
articulated lift-off (2). The earlier PPO trials trained only
band 2, so they asked a zero actor to explore wing, leg, tail, sweep, and
pronation residuals without first learning the task's supported walking rung.

Source `0a817bb` corrects the selector for this experiment: a mixed BirdFlow
curriculum is evaluated on its newest band, and a separate held-out evaluation
must show that the preceding band did not regress. This changes selection
evidence only. The estimated-hybrid mechanics, visual lock, 4.6 Hz clock,
ABA/contact solver, aerodynamic closure, reward, termination conditions,
action ABI, and `tracking >= 0.70` lift-off gate remain unchanged.

The protocol allows at most two remote Apple M4 Pro learner runs, each with
128 environments × 128 steps × 512 updates (`8,388,608` samples), chunk 8,
fixed learning rate `1e-4`, initial log standard deviation `-3`, learner seed
`2650443581`, and checkpoints every 128 updates. Stage 1 begins from a new
zero-actor-output Crow policy and uses band 1 only. Its 64-environment,
5,000-step, no-reset, held-out selector evaluates band 1 plus protected band
0. It may advance only if it has no failed environment steps, no new
non-timeout physical-boundary failure, a positive staged outcome comparison,
and no band-0 regression.

Stage 2 is conditional on a selected Stage-1 deployment. It initializes that
actor with a fresh critic, trains on bands 1--2, and uses the same held-out
width, horizon, and seed. Its selector evaluates band 2 directly and protects
band 1. It may advance only if it also reaches tracking at least `0.70`, has
zero non-timeout physical-boundary failures, has positive staged progress, and
does not regress walking. Failed checks preserve the incumbent and terminate
this pre-registered protocol; no parameter sweep, policy replay, GIF, picture,
or README entry follows from a rejected candidate.

The `numi crow train` hand-off rejects `--zero-actor-output` when an actor
PolicyPack is supplied. This is an integration safeguard, not a change to the
pre-registered experiment: actor transfer must not be silently replaced by a
zeroed policy.

### Stage 1 outcome — rejected

The pre-registered Stage-1 run completed on the remote Apple M4 Pro at source
`d1181af481d3d9d2d7531aab48afa2a6eb3d5da3`. Its immutable artifact root is
`.numi/runs/crow-staged-curriculum-20260824-v1/stage1-band1-128x128x512/`.
It trained the new zero actor for all 512 updates (`8,388,608` samples) in
536.3824 s, reporting 429,268.085 GPU ms, 15,639.232 end-to-end environment
steps/s, and zero failed environment steps. Those are learner-runtime facts,
not held-out flight evidence.

The fixed selector then evaluated the incumbent, every scheduled checkpoint,
and the final candidate with 64 environments × 5,000 no-reset steps and held-
out seed `2650443581` on band 1 and protected band 0. The final candidate had
zero physical-boundary failures on both rungs and materially higher band-1
tracking (`0.995926` versus `0.911982`), but its mean tilt was `0.033954` rad
versus the incumbent's `0.003917` rad. That increase is a hard regression
under the pre-registered selector, so the champion is the incumbent,
`candidate_advanced_deployment` is `false`, and the learned candidate remains
retained only for diagnosis. The protected band-0 comparison had no
regressions.

Accordingly Stage 2 was not started. No Crow policy replay, GIF, picture, or
README media was created. The run's SHA-256 checksums are `95f6ce…60d0a` for
`evidence.json`, `a9bb72…7cfe` for `selection/selection.json`, and
`704e53…99d14` for `artifacts.sha256`; the complete values are in the retained
remote artifact root.

## Ground-leg residual contract and reissued curriculum (pre-registered)

The rejected Stage-1 candidate reveals one implementation-level confound that
is separate from Crow biological data: in band 1, the Metal task explicitly
folds both flap actions, but its generic residual route still exposed the
shoulder-sweep, distal-pronation, and tail position drives alongside the six
leg drives. The candidate improved its walking tracking while exceeding the
tilt gate. That does not prove a particular joint caused the tilt, but it
makes non-leg residual authority an unnecessary and falsifiable explanation
to remove before another learner run.

The reissued compiled task adds
`MR_TASK_PROGRAM_AVIAN_CROW_GROUND_LEG_RESIDUAL`. In carrier-supported band 1,
only action indices 7--12 (bilateral hip, knee, and ankle position drives)
retain the existing `0.25` learned residual around the existing live gait
carrier. Flaps remain folded and sweep, pronation, and tail position targets
remain at their mechanism defaults. Band 0 retains its passive support
behavior; band 2 and later retain the existing live altitude, speed, tail,
and pronation trim controller plus their bounded residuals. This is an action-
authority and task-fingerprint change only: it does not modify morphology,
mass/inertia, solver, aerodynamic coefficients, reward weights, termination
bounds, 4.6 Hz clock, or policy architecture.

Before learning, one remote-M4 actuation-isolation test is authorized. It uses
the revised compiled task at band 1, 64 environments, 5,000 no-reset steps,
and seed `2650443582`; it compares zero actions with an otherwise identical
float32 action stream that drives only the now-masked sweep, pronation, and
tail lanes. The program check must pass, both rollouts must have zero failed
environment steps and zero non-timeout physical failures, and their
environment-0 physical state traces must be byte-identical. Any difference
rejects this reissue and starts no learner.

### Actuation-isolation outcome — passed

At source `4d4c2a4`, the remote Apple M4 Pro rebuilt the Metal library and
passed `metalrobo_run_program_check`. The two prescribed band-1 rollouts are
retained under `.numi/runs/crow-ground-leg-residual-20260824-v1/zero-actions/`
and `.../nonleg-masked-actions/`. Both used the exact 64 × 5,000 no-reset
configuration and seed `2650443582`; both have task fingerprint
`3198934467138572318`, run fingerprint `8498436949874751629`, zero failed
environment steps, 64 ordinary timeouts, zero height/tilt terminations,
tracking `0.9119822925`, root height `0.1873070948 m`, and mean tilt
`0.0039169387 rad`.

The latter stream drives only action lanes 2--6 (sweep, pronation, and tail)
at full normalized amplitude and has SHA-256
`ed4953c7346fa7346c192e70428773bf83e58523a3bfc2c9254998514d60aeab`.
Despite those requests, the environment-0 physical traces are byte-identical:
both hash to
`f9a4dfd48cb3b3f65fee533ee16583849972312e8b70728cdb5127be0ec2110b`.
This proves the implemented band-1 action boundary, not that a flight policy
has been learned. It satisfies the isolation gate and authorizes the one
fresh Stage-1 learner run below.

Only after that isolation gate passes, one fresh zero-actor Stage-1 learner
run is authorized: 128 environments × 128 steps × 512 updates (`8,388,608`
samples), chunk 8, fixed learning rate `1e-4`, initial log standard deviation
`-3`, learner seed `2650443582`, checkpoint interval 128, and source-pinned
runtime artifacts. Its selector uses the same 64-environment, 5,000-step,
no-reset held-out band 1 plus protected band 0 and the same seed. It may
advance only with zero failed environment steps, zero non-timeout physical
failures, a positive current-band comparison, no protected-band regression,
and mean tilt no more than `0.005` rad above the matched incumbent.

At most one Stage 2 is conditional on a selected Stage-1 deployment. It
transfers that exact actor with a fresh critic, trains bands 1--2 under the
same learner configuration and seed, and selects on held-out band 2 while
protecting band 1. It must also achieve tracking at least `0.70`, have zero
non-timeout physical failures, positive staged progress, no walking
regression, and no mean-tilt regression. There is no parameter sweep. Failed
gates retain artifacts but prohibit further learner runs, policy replay, GIF,
picture, or README media under this reissued protocol.

### Reissued Stage 1 outcome — rejected

The authorized reissued Stage-1 run completed on the remote Apple M4 Pro at
source `640f7b37d8ab2dd4ae21bcb96dd9e9d2ff00db06` (including the task-program
change introduced at `4d4c2a4`). Its immutable artifact root is
`.numi/runs/crow-ground-leg-residual-20260824-v1/stage1-band1-128x128x512/`.
It trained the zero actor for all 512 updates (`8,388,608` samples) in
537.3569 seconds, with 430,137.8354 measured GPU milliseconds and 15,610.87
end-to-end environment steps per second. The learner reported zero failed
environment steps.

The prescribed 64-environment, 5,000-step, no-reset held-out selector chose
the incumbent for current band 1. The candidate's tracking score improved from
`0.9119823` to `0.9899237`, and both candidate and incumbent had zero physical
failure rate, but mean tilt increased from `0.0039169` to `0.0790510` rad. The
`0.0751340`-rad increase exceeds the pre-registered `0.005`-rad ceiling, so
the selector records `mean tilt increased`, retains the candidate only as an
artifact, and sets `candidate_advanced_deployment` false. The candidate did
pass its protected band-0 comparison without regressions; that support result
does not waive the current-band tilt gate.

Accordingly, Stage 2 is not authorized. No Crow policy replay, GIF, picture,
or README media was created from this run. SHA-256 checksums are
`d19d6fb2e6af151bd79972c8d158328663d117e3717e3aca5e1b05f1f60d9b08` for
`evidence.json`,
`c4e89e9706977d36182fb5f89ddbca951db3bd5a2491a0f14050a7837da0fcbb` for
`selection/selection.json`, and
`1c80145457e8c16b6ed4c8797c3b0d69577d6641be5d9d519306fa3e98512a51` for
`artifacts.sha256`. The full evidence remains at the remote artifact root.

### Leg-residual attribution (pre-registered)

The new rejection establishes a current-band attitude regression but not the
size or temporal structure of the six live leg commands that produced it. One
nonvisual, no-learning attribution evaluation is therefore authorized before
any future task change. It is a diagnostic execution record, not a policy
promotion, replay, or media capture.

At source `7fb13e3`, with the same compiled task fingerprint
`3198934467138572318`, the remote Apple M4 Pro will evaluate exactly the
retained Stage-1 candidate PolicyPack
`e5b6a6119c311a9010956e920ab2f3134d6a4818c8078f91544018c571da4a93` and the
protected incumbent PolicyPack
`3b6684fc91010f9b00dfd1f9612147b890eedb3a21a46022ce6004a200b404d9`. Each
uses band 1 only, 64 environments, 5,000 steps, one repeat, chunk 1,
`--no-scheduled-resets`, seed `2650443582`, and a `PolicyRolloutPack` plus an
environment-0 physical state trace. The pack records the device-generated
normalized policy latents for every action at every time step; its action bias
is zero and action scale is one, so those records are the normalized commands
that enter the live residual path before that path's fixed `0.25` multiplier.

The analysis will report per-action mean, RMS, mean absolute value, and
maximum absolute value, separately for masked channels 0--6 and live leg
channels 7--12. The candidate execution must reproduce its retained selector
metrics and state trace hash before those command summaries are interpreted.
The incumbent establishes the fixed-carrier baseline. This diagnostic cannot
authorize a learner, policy deployment, flight replay, GIF, picture, or README
media. It can only identify whether a single subsequently pre-registered
leg-authority intervention is scientifically warranted.

### Leg-residual attribution outcome

The two pre-registered M4 executions reproduced their retained held-out
records exactly. The candidate's 64 × 5,000 record has the same tracking
(`0.9899237`), mean tilt (`0.0790510` rad), maximum tilt (`0.1199241` rad),
final forward progress (`10.6673941 m`), zero failed environment steps, and
environment-0 state-trace SHA-256
`1cfffbc9ce5d63b9a977092f81c453bb3105f1a53286def45e481668494a5e3d` as the
selector record. The incumbent likewise reproduced its metrics and its
environment-0 trace SHA-256
`f9a4dfd48cb3b3f65fee533ee16583849972312e8b70728cdb5127be0ec2110b`.

The incumbent PolicyPack (revision 1) has exactly zero normalized output on
all 14 channels. The rejected candidate (revision 513) emits nonzero commands
on masked channels 0--6 (combined RMS `0.0852032`), but the earlier
action-isolation result proves they cannot affect band-1 physics. Its six live
leg channels have combined RMS `0.3898442`; after the fixed `0.25` residual
multiplier, that is an RMS effective leg command of `0.0974611`.

| Live leg action | Raw mean | Raw RMS | Mean absolute | 95th percentile absolute | Maximum absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| left hip | -0.21539 | 0.49786 | 0.40470 | 0.94413 | 1.17195 |
| left knee | -0.12432 | 0.16920 | 0.14136 | 0.29160 | 0.39321 |
| left ankle | -0.23508 | 0.38517 | 0.33412 | 0.61382 | 0.65901 |
| right hip | -0.23344 | 0.55560 | 0.49310 | 0.85301 | 0.97550 |
| right knee | -0.15897 | 0.34447 | 0.31241 | 0.52265 | 0.67356 |
| right ankle | -0.11892 | 0.24426 | 0.19291 | 0.45058 | 0.54411 |

This is execution evidence that the only live learned authority is substantial
leg authority; it does not by itself prove that a lower residual envelope will
pass the tilt gate or describe biological Crow gait.

### Fixed leg-amplitude counterfactual (pre-registered)

One nonvisual physical-control counterfactual is authorized. It decodes the
verified candidate `PolicyRolloutPack` latents into a little-endian Float32
action stream in the native step-major/environment-major/action-major order.
First, that unmodified stream must reproduce the candidate trace and evidence
above under the same M4 task, band 1, 64 environments, 5,000 steps, one
repeat, chunk 1, seed `2650443582`, and no scheduled resets. This validates
the artifact decoder and action ordering.

The only counterfactual then multiplies lanes 7--12 by `0.20`, leaving all
other raw lanes and every run condition byte-for-byte unchanged. Because the
task's live leg residual is fixed at `0.25`, this tests an effective residual
multiplier of `0.05` without changing the policy, morphology, reward, solver,
aerodynamics, action ABI, or task fingerprint. The value was fixed before
running the counterfactual: it changes the observed candidate leg RMS envelope
from `0.0974611` to `0.0194922`, while retaining a nonzero residual around the
existing gait carrier.

The counterfactual is informative only if it has zero failed environment
steps, zero non-timeout physical failures, tracking at least the incumbent's
`0.9119823`, and mean tilt no more than the incumbent plus the existing gate
(`0.0089169` rad). Passing authorizes only a separately pre-registered task
change from the `0.25` ground-leg residual to `0.05` and a fresh qualification
protocol; it does not select this rejected actor, authorize a learner, or
create policy replay, GIF, picture, or README media. A failed counterfactual
ends this amplitude explanation without a scale sweep.

### Fixed leg-amplitude counterfactual outcome — control rejected

At source `a828700a08ff9cda0720020eb3254b2dd4163452`, the candidate action
latents were decoded into the prescribed Float32 stream. Its SHA-256 is
`7428ea026cb492b95d20c784d26578416a5c5b30a286e6029c709880d9edc1b1`.
The unmodified-stream control ran on the same M4 task and retained task and
run fingerprints `3198934467138572318` and `8498436949874751629`, zero failed
environment steps, and 64 ordinary timeouts. It was close to, but did not
exactly reproduce, the candidate execution: tracking was `0.9898425` rather
than `0.9899237`, mean tilt was `0.0791258` rather than `0.0790510` rad, and
final forward progress was `10.7266606` rather than `10.6673941 m`.

Most importantly, the control environment-0 trace hashes to
`0f0b3de382743d26026c18fb94fa42c1cdb270ac080bba35d597dad5a8a7c177`, not the
required candidate trace hash
`1cfffbc9ce5d63b9a977092f81c453bb3105f1a53286def45e481668494a5e3d`.
The early rows differ only at floating-point round-off scale and then diverge
over the long contact trajectory, but the pre-registered requirement was
byte-identical execution, not approximate agreement. The decoded latent record
is therefore insufficient as an exact external action-stream replay of the
compiled-policy path.

The generated 0.20 leg stream (SHA-256
`a797a01e5b193c1abe980d5bfdb926c40ad9186a8e2a101bc8ffd3d40f4ac3ab`) was
retained but not executed. No scaled counterfactual, source amplitude change,
learner, policy deployment, replay, GIF, picture, or README media is authorized
from this failed control. A later experiment would first need a separately
qualified way to apply scaling inside the same native policy-execution path;
that is an instrumentation problem, not evidence that the proposed 0.05
residual is safe or effective.

### Native-policy action-scale counterfactual (pre-registered)

The rejected external-stream control does not invalidate the recorded action
amplitudes; it invalidates substituting a host-uploaded action buffer for the
compiled policy path. The successor diagnostic therefore applies the one
fixed intervention through the existing PolicyPack action-scale field, which
the live Metal inference kernel multiplies into its device-resident action
buffer before the task consumes it.

First, the native writer must deserialize and reserialize the candidate
PolicyPack without changing any field. The resulting identity artifact must be
byte-identical to the retained candidate hash
`e5b6a6119c311a9010956e920ab2f3134d6a4818c8078f91544018c571da4a93`.
Only then may it write one counterfactual PolicyPack with the same actor,
critic absence, observations, bias, clip, contract, identity, and revision;
the sole changed values are action-scale lanes 7--12, each from `1.0` to
`0.20`. The masked channels retain their original scale because their task
authority is already zero. This is an exact native-policy execution change:
the existing `0.25` task residual makes it a `0.05` effective leg envelope.

The M4 evaluation uses the counterfactual PolicyPack directly (not an action
stream) at source `0196028`, band 1, 64 environments, 5,000 steps, one repeat,
chunk 1, held-out seed `2650443582`, and no scheduled resets. Its task and run
fingerprints must remain `3198934467138572318` and `8498436949874751629`.
It is eligible only as a causal amplitude result if it has zero failed
environment steps, zero non-timeout physical failures, tracking at least
`0.9119823`, and mean tilt at most `0.0089169` rad. A pass can authorize a
separate source-pinned `0.05` ground-leg residual task change and fresh
qualification; it cannot deploy this counterfactual PolicyPack, start a
learner, or create a policy replay, GIF, picture, or README media. A failure
ends this amplitude explanation without another scale.

### Native-policy action-scale counterfactual outcome — rejected

The native writer's identity round trip passed: its output is byte-identical
to the retained candidate PolicyPack and retains SHA-256
`e5b6a6119c311a9010956e920ab2f3134d6a4818c8078f91544018c571da4a93`.
The one authorized counterfactual PolicyPack changes only action scales 7--12
to `0.20`; its SHA-256 is
`3e97d127b3cbf1085317cd921403217a72f9b17e9db394bfc45ac550d85974cc`.
The actor weights, biases, policy identity and revision, semantic contract,
task fingerprint, and run fingerprint remain unchanged. It therefore applies
the scale inside the same native Metal policy-inference path, rather than the
rejected host action-stream path.

The prescribed M4 evaluation had task and run fingerprints
`3198934467138572318` and `8498436949874751629`, zero failed environment
steps, zero physics errors, and 64 ordinary timeouts. It did not pass the
pre-registered causal criteria: tracking was `0.9109834`, below the required
`0.9119823`, and mean tilt was `0.0109013` rad, above the required
`0.0089169` rad. Its maximum tilt was `0.0158081` rad and final forward
progress `0.0672778 m`. The evaluation evidence and state-trace SHA-256 values
are respectively
`89013f05de5673ec08d4daad4996ee2ba8bac978c288f15e8b94155435f0f08c` and
`f09731ad51853f0ac4f832b52a1dc5a6f0060448365ed233cf5aed0688eb3fcc`.

The fixed 0.05 effective leg envelope is therefore rejected as a sufficient
explanation or a qualifying task setting. No source residual-scale change,
second scale, learner, deployment, replay, GIF, picture, or README media is
authorized from this result. The retained artifact may inform a distinct,
separately pre-registered hypothesis about the learning objective, but it is
not a Crow flight result.

### Ground-tilt objective reissue (pre-registered)

Source `07628e0cfc564eb44f1690799ec4bb0534044a57` introduces one Crow-only,
fingerprinted task flag:
`MR_TASK_PROGRAM_AVIAN_CROW_GROUND_TILT_ENVELOPE`. In carrier-supported band 1
only, `tiltSquared` now uses `4 * max(tilt - 0.0075, 0)^2` before the existing
`-0.50` reward weight, producing a `-2 * max(tilt - 0.0075, 0)^2` hinge
penalty. Standing, lift-off, and flight bands retain their prior reward
behavior; the Dove, robot model, action ABI, gait carrier, residual scale,
solver, aerodynamics, resets, terminations, 4.6 Hz clock, policy architecture,
and learning hyperparameters are unchanged.

This is an objective-alignment hypothesis, not an amplitude sweep. In the
rejected Stage-1 candidate, the observed task-reward gain was `0.0064682`
(`0.0766220 - 0.0701539`) while mean tilt was `0.0790510` rad. Since the hinge
is convex and tilt is nonnegative, its expected additional penalty at that
mean is at least `2 * (0.0790510 - 0.0075)^2 = 0.0102391`; it exceeds the
observed reward advantage while the incumbent mean tilt (`0.0039169` rad) is
below the hinge. The `0.0075`-rad onset leaves a buffer below the unchanged
held-out mean-tilt ceiling of `0.0089169` rad. This bound motivates one test;
it does not guarantee a learned gait or describe biological Crow control.

Before learning, the M4 must rebuild the Metal program and pass
`metalrobo_run_program_check`. A zero-action band-1 isolation run then uses 64
environments, 5,000 no-reset steps, seed `2650443582`, and an environment-0
state trace. It must have zero failed environment steps and be byte-identical
to the retained ground-carrier trace
`f9a4dfd48cb3b3f65fee533ee16583849972312e8b70728cdb5127be0ec2110b`.
The reward is allowed to differ; the test is specifically a physical-path
guard that the objective-only change does not alter zero-action dynamics.

Only after that isolation passes, one fresh Stage-1 learner is authorized at
this source: zero actor, band 1 only, 128 environments × 128 steps × 512
updates (`8,388,608` samples), chunk 8, fixed learning rate `1e-4`, initial
log standard deviation `-3`, learner seed `2650443583`, checkpoint interval
128, and source-pinned remote artifacts. Its 64-environment, 5,000-step,
no-reset held-out selector uses the same seed and evaluates current band 1
plus protected band 0. It may be retained only with zero failed environment
steps, zero non-timeout physical failures, positive current-band progress, no
protected-band regression, and mean tilt no more than `0.005` rad above the
matched incumbent. This reissue authorizes no Stage 2, policy deployment,
replay, GIF, picture, or README media; any later flight stage requires its own
pre-registration after a selected Stage-1 policy.

### Ground-tilt objective isolation outcome — passed

At source `4e7aa2be5b86e8346c4d3aad6f1be766e48c89cf`, the remote Apple M4 Pro
rebuilt the Metal library and passed `metalrobo_run_program_check`. The
prescribed zero-action band-1 run is retained at
`.numi/runs/crow-ground-tilt-envelope-20260824-v1/zero-actions/`. It has the
new task fingerprint `1967076388838221657` and run fingerprint
`7217577838424799429`, zero failed environment steps, 64 ordinary timeouts,
zero height and tilt terminations, tracking `0.9119822925`, mean tilt
`0.0039169387` rad, and mean root height `0.1873070948 m`.

Its environment-0 physical trace is byte-identical to the preceding ground
carrier control and hashes to
`f9a4dfd48cb3b3f65fee533ee16583849972312e8b70728cdb5127be0ec2110b`.
The evidence hash is
`af91a17789b58669b8fb03df83fcc65b5f14f4436b084782d75aafc8f79f5819`.
This verifies the narrow boundary claimed above: the objective and its task
fingerprint changed, but the zero-action physical path did not. It authorizes
the single fresh Stage-1 learner exactly as pre-registered.

### Ground-tilt objective Stage-1 outcome — rejected

The one authorized learner completed on the remote Apple M4 Pro at source
`638874bf11a74853d540b806fc874df635edd1e3`. Its immutable artifact root is
`.numi/runs/crow-ground-tilt-envelope-20260824-v1/stage1-band1-128x128x512/`.
It completed all 512 updates (`8,388,608` samples) with zero failed
environment steps in `536.1245 s`, consuming `429,087.2512` GPU ms
(`15,646.75` end-to-end environment steps/s). The compiled task fingerprint
was `1967076388838221657`; the retained world label remains
`birdflow_american_crow_estimated_hybrid`, so this is not an empirical Crow
locomotion result.

The fixed held-out selector evaluated the incumbent and every checkpoint
(revisions 129, 257, 385, and 513) on current band 1 and protected band 0:
64 environments, 5,000 no-reset steps per condition, seed `2650443583`.
Every candidate checkpoint violated the current-band tilt condition. The final
candidate (revision 513) had zero non-timeout physical failures: all 64
terminations were ordinary horizon timeouts. It improved tracking to
`0.9911406` from the incumbent's `0.9119823` and showed `18.6141 m` mean
final forward progress, but its mean tilt was `0.09179525` rad rather than the
incumbent's `0.00391692` rad. The pre-registered ceiling was `0.00891692` rad
(the matched incumbent plus `0.005` rad), so the candidate exceeded it by
`0.08287833` rad. The selector therefore records `selected = incumbent`,
`candidate_advanced_deployment = false`, and the sole current-band regression
`mean tilt increased`.

Protected band 0 did not regress: the final candidate had zero non-timeout
physical failures, tracking `0.9999999872` versus `0.9999999872`, and mean
tilt `0.00050899` versus `0.00050364` rad for the incumbent. That safety
preservation cannot override the failed current-band tilt gate. Checkpoint 257
also had a `0.328125` physical-failure/termination rate in current band, while
the other screened checkpoints still exceeded the tilt ceiling; the selector
thus did not hide a favorable intermediate checkpoint.

The retained evidence hashes are
`05d5eac1c9aade47641b751b39fa3069b1440baab346fca0a9a7f7fd7865830f`
(`evidence.json`),
`4354111304a2dc96f77ae920037f3cfb1a8525b7b0d4a13a023ddfeaac1cc031`
(`selection/selection.json`), and
`60b78a2f0701f3e82f7bddea1b9b5bea23ab69e3bb213b8971bbc5e0d4350f24`
(`artifacts.sha256`). The final candidate and incumbent current-band state
traces hash to
`cc6f9b3a6a51788107f0913344605cb27d7a7b5f9dcd7af375f9c7f884c69a68`
and
`1d8ff1d8f9e2dc0c0d7c84599267d0ecab660cceb62b64571323e5283c3006ad`,
respectively.

This rejects the hinge reward as a sufficient Stage-1 objective alignment. It
does not authorize Stage 2, deployment, replay, GIFs, pictures, README media,
or a standing-to-flight claim. The candidate and all evidence remain retained;
any new experiment must name and pre-register a distinct causal hypothesis
before another learner run.

### Bilateral common-mode rejection counterfactual (pre-registered)

The tilt rejection is not evidence that left/right asymmetry is the cause. The
environment-0 held-out final-candidate trace is, however, strongly
pitch-dominated: root quaternion `y` has RMS `0.045909124` and maximum absolute
value `0.048993524`, while quaternion `x` has RMS `0.001700220` and maximum
absolute value `0.003578208`. The matched incumbent values are respectively
`0.001917251` and `0.000621338`. Thus a left/right symmetry-only explanation
is not supported by this trace.

A separate, non-qualifying actor inspection used the immutable final training
`rollout.rolloutpack` (16,384 training samples, not the held-out selector) and
the final deterministic actor at revision 513. Its common components
`(left + right)/2` are `-0.24612`, `-0.14338`, and `-0.23003` for hip, knee,
and ankle respectively; their corresponding differentials `left - right` are
`0.05928`, `0.03266`, and `-0.17883`. In band 1 the compiled carrier itself
is anti-phase. This does not prove the common residual caused pitch, but it
motivates one falsifiable action-subspace test rather than another reward or
amplitude sweep.

The hypothesis is: removing only the bilateral *common* residual component
will prevent the candidate from buying forward tracking with a persistent
fore-aft pitch, while retaining each pair's differential residual and the
unchanged anti-phase ground carrier. The new Crow-only, fingerprinted task
flag will apply in band 1 to the three fixed pairs `(7,10)`, `(8,11)`, and
`(9,12)` before their existing delay and per-joint first-order filters:

```
r_left  = 0.5 * (a_left - a_right)
r_right = 0.5 * (a_right - a_left)
```

The policy's raw latents remain recorded as generated, but its action history
will contain the projected commands actually sent to the six live leg drives;
the next observation therefore reports executed rather than discarded
residuals. Non-leg lanes remain masked as before. This is a rank-three
common-mode projection, not a residual-scale change: it leaves each pair's
signed differential unchanged, leaves the `0.25` residual multiplier intact,
and introduces no additional learned or host-side controller.

Before any candidate execution, the remote Apple M4 Pro must rebuild the
Metal library and pass `metalrobo_run_program_check`. Its band-1 zero-action
isolation uses 64 environments, 5,000 no-reset steps, chunk 1, and seed
`2650443582`. It must have zero failed environment steps and reproduce the
existing environment-0 physical trace byte-for-byte with SHA-256
`f9a4dfd48cb3b3f65fee533ee16583849972312e8b70728cdb5127be0ec2110b`.

Because the added task flag correctly changes the semantic task fingerprint,
the rejected final actor and its incumbent must each be re-emitted through the
native PolicyPack writer with only the new semantic contract. Before rollout,
the M4 check must prove every actor layer weight, bias, observation
normalization vector, action bias/scale vector, policy ID, and revision is
bitwise identical to its source pack; only the task fingerprint and resulting
artifact content hash may differ. This is a contract rebind, not training or a
policy update.

One nonvisual M4 counterfactual is then authorized: the re-bound final
candidate and re-bound incumbent run current band 1 and protected band 0, each
at 64 environments × 5,000 no-reset steps, one repeat, chunk 1, held-out seed
`2650443584`. The candidate must have zero failed environment steps, zero
non-timeout physical failures, tracking at least the matched incumbent, no
protected-band regression, and mean current-band tilt no more than the matched
incumbent plus `0.005` rad. A pass identifies a viable action-subspace
hypothesis only; it still authorizes no learner, Stage 2, replay, GIF,
picture, README media, flight assertion, or claim of Crow biomechanics. A
failure ends this common-mode explanation without a projection variant.

### Bilateral common-mode rejection outcome — rejected

At source `a6ad92dc2fdac4cb0e08ac61bfa2302df34b1204`, the remote Apple M4 Pro
rebuilt the Metal task and passed `metalrobo_run_program_check`. The new task
fingerprint was `11608351420320865043`. Its prescribed zero-action isolation
completed with zero failed environment steps and 64 ordinary timeouts; tracking
`0.9119822925`, mean tilt `0.0039169387` rad, and mean root height
`0.1873070948 m` reproduced the control. Its physical trace is byte-identical
to the prior control at
`f9a4dfd48cb3b3f65fee533ee16583849972312e8b70728cdb5127be0ec2110b`.

The canonical writer re-emitted the final candidate and incumbent at the new
task contract without a learner update. Actor signatures, every actor tensor,
normalization/action vector, policy ID, and revision were bitwise identical:
candidate revision 513 has actor SHA-256
`f57733109025a8b13bb71b9ccd5d7b7144efbeaa7d15ead7a6ba9471583caba3`, and
incumbent revision 1 has actor SHA-256
`52684a9455d2d46f5a068e35c7f42bb21886fede0a8f363053acd443c5501e2a`.
Only the task contract and resulting PolicyPack content hashes differ.

The one 64 × 5,000 current-band evaluation at held-out seed `2650443584` was
physically clean: the projected candidate had zero failed environment steps,
zero non-timeout physical failures, and 64 ordinary horizon timeouts. It
improved tracking to `0.9785153` from the matched incumbent's `0.9119822` and
reduced mean tilt from the unreprojected candidate's `0.09179525` to
`0.02100372` rad. It nevertheless fails the pre-registered current-band
ceiling: the matched incumbent tilt is `0.00391697` rad, making the ceiling
`0.00891697` rad; the candidate remains `0.01208676` rad above it. Its
maximum tilt was `0.03275902` rad and mean final forward progress was
`14.243936 m`.

The protected band remained clean and did not regress: candidate versus
incumbent tracking was `0.9999999844` versus `0.9999999843`, mean tilt was
`0.0005029530` versus `0.0005014059` rad, and both runs had only 64 ordinary
timeouts. This does not waive the failed current-band attitude gate. The
common-mode projection therefore is not a qualifying explanation and is
reverted from the active task. No projection variant, learner run, Stage 2,
deployment, replay, GIF, picture, README media, or standing-to-flight claim is
authorized from this result.

The retained current-band candidate evidence and trace SHA-256 values are
`509b734f20a09c94ae88cf9da9e914fc28867da92fc2d09a29f370bc1f1af17a` and
`e3545233e1dfe66071c49063f02cf09e24f63c3cbbaa3f413efb148771e4b12d`.
The zero-action evidence and manifest hashes are
`70d22612ed5dd6f8620b61594b2be379cf4fa01ae3f0d23f46bdea3e53f2776b` and
`3443e15540088ded3733431f69886b1adb22f025707afe1ce8fae2c749ac0c11`.
All artifacts remain under
`.numi/runs/crow-ground-common-mode-20260824-v1/` on the M4.

### Swing-phase residual allocation counterfactual (pre-registered)

This is not a variant of the rejected common-mode spatial projection. That
experiment removed a bilateral coordinate subspace at every control step; the
present hypothesis instead allocates the unchanged per-leg residual in time.
The final actor's 16,384 captured training observations are finite, as are its
evaluated outputs. The output is predominantly DC rather than carrier-phase
locked: the left/right hip means are `-0.21648` / `-0.27576` with phase-series
AC RMS `0.02408` / `0.02650`; the left/right ankle means are `-0.31944` /
`-0.14062` with AC RMS `0.01816` / `0.01352`. This is diagnostic-only
training-batch evidence, not a held-out rollout or an estimate of Crow gait.

The distinct hypothesis is: persistent learned residuals perturb the carrier's
supporting leg as well as its swing leg, creating the observed pitch despite
the physical carrier's anti-phase gait. In band 1 only, the new Crow-only
fingerprinted task flag will preserve each live policy residual exactly when
its corresponding carrier leg is in swing and set its requested residual to
zero during that leg's support half-cycle. The three left actions (7--9) use
`sin(2*pi*t/0.50) > 0`; the three right actions (10--12) use the opposite
sign. The binary gate acts before the existing delay and first-order filter, so
the action history and subsequent policy observation represent the command
that actually reaches the drive.

This changes neither residual scale (`0.25`), action ABI, actor weights,
carrier phase, reward, morphology, solver, aerodynamics, reset, terminal
conditions, optimizer, or training schedule. It does not reconstruct a
measured avian support sequence; it is a one-shot causal allocation test in
the estimated hybrid.

The M4 must rebuild and pass `metalrobo_run_program_check`, followed by a
64-environment, 5,000-step, no-reset, chunk-1 zero-action isolation at seed
`2650443582`. It must have zero failed environment steps and reproduce the
physical state trace SHA-256
`f9a4dfd48cb3b3f65fee533ee16583849972312e8b70728cdb5127be0ec2110b`.
The rejected candidate and incumbent must then be re-emitted through the
canonical writer with only the new task contract, retaining bitwise-identical
actor tensors, normalization/action vectors, policy IDs, and revisions.

One matched nonvisual counterfactual is authorized at held-out seed
`2650443585`: re-bound candidate and incumbent each run current band 1 and
protected band 0 at 64 environments × 5,000 no-reset steps, one repeat, and
chunk 1. The candidate must have zero failed environment steps, zero
non-timeout physical failures, current-band tracking at least its matched
incumbent, no protected-band regression, and mean current-band tilt no more
than the matched incumbent plus `0.005` rad. A pass identifies only an action
allocation mechanism; it authorizes no learner, Stage 2, deployment, replay,
GIF, picture, README media, flight assertion, or Crow-biomechanics claim. A
failure ends this support-phase hypothesis without a gate-shape variant.

### Swing-phase residual allocation outcome — rejected

Source `f9d4880a` first introduced the counterfactual, but its flag reused bit
17 and therefore collided with the previously rejected common-mode task
fingerprint (`11608351420320865043`). The first zero-action artifact is
retained at `.numi/runs/crow-ground-swing-phase-20260824-v1/zero-actions/`,
but is explicitly inadmissible provenance evidence: execution stopped before
either actor was rebound or evaluated. Source `2f1e1c1c57803a2152d490f388b2500e8bd383bb`
assigned the experiment its own bit 18 and the M4 rebuilt it successfully with
`metalrobo_run_program_check`.

The corrected zero-action isolation, at 64 environments × 5,000 no-reset
steps, chunk 1, seed `2650443582`, had task fingerprint
`15631006314005615029`, run fingerprint `10524091870655699159`, zero failed
environment steps, and only 64 authored timeouts. Its state trace was the
required SHA-256
`f9a4dfd48cb3b3f65fee533ee16583849972312e8b70728cdb5127be0ec2110b`;
the evidence and runtime-manifest SHA-256 values were
`7117cef5655ef646ad5d46f7dae43ddc72072baf44843db2a3476d0079e76e24` and
`bc77783aa1830ccf362e661aee9a35ff2dcdeef4ef8f36f1e8b1ccb7a23c7bb5`.

The canonical writer rebound the final candidate and incumbent only to that
new task contract. The actor tensors, normalization/action vectors, policy
ID, and revision were bit-identical before and after: candidate actor SHA-256
`7fd1eaf41c4e327c404a791de41f7f069ea03e3ebb175554b8ec26fc7eed3d9d`
(revision 513) and incumbent actor SHA-256
`1a7a9e1051d7643aa639e28a1d2269727a08f42600450f647fce324f73f27192`
(revision 1). The rebound-policy SHA-256 values were
`438cfb3fee5d1ca2bdba129a105453c4e2925e7361868ea4502dc20bba5ec3a9`
(candidate) and
`7676df35bc96bd51730257cb7380d92a617706a943e8390482b5e7cee2966456`
(incumbent).

On the one authorized held-out seed `2650443585`, all four 64-environment,
5,000-step, no-reset, chunk-1 M4 runs had zero failed environment steps and
only 64 authored timeouts. In current band 1, the candidate's tracking was
`0.9022161313` versus the incumbent's `0.9119821667`, and its mean tilt was
`0.0131824957` rad versus `0.0039168835` rad. The registered tilt ceiling was
therefore `0.0089168835` rad, which the candidate exceeded by
`0.0042656122` rad; it also failed the tracking gate. The protected band 0
comparison was effectively unchanged (candidate / incumbent tracking
`0.9999999866` / `0.9999999867`, mean tilt `0.0005083765` /
`0.0005041855` rad), but that does not waive the current-band failure.

The four evidence SHA-256 values, in candidate-band-1, incumbent-band-1,
candidate-band-0, incumbent-band-0 order, are
`1188671d0aa1f94d7e6577029deb15fd0566f514a3177d2e3bc8eaff9118f00d`,
`545203da0314c4d9d1c453d956df379758163e62051ed62e157793f68f11e3f5`,
`f20fa20832c04a2d01c9a40a728ae9d7c2284b05f806f87b3f34d1f14740ef2d`,
and `e3e070ac97f14d1daf7c90d93a5cd65f4a7388d9a8120cfcd5dcf6542522d04b`.
The matching state traces are retained under
`.numi/runs/crow-ground-swing-phase-20260824-v1/` on the M4. The temporal
support-phase hypothesis is rejected, so the action gate is removed from the
active task. No gate-shape variant, learner run, Stage 2, deployment, replay,
GIF, picture, README media, standing-to-flight claim, or Crow-biomechanics
claim is authorized from this evidence.

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
The unsent acquisition and acceptance specification is maintained in
[American-crow kinematic data intake](AMERICAN_CROW_KINEMATIC_DATA_INTAKE.md).

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
