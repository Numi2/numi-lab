# Numi Crow neural-control roadmap

This roadmap separates implemented runtime boundaries from trained or
qualified behavior. The bird is an estimated American-crow hybrid. Nothing in
this document is measured animal flight or hardware-flight evidence.

## Implemented stack

| Level | Runtime contract | Current evidence |
|---|---|---|
| v7 hierarchical | 15-action, 84-observation universal journey actor plus state-triggered approach-pitch supervisor | Existing qualified incumbent; supervisor retains actuator authority |
| v8 neural-only | Same 15/84 policy ABI; supervisor removed; warning/full pitch envelopes are diagnostic outcomes only | Promoted through all 11 protected milestones; independent 33-run qualification and accepted replay complete |
| v9 visual neural | v8 dynamics and authority plus four 16x9 masked-depth frames and 24 derived sensor features; 684 actor inputs, 84 critic inputs | Actor-transferred only from promoted v8, then promoted through all 11 protected milestones; independent 33-run qualification and accepted replay complete |
| v10 world-model navigation | Current development ABI has 697 actor inputs: v9, eight route values, and a five-way stage one-hot; plus a collidable gate/slalom/perch course, deterministic layout splits, accepted RGB-D replay, MLX latent dynamics, trust-region CEM planning, a frozen-base stage adapter, and a fingerprinted turn-preview yaw reference | The historical 684-input candidate passed its original 18-run comparison. The current five-waypoint learner progresses beyond waypoint one but is not promoted because it fails the predeclared completion gate; see [Numi Crow v10 world-model navigation](NUMI_CROW_WORLD_MODEL_NAVIGATION.md) |

Every variant has a different task, observation, run, and PolicyPack
fingerprint. A state-only pack cannot load as a visual policy. V8 and v9
promotion additionally require approach warning occupancy at or below 0.05 and
zero full-envelope occupancy on approach, touchdown, landed hold, and full
journey bands.

`CrowReplayPack` (`numi.crow-replay.v1`) exports accepted `q`, `v`, composed
body poses, accepted actions, transition metrics, outcomes, and fingerprints
at one-step submission cadence. Its canonical payload is SHA-256 locked.
BirdFlow validates that lock and renders every camera from the same replay.
Root lift gates the standing-to-flight surface handoff, so a failed takeoff
cannot become presentation flight merely because render time advanced. Feather
deformation remains an estimated high-detail retarget, not the native Numi
sensor image or a claim of exact biological plumage motion.

## Completed neural qualification on 28 August 2026

The state-only v8 curriculum promoted a neural-only policy through all eleven
milestones with every earlier band re-evaluated after each promotion. Its
independent final matrix contains 33 autonomous runs: eleven fixed bands at
three seeds, 32 environments and 1,600 steps per run. Across 1,056 environment
lanes it recorded zero failed environment steps and zero non-timeout
terminations. The minimum run-level mean tracking score is `0.6813128`, the
largest run-level mean tilt is `0.0896696 rad`, the global maximum tilt is
`0.2198471 rad`, and the largest root height is `0.9389769 m`. The deployed
PolicyPack SHA-256 is
`072d842d60a9a4291f2c52d0bb07770702e9026ae6e0d068f721486547def58b`.
Its accepted-state replay SHA-256 is
`740c6e801322af25308a524d5a0d60357c60ca03517dbdd59d94e2600a475cc3`;
the canonical replay payload SHA-256 is
`a7ae9a76ae6fc802ff0d831c88a0eecca45443e97a0128654900111adbf7d0b8`.

V9 began by actor-transferring that promoted v8 policy into the 684-input
sensor ABI with zero-weight visual columns and a fresh critic. Rejected v8 and
v9 candidates were never parents. The final v9 candidate is a fingerprint- and
ABI-exact neural actor assembly: it retains every hidden layer and 14 of 15
output rows from promoted v9 revision 2621, while action row 6 (tail pitch)
comes from v9 revision 2721. The 512-environment selector evaluated the merged
candidate against the incumbent on full journey and protected bands 0-9 and
promoted it without an absolute-contract regression. This output-row merge is
recorded exactly; it is not a teacher, runtime supervisor, or non-neural
actuator path.

The independent v9 final matrix repeats the same 33-run, three-seed contract.
It recorded zero failed environment steps and zero non-timeout terminations;
minimum run-level mean tracking is `0.6708923`, largest run-level mean tilt is
`0.0863282 rad`, global maximum tilt is `0.1825500 rad`, and largest root height
is `1.3927420 m`. The deployed PolicyPack SHA-256 is
`e444223ef9867e8b6323cdb7e8e5030106ab8ef6a7e39a3623ee46d34d2e7bcb`.
Its accepted-state replay SHA-256 is
`6ffab1a5b04b977c5b3c2876b452dfd4895cbc259d231350ad9738b014182554`;
the canonical replay payload SHA-256 is
`6100dd08e2abecbda833f7a3761cda04c8fb5e36e0ba75ad2bde97b77046a881`.

These results qualify the two simulated milestone contracts at the recorded
seeds and revisions. They do not establish measured-crow aerodynamics,
biological fidelity, robustness outside the authored reset distribution, or
hardware flight.

V10 now implements the next obstacle/perch gate described below. Its retained
candidate reduced forbidden-contact terminations on the training course and
both held-out geometry splits while retaining the declared height, tracking,
termination, and tilt envelopes. That narrower result is documented in
[Numi Crow v10 world-model navigation](NUMI_CROW_WORLD_MODEL_NAVIGATION.md);
it does not supersede the v9 all-milestone qualification.

## Curriculum operation

The durable supervisor covers all eleven independently resettable milestones:
standing, walking, takeoff, cruise, takeoff-cruise, left turn, right turn,
approach, touchdown, landed hold, and full journey. Each candidate is evaluated
against its incumbent on the same held-out seed and every earlier protected
band. Rejected candidates remain on disk but never become a later parent.
The native journey teacher remains connected during resumed training by
default so harder rungs retain physically executed imitation targets. It has
zero authority in autonomous held-out evaluation and is never embedded in the
deployment PolicyPack. Training defaults to 0.25 student authority so the
actor encounters states influenced by its own actions instead of learning
only on teacher trajectories; configure this with
`NUMI_CROW_TEACHER_STUDENT_AUTHORITY`. Set
`NUMI_CROW_TEACHER_DISTILLATION=0` and student authority 0 only for a
pre-registered no-teacher ablation.
Checkpoint selection defaults to 50-update spacing and still includes the
final candidate; override it with `NUMI_CROW_CHECKPOINT_INTERVAL` when a
pre-registered experiment needs denser temporal sampling.
When held-out evidence shows repeatable catastrophic forgetting, set
`NUMI_CROW_REHEARSAL_DEPTH` to train on the current band plus that many recent
protected bands. The default is zero so retries do not silently change their
data distribution. Selection remains fixed-band and replays every earlier
milestone regardless of the rehearsal window.
For a stable curriculum floor instead of a sliding window, set
`NUMI_CROW_REHEARSAL_MINIMUM_BAND`; for example, value 2 retains every flight
competency from takeoff through the current milestone while leaving standing
and walking to the independent protected selector.
When the hardest rung is underrepresented in a wide rehearsal range, set
`NUMI_CROW_DIFFICULTY_SAMPLING_EXPONENT` explicitly. Values below one bias
resets toward the maximum band while retaining earlier-band rehearsal; this is
an execution-profile input recorded in evidence and the run fingerprint, not a
TaskPack or PolicyPack ABI change. For bands 2-10, `0.25` assigns about 37.6%
of resets to band 10 instead of about 6.5% under the authored `1.75` exponent.
Balanced retention gives equal total label authority to every protected band
represented in an update. An explicitly prioritized band receives its factor
when represented; if stochastic reset persistence omits it from one update,
the worker balances the remaining represented bands and continues. Promotion
does not inherit that tolerance: the selector still evaluates every protected
band independently and fails closed on any missing or regressed evidence.
The journey teacher's pitch-moment label uses accepted root pitch and
body-frame pitch rate. Because the label passes through the ordinary actuator
response filter before reaching the body moment, its bounded request is
`clamp(-8 * pitch - 2 * pitchRate, -1, 1)`. This is an
invocation-scoped training label, not a deployment supervisor. On the fixed
64-environment full-journey teacher probe at seed `2650443581`, changing only
that request from `-2.4/-0.25` to `-8/-2` reduced warning-envelope occupancy
from 0.290332 to 0.007520 and full-envelope occupancy from 0.116787 to
0.006240. All 64 teacher-controlled environments still terminated, so this
probe supports the label correction but does not qualify the teacher or a
deployable policy; autonomous held-out selection remains authoritative.
At the authored landing boundary for landed-hold and full-journey training,
the already-qualified actor is retained as both carrier and distillation label
rather than being blended toward the flight teacher's neutral action.
V8/V9 also include a fingerprinted reward-only pitch hinge from 18 seconds:
it begins at 0.12 rad, inside the 0.16-rad held-out warning boundary. This
aligns PPO with the selection contract without correcting state or actions.

State-only neural curriculum:

```sh
NUMI_CROW_CURRICULUM_ROOT="$PWD" \
NUMI_CROW_CURRICULUM_BUILD="$PWD/build-crow-journey-ninja" \
NUMI_CROW_CURRICULUM_MLX="/path/to/python-with-mlx" \
NUMI_CROW_CURRICULUM_RUNS="$PWD/.numi/runs/crow-v8-curriculum" \
NUMI_CROW_COURSE=state \
./tools/crow_journey_curriculum_supervisor.sh
```

When an integrated state-only milestone creates a materially new value
distribution, explicitly set `NUMI_CROW_PARENT_MODE=actor-transfer`, provide
the promoted parent PolicyPack, and omit learner state. This retains the actor
but initializes a fresh critic; selection still compares against the source
deployment and protects every earlier band. Use `NUMI_CROW_LEARNING_RATE` to
pin a conservative positive PPO rate when actor retention is more important
than fast adaptation. `NUMI_CROW_INITIAL_LOG_STANDARD_DEVIATION` can likewise
match a transferred actor's exploration head instead of accepting the fresh
learner default. The supervisor records both controls in run arguments.
For a reward-only task migration that must retain already promoted behavior,
set `NUMI_CROW_RETENTION_POLICY` to that promoted PolicyPack, disable the
scripted teacher, and optionally tune the positive
`NUMI_CROW_RETENTION_COEFFICIENT` (default `1.0`). The worker evaluates the
frozen source actor on each current observation and applies the existing
Huber distillation loss without changing which actor controls physics. A
retention source is training regularization, not promotion evidence; the
held-out target and protected-band selector remains authoritative.

Sensor-fast curriculum uses `NUMI_CROW_COURSE=sensor-fast` and automatically
selects the v9 visual ABI and its authored camera. Begin it at band 0 from a
promoted v8 parent via `NUMI_CROW_PARENT_POLICY`, without setting
`NUMI_CROW_PARENT_STATE`. The supervisor imports that actor into the larger
v9 observation ABI with zero-weight visual inputs and a fresh critic, then
uses full PolicyPack plus learner-state resume only between promoted v9 rungs.
`NUMI_CROW_PARENT_MODE=actor-transfer` makes this boundary explicit. Do not
transfer a rejected smoke pack or skip protected v9 milestones.

The sensor-fast visual pack reserves at most 1 GiB of retained renderer memory.
That is a compile ceiling, not a request to allocate the whole budget. A
one-update, one-step, 2,048-environment probe on the 24 GiB Apple M4 Pro at
revision `008bd8f` retained 942,559,086 bytes, reported 809,634,028 transient
private bytes and 71,509,684 MLX peak bytes, completed with zero failed
environment steps, and measured 6,110 environment-steps/s. The earlier 64 MiB
ceiling admitted only 128 environments (64,458,606 retained bytes) and rejected
the authored 2,048-environment curriculum before its first training step. These
numbers qualify capacity for this exact probe; they are not a sustained-run
memory or throughput claim.

After band 10 advances, run the final three-seed matrix and accepted-state
replay export with the promoted deployment PolicyPack:

```sh
NUMI_CROW_QUALIFICATION_ROOT="$PWD" \
NUMI_CROW_QUALIFICATION_BUILD="$PWD/build-crow-journey-ninja" \
NUMI_CROW_QUALIFICATION_POLICY=/path/to/promoted/deployment.policypack \
NUMI_CROW_QUALIFICATION_RUNS="$PWD/.numi/runs/crow-v9-qualification" \
./tools/crow_journey_qualification.sh
```

The command fails closed unless all 33 autonomous runs meet the physical and
milestone gates. It emits `qualification.json`, SHA manifests, and one
`accepted-full-journey.crowreplay.json` for BirdFlow's multi-angle renderer.

## Research choices and breakthrough gates

- High-degree-of-freedom bird-inspired flapping control has demonstrated
  multimodal trajectory tracking with model-free RL in simulation. That
  supports one command-conditioned actor, but not removing per-mode held-out
  gates: [Cai et al., 2024](https://arxiv.org/abs/2411.15130).
- Teacher/student sensor-space locomotion motivates the current privileged
  critic and deployable sensor-history actor split. It does not justify teacher
  actuator authority at deployment: [Khadiv et al., 2023](https://proceedings.mlr.press/v211/khadiv23a/khadiv23a.pdf).
- Visual locomotion work uses a high-level vision policy over a lower-level
  controller. V9 first tests a unified visual actor because the Crow command
  and action ABIs are already compact; a separate high-level planner should be
  added only if obstacle/perch tasks demonstrate a temporal-planning failure:
  [Yu et al., 2022](https://proceedings.mlr.press/v164/yu22a.html).
- Learned robust MPC distilled to a small fast network is compelling for
  aggressive flapping flight, but requires an identified dynamics/tube model
  that this estimated hybrid does not yet possess. Treat it as the next
  model-based branch, not a current capability:
  [Hsiao et al., 2025](https://arxiv.org/abs/2508.03043).
- Continual-control work on growable networks and replay decay suggests a
  future modular PolicyPack format only after the universal actor shows
  capacity interference under matched data and compute. The current milestone
  supervisor supplies the required forgetting evidence first:
  [Kang et al., 2025](https://proceedings.mlr.press/v267/kang25c.html),
  [Malagon et al., 2024](https://proceedings.mlr.press/v235/malagon24a.html).
- Latent world-model MPC is a candidate for perch/obstacle planning after v9
  establishes a stable sensor baseline. It is not a replacement for the native
  physics qualification loop:
  [Hansen et al., 2022](https://proceedings.mlr.press/v162/hansen22a.html),
  [Lin et al., 2026](https://proceedings.mlr.press/v331/lin26a.html).

The next genuine breakthrough is therefore not a larger network. V8 and the
sensor-fast v9 baseline have cleared their authored milestone contracts; the
next gate is a pre-registered visual obstacle/perch task with held-out scene,
lighting, and geometry splits. Only evidence of a temporal-planning or capacity
failure there should motivate a high-level planner, world model, online
adaptation, or growable PolicyPack ABI.
