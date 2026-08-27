# Numi Crow neural-control roadmap

This roadmap separates implemented runtime boundaries from trained or
qualified behavior. The bird is an estimated American-crow hybrid. Nothing in
this document is measured animal flight or hardware-flight evidence.

## Implemented stack

| Level | Runtime contract | Current evidence |
|---|---|---|
| v7 hierarchical | 15-action, 84-observation universal journey actor plus state-triggered approach-pitch supervisor | Existing qualified incumbent; supervisor retains actuator authority |
| v8 neural-only | Same 15/84 policy ABI; supervisor removed; warning/full pitch envelopes are diagnostic outcomes only | Compile, CLI, Metal rollout, replay export; no promoted neural policy |
| v9 visual neural | v8 dynamics and authority plus four 16x9 masked-depth frames and 24 derived sensor features; 684 actor inputs, 84 critic inputs | Compile and one-update Metal/MLX smoke; smoke candidate rejected by protected-band selection |

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
The journey teacher's pitch-moment label uses accepted root pitch and
body-frame pitch rate with the previously qualified proportional-and-rate
target. This remains an invocation-scoped action label executed through the
ordinary bounded body-moment actuator; it adds no deployment supervisor.
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
deployment and protects every earlier band.

Sensor-fast curriculum uses `NUMI_CROW_COURSE=sensor-fast` and automatically
selects the v9 visual ABI and its authored camera. Begin it at band 0 from a
promoted v8 parent via `NUMI_CROW_PARENT_POLICY`, without setting
`NUMI_CROW_PARENT_STATE`. The supervisor imports that actor into the larger
v9 observation ABI with zero-weight visual inputs and a fresh critic, then
uses full PolicyPack plus learner-state resume only between promoted v9 rungs.
`NUMI_CROW_PARENT_MODE=actor-transfer` makes this boundary explicit. Do not
transfer a rejected smoke pack or skip protected v9 milestones.

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

The next genuine breakthrough is therefore not a larger network. It is a v8
policy that clears all protected state milestones without supervisor authority,
followed by a v9 transfer that retains those milestones while improving a
pre-registered visual obstacle/perch task. Only then should online adaptation,
world models, or growable modules enter the deployed PolicyPack ABI.
