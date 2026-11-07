# American crow RL environment

## Objective and ownership

The primary research outcome is lifelong embodied intelligence: foraging,
memory, manipulation, and social learning. Develop the existing American crow
into an embodied, persistent ecology with
active perception, locomotion, manipulation, finite food and water, internal
needs, memory, and social learning. Metal owns accepted physics, contacts,
observations, task state, randomness, and policy inference. Swift owns bounded
submission and publication. MLX learns from accepted batches. Python and shell
may author inputs and orchestrate complete native executions; they must never
introduce a second per-step simulator or an alternate physics authority.

The selected source is the existing `agent/crow-world-model-navigation` track
at `577c54ff`, isolated on macmini. Its historical v8/v9 qualification is not
current-checkout qualification. Preserve the original BirdFlow estimated crow,
artifacts, source history, and every negative result. Do not silently migrate
legacy PolicyPacks to new sensor, action, world, or task semantics.

## Live inspection on 7 November 2026

- Execution host: macmini, Apple M4 Pro, 16 GPU cores, 24 GiB RAM.
- Existing crow: 14 articulated bodies, 13 joints, 19 generalized velocities,
  15 journey actions; the head and beak have no independent physical controls.
- V9: 684 actor inputs, including four 16x9 masked depth frames and derived
  features. V10 currently has 807 actor inputs including privileged route
  targets and isolated stage adapters.
- The authored navigation camera is attached to `crow_body` at
  `[-0.85, -0.85, 0.45] m`. This is an offset body camera, not a crow eye.
- Two body-wrench action channels and time-conditioned motion carriers remain
  in the hybrid. They must remain explicit and cannot establish muscular or
  aerodynamic realism.
- Historical full-route development results are far below the retained 90%
  reliability gate. Isolated-stage success and model-predicted return do not
  qualify autonomous behavior.
- The shared MLX environment imported MLX but lacked NumPy. Use a dedicated,
  dependency-locked learner environment and import the actual worker at
  preflight; importing `mlx.core` alone is insufficient.

## Executable engineering boundary

From this source checkout, build `metalrobo_task_rollout`,
`metalrobo_task_train`, `metalrobo_robot_catalog`, and
`metalrobo_crow_embodiment_check`, `metalrobo_crow_foraging_check`, and
`metalrobo_crow_eye_check` and `metalrobo_retinal_stream_check` with
CMake/Ninja Release.
Create the learner with CPython 3.13.14 using `uv venv python/.venv`, then
`uv pip install --python python/.venv/bin/python -c
profiles/crow-rl-requirements.txt -e ./python`. On macmini this checkout uses
a local `python/.venv` symlink to its dedicated external environment. The
symlink is ignored; the dependency constraints and source remain portable.
Run the capability through `tools/numi crow-env`.

`numi crow-env doctor` inspects actual executables, native library, metallib,
learner import, hardware, and disk headroom. `probe --output NEW_DIRECTORY`
records immutable input identities before bounded native runs, tests exact
policy/observation/transition replay across repeated and differently chunked
execution, and measures visual rollout scaling. It does not qualify a learned
motor policy, biological fidelity, end-to-end training speed, or ecology.

`bootstrap --source-policy PACK --variant v9-visual-neural|v10-world-model
--output NEW_DIRECTORY` explicitly transfers a retained actor into the current
fingerprinted task. It retains the source hash and worker output. New columns
are inserted at actor index 84 with zero weights; the critic is fresh. The
result has no new training updates and is an unqualified initialization.

`learn-probe --source-policy PACK --output NEW_DIRECTORY` performs two native
PPO updates and one resumed update. It requires every expected policy
revision to be published and the learner state to report restored. It records
both training-loop and whole-process throughput. These short runs cannot
qualify policy quality, sustained throughput, or complete world restoration.
`motor-probe --policy PACK --output NEW_DIRECTORY` diagnoses seven milestones
on one development seed. `evaluate --policy PACK --mode development` runs the
full v10 route on three development seeds; explicit `--mode qualification`
uses the separate seed namespace and three geometry splits. Navigation
admission checks every lane and rejects inconsistent or truncated evidence.

The rollout observer publishes `first_episode_navigation`. Each environment
contributes only its first episode since context creation. Submission chunks
and benchmark repeats do not create new experimental units. Physical failure
closes the first attempt permanently. A later successful retry cannot repair
it. Reaching a waypoint and subsequently crashing is reported separately from
safe completion. Unfinished horizons are explicitly censored. This observer
does not alter physics, reset behavior, RNG, actions, rewards, or policy ABI.

The separate policy-sampling correction removes the submission-local buffer
index from the random key in both ordinary and residual policy kernels.
Samples are now keyed by seed, environment, episode, accepted episode step,
policy revision and action. Historical chunk-one random streams are preserved.
Old results from larger chunks are not retroactively qualified by this fix.

## Experimental active-head embodiment

`american_crow_active_head_estimated_v1` is a distinct native RobotPack with
17 bodies, 16 joints and 22 generalized velocities. It adds neck yaw, neck
pitch, and lower-beak articulation, plus separate head and upper/lower beak
collision proxies. New joint ranges, effort limits, masses and dimensions are
prospective engineering parameters; none is a same-specimen measurement.

The neck/head/beak partition takes 40 g from the original lumped torso. It
shifts the torso COM frame and transforms inherited root anchors, collision
offsets and aerodynamic application points. The full neutral spatial inertia
and gravitational first moment are preserved, including the torso's required
off-diagonal inertia. The new DOFs permit actual physical relative motion.
Torso collision geometry is revised to avoid a duplicate lumped head. The
original feathered presentation mesh still needs a matching articulated asset
binding; the experimental pack is not exposed as a qualified visual creature.

`head-admission --output NEW_DIRECTORY` executes 36 FP64 reference pose/twist
controls and a 12-environment Metal ABA batch. It compares independent
accelerations/configurations, exercises each new joint through ordinary
effort, and requires exact replay. This is single-step mechanical admission.
It does not test support/contact behavior, perception, sustained control,
ecological tasks, biological realism, or a complete native life checkpoint.

`v11-embodied` binds that robot to a distinct development task: 93 actor and
critic inputs and 18 actions. Its nine additional inputs expose previous head
actions, joint positions, and velocities. These are proprioception; the task
does not yet have eye images. Fresh training starts with a zero-output actor
and no teacher. Unmatched feathered capture/window presentation is rejected.

The generic joint-position actuator can explicitly request posture authority
independent of the old journey's standing/liftoff action masks. Only the three
new head actions request this flag. Invalid flags fail compilation without
replacing the accepted run. Task-program ABI 45 makes changed actuator
semantics explicit; old PolicyPacks require an explicit transfer and remain
unqualified.

`body-admission --output NEW_DIRECTORY` compares neutral and commanded native
contact-world executions, with eight environments and one traced lane over
2.56 seconds. Head yaw, pitch and beak commands must produce physical relative
motion without physical terminations or device failures. This caught a real
failure: the inherited standing curriculum suppressed all three head commands.
The repaired control passes. It is a bounded contact/actuation control, not
proof of sustained locomotion or manipulation.

`learn-probe --variant v11-embodied --output NEW_DIRECTORY` runs two PPO updates
and a resumed update under the new contract. The retained development probe
accepted all 196,608 transitions and published all updates. Resumed training
restored learner state and measured about 30,102 completed transitions/s over
the entire resumed process. This is a short proprioceptive probe; compare it
only with identical sensing, batch and rollout semantics. No full world state
restore or policy qualification is implied. `partition-probe --variant
v11-embodied --policy PACK --base-policy PACK --output NEW_DIRECTORY` checks
ordinary and residual stochastic replay across submission chunks 1 and 8.

## Finite nutritional intake ledger

`FiniteResourceContext` is a native MetalWorld post-commit extension. It
reads accepted physical mouth and source-body poses and the physical jaw
coordinate; it never writes mechanics, contacts, or actions. A bite requires
an open–close gesture whose closing edge occurs inside an authored intake
volume. Holding closed cannot repeat a bite. A missed close consumes the
gesture. Overlapping volumes have stable authored priority and only one debit.
Integer micrograms transfer from a finite patch into a capacity-limited crop.
No nutritional payload is created, including on the final partial bite.

These payloads are nutritional tracers in intake volumes, not resolved solid
food grains. Crop payload does not yet alter body mass or undergo swallowing
or digestion. Water, physiology, pickup/carry/cache mechanics and feeding
reward integration remain pending. The module has no alternate per-step host
simulation; its encoding callback schedules the compiled Metal kernel on the
existing world command buffer. It rejects a different world identity, checks
accepted physics status, and freezes the ledger on a rejected physical step.

`foraging-admission --output NEW_DIRECTORY` runs the native control with eight
canonical starts, 448 steps / 8.96 seconds per trial, and four physical trials.
The retained control consumes exactly 45,000 micrograms in three bites in
every lane. Turning away misses the volume and consumes zero. Chunks 8 and 64
give identical complete resource state. An ordinary task reset cannot refill
or erase resources. Synthetic GPU controls independently exercise rejection,
overlap, held closure, finite exhaustion, crop capacity and restored continuation.
These canonical starts admit the transaction; randomized search is a separate
behavioral experiment. The earlier randomized fixture's misses are retained.

Resource-only checkpoints bind the compiled program and world, use SHA-256,
preserve the integer lifetime clock and jaw latch, and reject truncated,
corrupted, mismatched, trailing or conservation-invalid data transactionally.
They do not restore the surrounding world, contacts, policy, sensory history
or memory. A complete life checkpoint is still pending.

## Native feeding-station learning task

`v12-feeding-station` connects the ledger to accepted native sensing, reward,
PPO and publication. It has 97 actor/critic inputs and 18 action channels.
The four nutritional inputs are crop fullness, newly consumed payload,
mouth contact with an available intake volume, and local tracer availability.
Availability is zero outside that volume. No remote patch coordinates or
eye images enter this contract. These are estimated intake/taste proxies.

The station is a canonical grounded mouth-skill curriculum. The inherited
standing mask holds the other controls while three independent head/jaw
actions remain active. It is not a free-locomotion or food-search task.
There is no teacher, flight-stage schedule, reward for holding closed, or
infinite food. Forty-five milligrams in the initial patch can yield at most
4.5 utility units over the entire lifetime. Reward is accepted kg/s times
100,000 integrated over the actual control interval. Ordinary task resets
preserve food and crop. A reset clears the jaw edge without crediting a bite;
both synthetic and native reset-during-closure controls exercise this case.

Task ABI 47 adds nutrition state and the immutable state-program identity.
Resource ABI 3 binds gesture/reset semantics. Geometry, quantity, bite size,
crop capacity, mouth thresholds and projection semantics participate in the
policy's task fingerprint. A different runtime resource program is rejected.
TaskPack persists this identity in the optional tagged `NUTRI001` extension;
legacy packs without an extension retain their old bytes and a zero identity.
Old readers reject the new extension. This does not silently transfer old
PolicyPacks or make resource-only checkpoints into complete world checkpoints.

`feeding-probe --output NEW_DIRECTORY` uses authored native action streams
to verify all connected C/Swift/Metal observations, rewards and outcomes.
The two chunk sizes each produce 4.5 utility per lane. Turning away and holding
closed produce zero; every control has zero physical terminations and failed
physics steps. `foraging-admission` now additionally checks five native trials,
including reset during a closing gesture, TaskPack identity preservation and
runtime rejection of a changed food program.

The default 0.2 exploration standard deviation produced no intake reward in
the reported PPO updates, so that run is only a pipeline control. Wider
unit-standard-deviation exploration with fixed learning rate 0.0001 produced
positive accepted food reward. The first wider-exploration resume rejected a
configuration mismatch; the corrected probe preserves the same exploration
configuration when restoring learner state. Both negatives are retained.

The corrected probe accepted 196,608 transitions and published revisions
1 to 3 to 4. Its final initial-process update had mean food utility 0.0003052
per transition; the resumed update had mean 0.0040665 and maximum about 2.
These numbers are evidence of real food debits, not evidence of behavioral
improvement: the resumed process starts a new independent world with fresh
finite resources. It restores the learner only. Whole-process throughput was
about 18,260 and 19,391 transitions/s, with MLX peak memory about 458 MB.
These are short proprioceptive probes, not sustained ecological performance.

Run this probe explicitly with `learn-probe --variant v12-feeding-station
--initial-log-standard-deviation 0 --fixed-learning-rate --require-intake-reward
--output NEW_DIRECTORY`. `feeding-policy-probe --policy PACK --output
NEW_DIRECTORY` screens a deployed policy on three development seeds without
spending qualification seeds. The default requires a deterministic PolicyPack;
`--stochastic` explicitly selects a stochastic pack. A mode mismatch is rejected
before native execution. The final native resource ledger now publishes exact
per-lane remaining and consumed micrograms, crop payload, bite counts and
accepted control ticks. These must conserve 45,000 micrograms and agree with
reward and the typed intake outcome. Missing lanes and invalid clocks are errors.
The three-update deployment screen consumed zero food on all three development
seeds, with zero physical terminations and device failures. This candidate is
not a learned feeding skill. Longer primitive training and temporal exploration
remain open; positive stochastic training reward alone cannot qualify deployment.

## Frozen learned feeding comparison

The bounded development campaign uses independent finite lifetimes, with policy
and Adam state carried between processes. Food remains depleted across four
PPO updates inside each lifetime. It is not a complete life-state restore.
Each update has 512 environments and 128 steps, four PPO epochs, minibatches
of 8,192, fixed learning rate 0.0001 and initially unit exploration deviation.
Swift retains compact metrics for every update, allowing whole-lifetime food
budgets and revisions to be checked instead of inspecting only the last batch.

The first campaign was interrupted after eight completed lifetimes because its
registered deterministic baseline accidentally used a stochastic training pack.
The incomplete ninth lifetime and all earlier evidence remain retained. The
original 32-lifetime prediction is inconclusive; it was not silently completed,
extended or repaired after observing results. The eight complete lifetimes
contain 2,097,152 accepted training transitions and revisions 1 through 33.
They ran at approximately 18,700–18,900 whole-process transitions/s, with about
458 MB MLX peak allocation. These are proprioceptive fixed-station measurements.

The follow-up froze the initial and eighth-lifetime policies without further
learning. A prospective comparison tested each actor in deterministic and
stochastic modes on three unused development seeds, 512 lanes each, for
1,600 steps / 32 s. Each mode pair was checked for identical actor coefficients,
normalization, action transforms and task identities. Policy mode and learned
standard deviations are retained in the evidence. Policy revision participates
in the RNG key, so equal seeds do not imply identical noise between revisions.

| Frozen policy mode | Fully depleted 45 mg, by seed | Total |
| --- | --- | --- |
| Initial deterministic | 0, 0, 0 / 512 | 0 / 1,536 |
| Initial stochastic | 7, 7, 6 / 512 | 20 / 1,536 |
| Trained deterministic | 0, 0, 0 / 512 | 0 / 1,536 |
| Trained stochastic | 512, 512, 512 / 512 | 1,536 / 1,536 |

All twelve runs had zero physical terminations and device failures. Direct
integer inventory, crop, bites and lifetime clock agreed with reward in every
lane. The registered prediction required at least 95% complete lanes for the
trained stochastic policy and at most 30% for the initial stochastic policy on
each seed; it was supported. This is one independently trained policy and 6,144
test trajectories, not thousands of independently trained animals.

The retained candidate therefore provides a learned stochastic feeding primitive
at a canonical supported intake station. Deterministic execution does not feed.
The patch is a nutritional intake-volume tracer, not solid food; the body and
head are estimated, other motor controls are held by the standing curriculum,
and there are no head-eye images, search, carrying, swallowing or learned memory.
The complete ecological environment remains unqualified. Original navigation
qualification seeds and layouts remain unspent.

`feeding-policy-comparison` records all four frozen policy inputs, calibration
bindings, rotated arm order, raw per-lane evidence and analysis. Native C API
controls additionally reject absent/uninitialized resources and wrong output
sizes without mutating caller memory, and require repeated readback to be pure.
The first readback export had a Swift JSON type-inference error; that failed
instrument trial is retained separately from the corrected calibration.

## Implementation order and acceptance

1. Restore and measure the native motor instrument. Check exact replay,
   reset/termination semantics, visual cadence, full learner imports, memory,
   and actual learning throughput. Requalify motor behaviors on current code.
2. Add physically articulated head/neck and beak mechanics in a new RobotPack.
   Audit mass/inertia and joint drives independently. Mount sensory origins on
   the accepted head pose; test occlusion and active looking. Bind a new policy
   contract. Keep the old camera and policies only as labeled controls.
3. Author a compact habitat in the existing ScenePack/WorldProgram. Start with
   a clearing, varied perches, water, finite food patches, and manipulable
   objects. Contact geometry and render geometry must agree. Use the same
   solver for travel, support, carrying, and object contact.
4. Compile a generic finite-resource/task-state program. Consumption requires
   accepted beak/object geometry and a valid action/contact sequence. Debits and
   credits commit with accepted state; rejection rolls them back. Food cannot
   duplicate through pickup, dropping, resets, restore, or failed transactions.
5. Add estimated energy, hydration, fatigue, digestion and rest with an
   explicit SI timebase and uncertainty. Use integrated physiological utility
   and costs; distinguish temporary shaping from the deployed objective. Test
   inactivity, overeating, drop/pickup cycles, and premature termination.
6. Add persistent memory and complete world/policy/RNG checkpoints. Restore
   exactly through observation histories, contacts, resources, identities, and
   internal state. Benchmark delayed retrieval and changed food availability.
7. Add individually identified companions with separate observations and
   knowledge. Information travels only through visible behavior and sound.
   Test informed versus uninformed demonstrators and occluded receivers.
8. Expand weather, time of day and seasonal availability only after the smaller
   habitat closes its physical, resource, behavioral, and performance gates.

Every stage retains a runnable accepted parent and introduces one testable
mechanism. Failure does not authorize relaxing a threshold, promoting an
isolated curriculum result, or calling a pending module implemented.

## Performance and qualification

Profile native rollout, visual sensing, learner update, publication, and
checkpoint I/O separately, then report end-to-end completed learning
transitions/s. Record retained native bytes, transient/private bytes, MLX peak
bytes, swap delta, physics failures, and wall time. Optimize the measured
dominant stage while holding sensor, reset, random-stream, task, and physics
semantics fixed. Select batch size from measured throughput and memory.

`profiles/american-crow-rl-v1.json` separates development/profiling seeds from
prospective qualification seeds. A flight-course gate requires at least 28/32
safe first-attempt completions on every seed and at least 90% in aggregate,
with zero failed physics steps, zero teacher authority, fixed full-route band,
and no curriculum or scheduled resets. Waypoint proximity is not supported
landing; the latter needs independently measured foot/perch support and hold.

The complete objective remains open until every requested ecological module
is executable, exercised, and qualified at its stated scope. The crow remains
an estimated simulation model unless independent measurements establish a
stronger biological claim.


## Head-mounted geometric camera instrument

`eye-admission --output NEW_DIRECTORY` runs two native cameras on the accepted
physical head, using exact collision-box meshes and an explicitly bounded
16 m plane proxy. It changes no physics geometry or control semantics. Both
parallel pinhole cameras have estimated engineering mounts, 160 x 120 pixels,
45 degree vertical field of view, and 0.1 mm depth quantization. These fixtures
are not measured American-crow eyes; the original feathered asset remains
unadmitted for the split-head embodiment.

The five native controls cover neutral, physical occlusion, opposite head yaw,
head pitch, and chunk 1 versus 8. Every camera also repeats its render exactly.
An independent FP64 ray/box and ray/plane oracle uses accepted body states and
camera mounts, excluding one-pixel silhouettes. The extended admission passed
in 20261107-head-camera-admission-04: zero identity/depth-validity errors across
198,286 stable geometry pixels and 92,374 stable empty pixels; maximum metric
depth error 55.83 micrometres versus the prospective 200 micrometre threshold;
maximum camera pose component error 8.51e-8 versus 2e-6. The neutral target has
360 pixels per camera/lane and disappears under occlusion and averted yaw.
Seven malformed-authoring cases are rejected. This is geometric instrument
calibration on two lanes, not a policy or ecological result.

Two earlier attempts failed asset validation before any rendering: a missing
vertex tangent sign, then incorrect primitive-to-instance ownership. Attempt
03 passed the original geometric prediction; attempt 04 prospectively added
empty-ray and malformed-input controls without relaxing thresholds. All attempts,
raw float images, depth, identity/validity planes, accepted poses, binary layout
metadata, and native PNG previews are retained. Raw RGB is linear; exported PNGs
apply the sRGB display transform. CTest `metalrobo.crow.head_camera` exercises
the same bounded native controls and retains its own output directory.

The head-camera probe currently publishes frames for inspection. It does not
feed either image to the v12 actor, enforce the authored 15 Hz cadence, or model
sensor latency. The next admission is a device-resident causal retinal stream,
then a distinct task contract and learned search. Existing v12 policies retain
their own 97-input contract. New trials now retain the exact tracked source
patch plus untracked runtime source, rather than hashes alone.


## Device-resident retinal timing layer

`retina-admission --output NEW_DIRECTORY` calibrates a native RGB stream
independently of the renderer and learner. `RetinalStreamContext` borrows a
caller-owned Metal command buffer, native task episode clock, one linear RGB
buffer per eye, and actor output storage. It encodes area reduction, rational
sample cadence, delayed delivery and publication; no host readback or per-step
host sensor decision occurs. It reads only episode/initialization fields from
the task state, and accepts no depth, segmentation, identity or world transforms.
It never writes physics, the task state or another actor feature.

The development timing is 3/10 samples per 50 Hz accepted control tick (15 Hz)
with a two-tick delay (40 ms). A fixed channel response c/(1+c) follows exact
area averaging. Held pixels carry explicit valid and age-in-seconds features.
A repeated accepted tick is idempotent even if the input image changes.
Per-lane episode changes clear visible and pending frames; the delayed stream
stays invalid until the new frame is ready. Clock gaps/backward clocks and
nonfinite, negative or overflowed captured radiance fail closed per lane.
Errors remain latched; callers must inspect native error publication before
accepting an experiment. Concurrent command buffers cannot mutate one stream.
A failed Metal command poisons the context. Snapshot is an inspection boundary,
not a sensory/world checkpoint or a restore API.

The controlled native GPU instrument passed in retinal-timing-admission-02:
three lanes, fifty accepted ticks with a duplicated seventh tick, and a reset
in one lane. Chunk 1/8 streams and final states are bit-identical. All features
and guards agree with an independent FP64 analytic area/timing oracle within
4.77e-8 (registered limit 3e-6), including sample ratios 1/1 and 1/50 and delay
boundaries 0, 1 and 16 ticks. Eight invalid configurations and null/short/missing
input controls reject. Clock gaps, nonfinite RGB and finite radiance reduction
overflow invalidate the affected stream without crossing lanes. Attempt 01
passed the original controls; attempt 02 prospectively added the boundaries
and overflow negative with unchanged numerical thresholds. Attempt 03 also
rejects source-image index overflow and retains raw output before assertions.

These are synthetic native timing fixtures. No v12 actor consumes these RGB
features yet, and the native renderer has not been composed with this stream
inside the accepted rollout/inference command buffer. The next gate must bind
the full sensor semantics/content to a distinct actor contract, compose both
physical head cameras without CPU image round trips, test visibility and
cadence causally, and measure complete rollout/learning performance. The
separate additive RetinalStream.metallib leaves the established physics/policy
metallib unchanged. No biological eye timing or acuity is claimed.
