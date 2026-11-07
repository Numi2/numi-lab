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
`metalrobo_crow_eye_check`, `metalrobo_retinal_stream_check`, and
`metalrobo_crow_retinal_contract_check`, `metalrobo_crow_retinal_live_check`,
`metalrobo_crow_retinal_api_check`, and `metalrobo_crow_life_checkpoint_check` with
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

These timing-only fixtures did not compose the native renderer with an actor.
The separate live v13 instrument below now exercises that composition. No v12
actor consumes these RGB channels. The subsequent gate had to bind
the full sensor semantics/content to a distinct actor contract, compose both
physical head cameras without CPU image round trips, test visibility and
cadence causally, and measure complete rollout/learning performance. The
separate additive RetinalStream.metallib leaves the established physics/policy
metallib unchanged. No biological eye timing or acuity is claimed.


## Typed retinal actor contract and native rejection boundary

The native C++ `makeAmericanCrowRetinalFeedingTaskPack` defines a distinct v13
contract: 1,251 actor features (the 97 existing channels plus two 16 x 12 RGB
planes, valid and age), 97 critic features and 18 actions. It retains the v12
mechanics, action meanings and finite-resource state-program identity. This
factory and the `retinal-contract-admission` instrument are implemented. The
separate live instrument below now supplies real rendered actor images. A v13
C API/Swift rollout/training entry is still pending.

SensorPack owns retinal camera mounts, source and output dimensions, optics,
content hashes, rational cadence, latency, depth range and the fixed response
semantics. Its content fingerprint is bound into both the observation and
task fingerprints. Moving files with identical content preserves identity;
changing a mount, timing or asset content rejects the old policy even with
identical network dimensions. Retinal channels must be one complete, ordered,
untransformed actorCurrent suffix, outside temporal proprioceptive history and
the critic. The native observer must declare the exact observation fingerprint.
Missing or mismatched observers cannot submit a retinal task; a valid generic
callback cannot silently stand in for the required retinal identity.

The shared task structs retain their ABI-47 layout. Retinal tasks receive an
additive executable tag 48; legacy tasks still compile with tag 47 and unchanged
fingerprints. Task kernels explicitly support both tags. A capability exported
from the same task-shader translation unit is checked before submitting tag 48,
so an old metallib is rejected rather than silently returning from its kernels.
The C API structs are unchanged; C++ consumers must rebuild against the changed
native observation-program metadata. The original frozen v12 policy remains
bound to its original task, and must not be dimension-rebound as v13.

`retinal-contract-admission --output NEW_DIRECTORY` checks 12 malformed
manifests, nine same-dimension sensor mutations, exact path relocation and the
native missing-observer, mismatched-identity and stale-shader rejections. A
current-shader probe reaches the declared callback layout, then deliberately
refuses to supply an image; it produces no accepted world submission. Its
fixture asset paths/hashes are compiler metadata only. No real visual assets
are loaded and no actor images are rendered in this instrument. All predictions
passed in retinal-contract-admission-02. Attempt 01 passed the earlier contract
controls; 02 additionally requires the observer fingerprint and tests mismatch.
The actual ABI-47 shader fixture is preserved with its receipt under
artifacts/crow-rl-retinal-contract-20261107/inputs for portable regression.

The contract-only instrument establishes metadata and rejection behavior.
The live instrument below separately admits camera/stream/actor composition.
The distinct v13 contract still needs a C API/Swift learner entry, visual
learning controls and complete measurements at production batch size.


## Live native head cameras, retinal stream and pixel-dependent actor

`tools/numi crow-env retinal-live-admission --output NEW_DIRECTORY` runs six
bounded native controls. `RetinalObserverContext` renders both physical
head cameras directly from the device-resident current body buffer, then
encodes `RetinalStreamContext` in the same caller-owned command buffer before
native inference. It never reads pixels, physics state or task clocks back in
that callback, and never commits or waits there. Only RGB goes to the retinal
reducer; renderer depth, identities and geometry do not enter the actor.

The current observer intentionally accepts a fixed collision-proxy fixture,
neutral studio and fixed area-light rig. It reconstructs the template against
the actual physics model, checks each source-world/content hash, reconstructs
camera bindings and fixed lighting, and validates two physical mounts against
the bound SensorPack. Unsupported variations, optics, noise or exposure reject.
This prevents an arbitrary precompiled scene from impersonating the bound
observer. Eight malformed observer controls reject before native rollout.
An observer owns its images until command completion; concurrent commands
reject and a GPU failure poisons the owner. Callers must inspect the quiescent
retinal error publication before admitting a batch to evidence or learning.

The declared actor has one nonzero rule: yaw is twice red-minus-blue in left
retinal cell [6,8], clamped to [-1,1]. All other controls are zero. Each condition
gets a fresh artifact bound to its exact task/world; these are authored control
actors, not a trained policy transferred across changed worlds. With the
physical target visible, tick-2 yaw is +0.426903; a physical occluder makes it
-0.356149. Disconnecting the coefficients leaves every action zero. The
pixel-dependent action changes accepted physical state. Both eyes supply their
complete RGB features; no semantic target coordinate enters this actor.

Trial 12 passes 384 accepted transitions (six runs, two lanes, 32 ticks), zero
sensor errors or physical terminations, two blind cold-start ticks, integer
15 Hz cadence and 40 ms delay. A lane-1 reset at tick 17 repeats the two blind
ticks without contaminating uninterrupted lane 0. Actor observations, actions,
latents, transitions, full q/v trajectories and final q/v/scene states are
bit-identical across native submission sizes 1 and 8, including the reset.

The two chunk-1 controls explicitly inspect rendered RGB *after completion*
for instrument validation. An independent FP64 area-mean, fixed radiance
response and frame-history oracle checks every delayed RGB feature. Its maximum
error is 9.72599e-8 against a prospectively fixed 3e-6 limit. This inspection
is outside the device callback and is not a production execution requirement.
Raw RGB, actors/actions, trajectories, transitions, reset states, authored
policies, source deltas and failed-attempt receipts are preserved in
artifacts/crow-rl-retinal-live-20261107, with lossless gzip and original hashes
for large raw arrays. Full-world restore is not implemented by these snapshots.

Attempts 01–03 failed setup/capacity admission; 04–05 passed the initial live
controls. Attempt 06 added the RGB oracle and per-lane reset controls but was
stopped at repeated scene construction after a sample showed libdispatch's
64-worker limit blocked in Metal I/O. Attempt 07 showed autorelease cleanup
alone did not fix the stall. The renderer now reuses one bounded serial I/O
loader per device for geometry and textures. Attempt 08 completes all six
cases in 4.10 seconds; 09 adds stricter negative, lane-isolation and trajectory
checks with unchanged behavioral/numerical thresholds. Attempt 10 publishes
contact usage; 11 rejects an undersized refinement at the first substep
(230 raw contacts and 140 blocks required). Attempt 12 retains the original
pair/manifold/convex/CCD envelope and allocates 256 raw contacts, 160 blocks,
480 rows and 320 endpoints, passing all controls. The earlier final-substep
usage report alone was insufficient for substep capacity admission.
This duration includes
construction and instrument readback and is not a training throughput figure.
Apple documents the separate I/O command and command-buffer bounds in its
[MTLIOCommandQueueDescriptor reference](https://developer.apple.com/documentation/metal/mtliocommandqueuedescriptor).

The profile uses 160 x 120 instantaneous parallel pinhole cameras and two
16 x 12 RGB grids. Mounts, acuity and timing remain engineering estimates,
not measured American-crow eyes. Rendering currently occurs every control
tick while GPU sampling admits frames at 15 Hz; optimizing that work and
measuring full batch performance are pending. The current fixed supported
feeding body is not ecological foraging, object manipulation or visual
learning. The full lifelong environment still requires robust locomotion,
solid-food pickup/carry/cache/swallow, water/energy physiology, trainable
persistent memory, trainable memory restore, social agents
and information ablations, and untouched held-out qualification.


## Retinal C API, Swift scheduler and MLX learner admission

The additive `mr_create_task_rollout_retinal` entry now admits v13 through the
regular native rollout and learning tools. Existing C struct sizes and legacy
ABI47 remain unchanged. The new 64-byte retinal configuration has explicit
size/version fields, and the v13 entry requires both visual and retinal
programs. Swift loads `numi.retinal-observation.v1` with explicit relative
visual assets. Missing raw images cannot silently become zero observations.
The critic remains 97 physical/nutritional features; the actor receives 1,251
features including 1,152 RGB values and validity/age. GPU sensor errors reject
the learner publication and require an explicit world reset before reuse.

Terminal policy evaluation now projects the accepted terminal physical state
and fills its retinal actor suffix in the same command buffer before inference.
The ordinary pre-step selector rejects `controlStep == controlStepCount`;
using it for the terminal camera left stale projection input. Trials01–02
exposed chunk-dependent image differences despite exact actions and physics.
A fixed noiseless rendering phase alone did not resolve this. Trial03 switched
to direct accepted-state projection and passed terminal/partition checks.
The explicit fixed-phase optics have a versioned retinal fingerprint, so old
v13 policies cannot be reused silently; legacy v12 fingerprints are unaffected.

Trials03–04 then failed the consumed-food precondition of the reset control.
The first gesture lacked settling; the established 64-tick settle, 32-tick
open and 32-tick neutral-close gesture also failed in04. Inspection found v13
missing the compiler's avian ground-curriculum flag inherited by v12. Trial05
adds that flag to the compiled task identity and passes all controls with the
same gesture: six malformed C configurations reject; terminal features equal
the next chunk's first actor features; complete actor/action/latent/transition
and final q/v streams are bit-identical for chunks 1/8. Explicit world reset
clears the retinal queue (two cold-start blind ticks) without replenishing
food or changing crop/consumption, and the resource clock continues.

Reproduce with a fresh output directory:

```
tools/numi crow-env retinal-api-admission --output NEW_API_TRIAL
tools/numi crow-env learn-probe --variant v13-retinal-feeding \
  --retinal-calibration NEW_API_TRIAL --environments 16 \
  --initial-log-standard-deviation 1 --fixed-learning-rate \
  --output NEW_LEARNER_TRIAL
```

The learning driver requires successful current-runtime API admission and
records exact visual artifact/asset hashes. Probes01–03 used 16, 64 and128
lanes, respectively, each with two initial 128-step updates and one restored
update. All expected samples/revisions publish (1→3→4), with zero failed
native steps. This tests learner-only restoration: each process starts a fresh
physical world. Small intake rewards during exploration do not establish that
the actor learned from images or acquired reliable feeding.

| Lanes | Initial / resumed loop steps/s | Initial / resumed whole-process steps/s |
| --- | --- | --- |
|16|1,405 /1,406|1,116 /1,195|
|64|3,908 /3,806|3,618 /3,478|
|128|5,346 /5,305|5,032 /4,933|

These single bounded engineering probes include native physics, both rendered
eyes, retinal timing, Swift scheduling, PPO and checkpoint publication; process
rates additionally include initialization. They are not a throughput distribution
or a production scaling qualification. Reported MLX peaks range from143 MB to
555 MB; they exclude complete process and renderer memory. Rendering still runs
every control tick even though the GPU admits images at15 Hz. Stage-level
profiling, complete peak-memory accounting and cadence-aware GPU rendering
remain performance work. No qualification seeds were used.

Evidence, all failed API attempts, learner checkpoints, policies, rollout packs,
source deltas and runtime hashes are retained under
`artifacts/crow-rl-retinal-learner-20261107`. At that checkpoint, whole-world restoration was still absent. Native v13 life
restoration is admitted below; persistent trainable memory, physical object manipulation, physiology,
social agents and ecological held-out qualification remain open. The full
lifelong environment is not qualified.


## Native v13 life checkpoint and fresh-process resume

`mr_task_rollout_save_life_checkpoint` and
`mr_task_rollout_restore_life_checkpoint` now bundle the current development
crow's native world, task/random history, delayed retinal queue and finite food.
Save requires successful, quiescent resident execution. Restore requires a fresh,
unadvanced handle with the exact next policy already installed. These methods,
like the rollout scheduler, require exclusive caller ownership; they do not add
concurrent handle mutation support. Swift exposes `saveLifeCheckpoint(to:)` and
`restoreLifeCheckpoint(from:)`. The rollout CLI accepts
`--resume-life-checkpoint FILE` and `--save-life-checkpoint NEW_FILE` with v13,
an explicit policy, `--no-scheduled-resets` and `--repeats 1`. Life execution
skips benchmark warmup: restoring or publishing a policy must not advance the
accepted life clock.

The physical component retains both sides and active selectors of q/v, scene
bodies and contact manifolds; convex caches; reset templates; task state,
evidence, actor/critic and action histories; encoder bias, previous joint
velocity, controller/body parameters and persistent compact contact data.
Task state retains the episode/step keys used by native stochastic sampling.
The sensory component retains pending capture metadata/pixels, visible pixels,
per-lane cadence/clock state and errors. The food component retains inventory,
crop, consumed mass, jaw-gesture state and accepted lifetime ticks. Stateless
render geometry and fixed noiseless optics reconstruct from exact authored
inputs. The checkpoint is taken only at publication boundaries; device
transitions and inference remain native Metal.

The outer `NUMILF01` envelope binds world, task, actor/critic observation,
action, next policy, base policy, resource, seed and lane identities. Nested
world, retinal and food components have exact length and SHA-256 guards.
The physical state excludes stateless policy weights from its execution identity
so a learner may publish its next revision between accepted transitions; the
outer envelope requires that exact next revision. The physical component binds
the loaded native library, core shader, Metal device and OS; the retinal
component binds its shader and sensor configuration. This version supports the
native v13 temporal-cone fixture without rods, muscles, rotors or teacher.
It is an exact-runtime development format, not a cross-version, cross-device
or arbitrary-world migration format. C struct sizes and legacy ABI47 remain
unchanged; the appended C++ context option requires native clients to rebuild.

Restore parses bounded input, validates the complete world/reset/layout/runtime
contract, and restores retinal and food candidates before publishing them to
the fresh owner. The first requested native submission restores persistent
private buffers before its first transition. It performs no synthetic settling
step. Save uses exclusive creation, file fsync and directory fsync; existing or
partial files are retained and cannot be silently overwritten. Checksums detect
corruption; they do not authenticate a deliberately rewritten checkpoint.
Optimizer and trainable weights remain separately retained dependencies.
At this native-only checkpoint, combined optimizer/life training resume was still
absent. The next section admits that separate Swift/MLX integration.

Reproduce with new output directories:

```
tools/numi crow-env life-admission --output NEW_NATIVE_LIFE_TRIAL
tools/numi crow-env life-swift-probe \
  --calibration NEW_NATIVE_LIFE_TRIAL --output NEW_SWIFT_LIFE_TRIAL
```

Trial06 uses two lanes and development seed2651109001 in the supported fixed
collision-proxy scene. After64 neutral settle ticks,32 open and32 close ticks,
both lanes have consumed20,000 micrograms. Seven stochastic image-dependent
policy ticks put the life at135 with capture134 still pending for delivery136.
Revision20 is then published without a physical transition. A saved checkpoint
must reproduce the uninterrupted next32 ticks in two independent processes
using chunks 1/8, including one lane reset at local tick9. Actor features,
actions, latents, transitions, final q/v, food and all three final hidden native
components are byte-identical. Ten malformed or incompatible restore controls
reject without changing published life state; overwrite rejection also passes.
The actor is authored to depend on pixels, not a learned memory policy.

Attempts01–04 are retained. They exposed an incorrect fixture revision,
immutable policy-heap growth clearing the model-upload marker despite valid
resident state, and raw flapping-program hashing including eight nonsemantic
alignment bytes. The fixes use the installed revision, permit an already-bound
world to reupload after immutable storage growth, and hash authored flapping
fields explicitly. Trial05 passes the original controls. Trial06 additionally
covers the next-policy publication boundary. Swift attempt01 exposed a skipped
policy warmup falling into the host-action branch; the corrected branch in02
resumes135→167 with the same chunk1/8 rollout fingerprint and conserved food.

The final regression passes12 native and19 Python checks. Frozen legacy v12
ordinary/residual fingerprints remain2255877855914022940 and
3082487550463979285. A16-lane ordinary Swift/MLX probe also publishes revisions
1→3→4 with zero failed steps and a separately restored learner. That probe still
starts a fresh physical world at the resumed learner stage; it does not establish
combined learner/life restore. No qualification seeds are used.

All six native attempts, both Swift attempts, runtime/source identities,
regression outputs, legacy replay and the small learner probe are retained in
`artifacts/crow-rl-life-checkpoint-20261107`. Lossless compressed payloads have
original/archive hashes. Receipts identify retained matching binaries for the
two successful native versions; earlier failed versions retain source deltas
and runtime hashes, not every historical executable.

Trainable persistent memory, robust locomotion,
physical pickup/carry/cache/swallow, water/energy physiology, social individuals
and controlled information ablations remain open. This closes native state
persistence for the present development fixture; the lifelong ecological
research environment remains unqualified.


## Coupled life and optimizer training checkpoints

`metalrobo_task_train --checkpoint-life` now publishes a
`revision-XXXXXXXX.trainingcheckpoint` directory at each configured checkpoint
interval. It requires `--checkpoint-directory`, `--checkpoint-interval` and an
explicit `--output-learner-state`. Each immutable directory contains the next
stochastic policy, MLX model/Adam state, native life, generating rollout and an
unselected deterministic candidate. Existing legacy snapshot copies are skipped
in this mode. The normal non-life training path retains its original warmup and
learner-only resume behavior.

`--resume-training-checkpoint DIRECTORY` supplies the exact saved policy and
learner inputs; omit `--policy-pack`, `--learner-state` and initialization flags.
Use new output paths, the same v13/task/learner configuration, seeds and rollout
horizon. Update count, submission chunk, checkpoint cadence and output locations
may change. Changed training options reject before learner startup. Native world,
sensors and shaders retain their own exact identity admission. Current life
training excludes teacher/base-policy authority, retention or motion priors,
interactive inspector mutation and resumed learning-rate/exploration overrides.
This is a coherent supported path for the present v13 model, not arbitrary task
migration.

The manifest binds all payload lengths/SHA-256, policy revision, cumulative
optimizer updates, accepted per-lane ticks, scheduler configuration, the exact
Swift executable, imported learner Python dependency closure, interpreter and
MLX/NumPy source/native distribution files. It uses the learner's existing
revision-derived minibatch shuffle; no random-stream reset or optimizer
approximation is introduced. The restored learner must report restored Adam
state, the saved revision and execution identity, and regenerate the exact saved
next PolicyPack before native life restoration. Input-bundle output overwrite
is rejected. Irrelevant experiment-driver edits are excluded from the learner
source closure.

Publication creates a new exclusive directory, copies and fsyncs immutable
payloads, saves the native life, hashes every payload, then writes and fsyncs the
manifest. A SHA-bound `COMMITTED` marker is atomically hard-linked only after
payloads and directory entries are synced. Restore rejects missing/invalid
markers and payloads before starting the learner. Partial directories are
retained. Save/restore still require exclusive scheduler ownership. This checks
an interrupted-directory model and the fsync publication protocol; forced
process-crash and physical power-loss fault injection have not been qualified.
The worker and optimizer remain outside the per-control-step simulation path.

Reproduce with a passed current-runtime retinal API trial and fresh directories:

```
tools/numi crow-env retinal-api-admission --output NEW_API_TRIAL
tools/numi crow-env life-training-probe \
  --retinal-calibration NEW_API_TRIAL --output NEW_TRAINING_LIFE_TRIAL
```

Trial05 uses16 lanes,37 ticks/update, two epochs and minibatches of97, so the
592-sample batch has an uneven final minibatch. One four-update reference is
compared with one fresh two-update run and two independent two-update continuations
at chunks 1/8. At every shared optimizer boundary, complete rollout and policy/
candidate bytes, every model and Adam tensor/metadata, and all three hidden
native components are bit-identical. Lifetime ticks are exactly37,74,111,148;
policy revisions progress1→3→5 across interruption. Lane12 has consumed20,000
micrograms before interruption and retains it afterward, with conserved stock,
crop and accepted clock. This is one learner trajectory plus replay controls,
not independent learned animals or evidence of trainable memory/visual skill.

Nine declared-cause negative controls reject uncommitted bundles, invalid marker,
missing/corrupt optimizer, changed configuration, mixed policy/optimizer,
wrong learner runtime, wrong scheduler executable and input-overwrite attempts.
Attempts01/02 rejected planning and boolean-parser errors before context creation;
03 exposed a legacy telemetry read before completed resident submission. The
fix publishes telemetry only after accepted updates rather than adding a hidden
warmup. Trial04 passes the original seven controls. Trial05 adds explicit clock/
food validation, the extra rejection controls, and imported learner-closure
identity instead of binding unrelated orchestration code. All failures and both
successful runtime/source versions are retained.

The final bounded probe measures16.1–16.5 ms average checkpoint publication per
stage, including native readback, file copies, hashing and fsync. This is a small
warm-cache engineering measurement, not a sustained I/O or throughput distribution.
The12-check native regression and26 Python checks pass. Ordinary learner-only
training reproduces the previous initial/resumed optimizer checkpoint hashes
exactly; frozen legacy feeding ordinary/residual fingerprints remain
2255877855914022940 and3082487550463979285. Both retained successful training
runtimes independently resume their original saved life and reproduce both future
optimizer boundaries. Exact-runtime recovery still requires matching installed
interpreter/MLX/NumPy dependencies; native and Python-source receipts are retained.

Evidence is versioned in `artifacts/crow-rl-training-life-20261107`, including
failed attempts, all update-boundary bundles, generating rollouts, raw regression
outputs, source deltas, lossless payload hashes and retained-runtime replay
receipts. No qualification seeds are used or candidate promoted. Persistent
trainable memory, solid-food manipulation, physiology, social agents, robust
locomotion, production scaling and untouched ecological qualification remain
required for the full lifelong environment.


## Native working-memory component

`GRUMemoryContext` now provides an optional GPU-resident reset-after GRU with
row-major r/z/n input/recurrent weights and separate biases. `gru_memory.py`
provides matching differentiable MLX cell and time-major sequence algebra. This
was first admitted as a prerequisite component in the experiment below. The
later native actor integration is documented in the following section. Recurrent
rollout publication and PPO remain pending; the trained feeding actor remains
feedforward.

The native operator borrows the real task clock read-only and encodes into its
owner's uncommitted command buffer. It retains two private hidden vectors: the
pre-cell state and the proposed next state. Accepted task-clock advancement
promotes the proposal; repeating inference on the same tick recomputes from the
same pre-cell state. Read-only previews select the appropriate state without
advancing it. Episode changes clear volatile working memory. Duplicate inference
and same-clock previews preserve the proposal's reset trace, including the
first tick of an episode. Installing same-shaped new weights retains history
and performs no physical or neural step. No per-tick host wait, readback or
state allocation is introduced.

Whole-grid input/clock validation precedes state writes. Invalid input, clock
gaps or reversal set sticky errors and reject publication without partially
changing other lanes' hidden state. Command ownership excludes competing
commands and host boundary operations until completion. Explicit reset recovers
a poisoned context. Quiescent component checkpoints bind parameters, library,
shader, device and OS; restore validates semantics and uploads candidate state
before swapping ownership. A standalone component checkpoint is separate from the crow life envelope.
The later actor integration embeds it in that envelope; a recurrent crow learner
is not yet implemented.

Build `metalrobo_gru_memory_check` and `metalrobo_gru_shaders`, then run:

```
tools/numi crow-env memory-admission --output NEW_MEMORY_TRIAL
ctest --test-dir build --output-on-failure -R '^metalrobo[.]learning[.]gru_memory$'
```

The bounded native instrument uses N/I/H=3/5/7 and 4/257/65, 40 authored neural
ticks, lane resets at 11/23, publication of new weights at 18, chunks 1/8, duplicate
inference, previews and a fresh-process restore. Maximum FP64 error is
2.067e-7 against the declared 3e-6 threshold. All future hidden/pre-cell/reset
traces and final component checkpoints match exactly. 31 controls reject invalid
parameters, checksums, seven correctly re-signed but semantically invalid
checkpoints, buffers/offsets, ownership, budget, nonfinite input and clock gaps.
The fixture clock is an explicitly synthetic neural instrument, never a Python
or host substitute for accepted crow transitions.

MLX/native sequence and pre-cell error is at most 2.087e-7. Independent FP64
reverse-mode derivatives check every element of all four parameter tables and
inputs: maximum error 8.253e-9, relative norm at most 1.459e-7. Twelve central
differences check the FP64 derivative implementation within 2.508e-12. Compiled
and eager MLX gradients match exactly on this fixture. Gradients cross ordinary
ticks, stop exactly at episode resets, and detach the captured initial state.
A causal input counterfactual verifies reset isolation and retained memory.
Eight full-sequence SGD updates reduce the fixed mathematical objective from
0.07463 to 0.05092; this is an algebra check, not PPO or learned crow behavior.
Four learner shape/dtype controls reject.

The admission retains a matching native library/executable/shader tree and
Python source. A second native process restores from that saved runtime and
reproduces the future traces/checkpoint byte for byte; a separate process using
the retained learner source reproduces its summary and all arrays exactly. Matching installed
Python/MLX/NumPy dependencies are still required.

Evidence, weaker initial checks, failed build/launcher attempts, exact source
deltas and recovery receipts are retained in
`artifacts/crow-rl-gru-memory-20261107`. Those component trials record actor integration as false at that milestone.
The following actor admission supersedes that integration status; sequence PPO
and embodied-memory qualification remain false. Next work must carry captured
hidden/reset traces through the sequence learner and its coupled training bundle,
then test causal memory during physical
foraging and cache tasks with untouched evaluation seeds. Production performance,
solid-object manipulation, physiology and social learning remain open.


## Native recurrent crow actor and life state

The optional recurrent actor now executes inside the real native Metal crow
rollout. PolicyPack format5 stores a nonempty dense encoder, reset-after GRU,
and nonempty action head. The critic remains feedforward. Dense-only writers
retain format4 and unchanged compiled policy ABI3 bytes; recurrent policies
require policy ABI4 and a declared shader capability. Compilation validates the
encoder/head widths, all GRU parameters and the encoder's input bound before
publishing an immutable policy. A resident life rejects architecture changes,
clearing recurrent policy authority, and legacy rollout publication that would
discard memory.

The GRU reads the accepted native task clock and episode. Actual hidden A/B
vectors stay device-private. Native rollout boundary traces expose the pre-cell
hidden state and reset mask for every actor evaluation, including requested
terminal previews. Extra previews do not promote hidden state. Next same-shaped
weights install at an idle boundary and retain history. The recurrent allocation
and traces are included in world memory planning and retained-buffer statistics.
No per-step Python transition, host wait, hidden-state readback or allocation is
added to inference.

The existing full v13 life envelope now embeds the GRU component checkpoint
inside its physical world component, alongside contact/task/RNG state. Retinal
queues and finite food remain separate envelope components. Checkpointing binds
the next published weights without a physical/neural step. Restore validates a
candidate GRU before publishing resident ownership. Exact native library,
shaders, device, OS, configuration and next policy remain required.

```
cmake --build build --target metalrobo_crow_recurrent_actor_check
tools/numi crow-env recurrent-actor-admission --output NEW_RECURRENT_ACTOR_TRIAL
ctest --test-dir build --output-on-failure -R '^metalrobo[.]crow[.]recurrent_actor$'
```

The two-lane supported v13 instrument runs40 real native ticks with an authored
1251→5 encoder, GRU7 and18-action head; only small beak actions are enabled.
It resets lane1 at9 and publishes next GRU weights at18 without advancing the
life. FP64 actor and pre-cell error is1.422e-8 against a prospective5e-6 bound.
Chunks1/8, extra intermediate previews and an independent process reproduce all
real observations, actions, pre-cell/reset traces, transitions and final q/v
exactly. All three final life-state component blobs also match byte for byte.
Final camera capture is aligned in all arms: the first attempt preserved every
future stream but correctly failed a comparison with mismatched endpoint camera
processing. It is retained, with its exact source/runtime and diagnostics.

Seven malformed compiler contracts, three live publication controls and a
retained policyABI3 shader reject without changing the subsequent life. Numi
retains the matching native runtime and independently restores the original life
checkpoint with it; all future stream bytes match. The external retained-child
check does not compare its final life component blobs; that comparison is made
by the native admission's own fresh child. Receipts describe both scopes.

14 selected native checks and26 Python checks pass. The ordinary feedforward
learner's model/Adam hashes and frozen ordinary/residual feeding fingerprints
remain unchanged. Raw trials, failed builds, runtime/source identities,
checkpoint payloads and regressions are in
`artifacts/crow-rl-recurrent-actor-20261107`.

This admits native execution and persistence, not learned embodied memory.
Python can now read and publish PolicyPack5 parameters (below), but training
cannot yet consume recurrent sequence records;
legacy rollout7 publication explicitly rejects recurrent actors. Next work must
version the hidden/reset rollout contract, implement contiguous sequence PPO
with detached captured initial state and episode gradient cuts, and qualify
coupled recurrent training resume. Causal memory tasks, solid-food manipulation,
social learning, physiology, production throughput and ecological evaluation
remain open. No qualification seeds are used or policy promoted here.


## Recurrent policy parameter interface

The additive `mr_write_recurrent_policy_pack` C function accepts the existing
unchanged `MRPolicyPackC` and a separate ABI1 `MRPolicyGRUMemoryC` descriptor.
It validates descriptor version/size, dimensions, tensor counts/pointers and
canonical GRU parameters before atomic PolicyPack5 publication. The legacy dense
writer and its C structure remain unchanged. Python `write_policy_pack` accepts
an optional `PolicyGRUMemoryArtifact`; dense clients bind only the old symbol,
preserving compatibility with earlier libraries.

`read_policy_pack` now validates recurrent descriptor dimensions, the exact
encoder/GRU/head boundary, all other dense adjacency, finite tensors and GRU
FP32 accumulator bounds. The NumPy flat actor evaluator and all three existing
feedforward learner/initialization/projection entry points explicitly reject
recurrent artifacts. They cannot silently omit memory. A future sequence learner
must accept captured initial state and contiguous time/reset masks explicitly.

```
tools/numi crow-env recurrent-policy-admission --output NEW_POLICY_INTERFACE_TRIAL
```

This admission reruns the real native actor/life test and independently retained
runtime continuation. It then reads the first and next native recurrent policies
and the dense control through Python and republishes all three through the
canonical C writers. Every policy byte matches. MLX evaluates both contiguous
weight segments on all80 real native observation slots, using the captured
segment-start hidden state and reset masks; actions and pre-cell states match
within1.491e-8 against the declared5e-6 bound. This is forward sequence algebra,
not PPO, a synthetic crow simulation or learned behavior.

17 controls reject four flat evaluator/learner uses, three invalid writer
inputs, three raw C descriptor ABI/size/dimension errors and seven correctly
re-signed semantic PolicyPack corruptions. A second process using retained
native/Python source reproduces its summary, every sequence array and all three
canonical policy files exactly. Matching installed Python/MLX/NumPy remain
required. Evidence and recovery receipts are in
`artifacts/crow-rl-recurrent-policy-20261107`.

At that policy-interface milestone, recurrent rollout publication and sequence
training remained pending. The following rollout admission supersedes the
publication status; contiguous PPO/BPTT, recurrent optimizer checkpoint/resume
and embodied memory qualification remain required. The legacy rollout7 writer
still rejects recurrent actors.


## Recurrent rollout handoff and ordinary CLI continuation

Format8 adds a mandatory ABI1 memory section to the ordinary step-major rollout
record: hidden width, pre-cell `float32[T,N,H]` state and `uint32[T,N]` episode
reset mask. It stores real actor evaluations only. Terminal actor previews stay
available through native readback, but a separate real-step-count getter bounds
the Swift extractor and excludes them from training. Each segment retains the
compiled generating-policy fingerprint and revision. A resident recurrent life
now requires an increasing revision whenever the policy fingerprint changes;
reused or reversed revisions cannot silently mix different generating weights.

The additive `MRPolicyMemoryBatchC`/recurrent writer leave the existing C rollout
batch and dense writer unchanged. Dense records remain byte-identical format7;
its old writer deliberately rejects recurrent actors. The format8 validator
checks exact tensor counts, bounded finite hidden state, Boolean masks and zero
pre-cell state on resets. Python validates the same contract. Flat `policy_batch`
consumption explicitly rejects recurrent records before it can lose memory.

Swift batch extraction/concatenation and the training append arena both carry
memory. Concatenation requires consistent hidden width and complete per-chunk
traces. The legacy append overload rejects recurrent use; the new overload
appends native streams directly without materializing another complete Swift
chunk. The training owner reserves the memory arena once at each update's first
boundary and retains its capacity between updates. Recurrent records also enter
the ordinary CLI trajectory fingerprint; dense fingerprint bytes stay unchanged.

```
cmake --build build --target metalrobo_crow_recurrent_rollout_check \
  metalrobo_crow_recurrent_actor_check metalrobo_task_train metalrobo_task_rollout
tools/numi crow-env recurrent-rollout-admission --output NEW_ROLLOUT_TRIAL
ctest --test-dir build --output-on-failure -R '^metalrobo[.]crow[.]recurrent_rollout$'
```

The native instrument exports18- and22-tick constant-policy segments around the
next-weight publication. It resets lane1 at9, aligns both endpoint captures, and
adds extra intermediate previews in its comparison arm. Native read/write
roundtrip, Swift chunk1 concatenation and Swift chunk8 training append reproduce
every segment byte and final q/v exactly. Python reproduces all native learner,
pre-cell and reset arrays. MLX sequence latents/pre-cell states match all80 real
slots within1.491e-8 against the prospective5e-6 bound. Ten malformed memory
exports, ten Swift misuse controls and eleven parser/flat-consumption controls
reject. Seven compiler, five resident publication and stale-shader controls also
pass. The rejected revision publications leave subsequent life unchanged.

The ordinary CLI uses a second, previously unused development seed2651109501.
At chunks1/8 it runs37ticks, saves full native life, and starts a fresh process
for13 additional ticks. Canonical rollout bytes and fingerprints match in both
phases. Accepted lifetime clocks are exactly37 then50, with no warmup. A retained
CLI binary/library/shader tree independently restores the original37-tick life
and reproduces every continuation rollout byte. MLX evaluation of the complete
50-tick record matches all100 real slots within1.491e-8, including the checkpoint
boundary. The continuation begins with hidden norm0.2611 and no episode reset.
A modeled zero-initial-hidden continuation changes its action mean by0.002862
against the prospective minimum0.0001. This is a neural algebra counterfactual,
not a native physical memory intervention or learned foraging result.

Retained native/Swift runtime and five Python modules reproduce the fixture and
ordinary-CLI summaries, every sequence array and policy/rollout bytes exactly.
Sources bind to preflight hashes; matching installed Python/MLX/NumPy and the
exact native device/OS/runtime remain required. Both full incremental builds
pass;15 selected native and26 Python checks pass. Feedforward learner model/Adam
hashes and frozen ordinary/residual feeding fingerprints remain unchanged.
Earlier successful variants, source/runtime identities, raw records, controls,
checkpoints, regression output and lossless payload inventories are in
`artifacts/crow-rl-recurrent-rollout-20261107`.

At the rollout milestone, contiguous PPO/BPTT, sequence minibatch scheduling
and coupled recurrent optimizer/life resume remained pending. The sequence PPO
admission below supersedes that implementation status. Embodied memory experiments, physical manipulation, social
learning, ecology and production performance remain unqualified. These replay
instruments use development conditions only and promote no learned policy.

## Evidence storage

Crow evidence belongs in the Git repository. Use bounded temporary captures; preserve failures and runtime identities, archive missing content once, commit and push, verify recovery, then remove the covered raw files. Keep only current build products and active policy/shader inputs locally. Do not retain duplicate run trees, runtime trees or bulky checked-out archive payloads. Shared Git objects, history and other projects remain protected.

`artifacts/crow-rl-git-recovery-20261107/README.md` documents the exact-file manifest, content deduplication, Git-object verification, restoration and cleanup receipts. `tools/crow_evidence_archive.py` provides the corresponding commands. Bulk archives can be absent from this worktree with per-index skip-worktree flags; recovery reads committed Git blobs directly. These storage operations establish preservation, not additional scientific qualification.

## Contiguous recurrent PPO and coupled training life

`tools/numi crow-env recurrent-ppo-admission --output NEW` now constructs its
own native fixture, authors a distinct stochastic seed and collects real native
rollouts. The production MLX worker recognizes recurrent PolicyPack5 and uses
contiguous truncated BPTT. It shuffles lane/time windows as units, detaches the
captured initial state, applies every reset mask and masks padded tails in all
losses and metrics. Only the active sequence minibatch reaches MLX; full visual
observations remain NumPy views. Old-policy means are frozen before the first
Adam step. The supported trainable encoder ends in tanh, preserving the native
GRU input bound. Flat teacher/retention/projection modes reject explicitly.

The native stochastic37+13-tick instrument uses two lanes on development seed
2651109503. Ten length8 windows cover all74 initial transitions once per epoch.
Padding changes no gradients; detached-initial and episode-reset gradient cuts
are exactly zero. Eleven selected parameter-family coordinates match independent
FP64 PPO finite differences within1.016e-5 against the prospective3e-5 bound.
Both updates pass native generating-state/log-probability/value parity; all four
GRU parameter families train. A fresh learner restores all model/Adam tensors
and reproduces its next update and policy bytes exactly. Eight invalid batch
and input controls reject without changing learner state.

The ordinary Swift/Metal/MLX training path then runs three uninterrupted updates
versus two plus a fresh-process continuation: two lanes,37ticks/update, chunk8,
sequence length32, development seed2651109507. At all three boundaries the
world/contact/task/RNG/GRU, delayed retina and finite-food components are byte
exact. Policies, rollouts and canonical learner files are also byte exact; all35
model/Adam tensors match. Accepted clocks are37,74 and111 with no warmup. All six
checkpoint manifests, commit markers and declared payload hashes validate.
Changed GRU input bound, changed sequence length and missing recurrent checkpoint
contract each reject before mutating state.

The initial whole-container byte prediction failed: native envelopes include
measured GPU/submission times, and restored Adam dictionaries changed tensor
serialization order. That attempt is preserved. Recurrent publication now sorts
tensor names. The corrected instrument validates native checksums, exact
clocks/counters and every causal component while retaining and validating the
two nonnegative timing measurements separately. No numerical tolerance changed.

The ordinary feedforward initial/resume learner checkpoint hashes remain exactly
29f79d3c670b7476757945a62104d78ba76012220be388c07b6f24e93f75d8d1 and
6c2d517062ccc6c28ea127ffea9f59fc12c00d33179140c6565503a758c882fe. Thirty-three
focused Python checks pass, including independent timeout/GAE and reset-edge
tests. Evidence, failures and source variants are archived once under
`artifacts/crow-rl-sequence-ppo-20261107`; temporary captures are pruned after
push and recovery verification. Earlier instrument profile flags describe their
historical stage; current integration is recorded in
`recurrent_ppo_development_instrument`.

This admits an engineering learning/persistence path on the small authored
encoder5/GRU7 controller. It does not demonstrate learned use of memory, lifelong
foraging, solid-object manipulation, social learning, biological fidelity or
production throughput. Configurable initialization is now available as described
below; controlled behavioral memory tasks, manipulation/social mechanisms and
complete-loop profiling remain required before the research environment can qualify.


## Configurable recurrent policy initialization

The ordinary `metalrobo_task_train` owner can now author a fresh recurrent
PolicyPack directly from the compiled task contract. With its existing native
world, policy paths and MLX arguments, use:

```sh
--initialize-policy crow-recurrent-development \
--recurrent-hidden-size 64 \
--recurrent-encoder-sizes 96,48 \
--recurrent-sequence-length 16 \
--minibatch-size 128
```

`--recurrent-hidden-size` selects a fresh GRU actor. Positive comma-separated
encoder widths default to128 when omitted. Every encoder layer uses tanh; the
last one enforces the native absolute input bound of1. A seeded fan-in uniform
initializer and zero biases author the encoder, reset-after r/z/n GRU,
identity-output Gaussian actor and separate ELU critic. The worker's existing
`--hidden-sizes` option chooses encoder widths when initializing it directly;
the learner API also supports independent critic widths.

`--zero-actor-output` works for a recurrent actor and preserves its encoder and
GRU. Source-actor import, input projection and nonpositive capacity reject.
Fresh initialization flags are omitted on resume; the checkpoint already binds
the exact architecture. Repeat `--recurrent-sequence-length` on resume: it is
part of the scheduler/learner contract, and changing it rejects before updates.
The default sequence length remains32. Feedforward initialization is unchanged.

The development admission used actor1251→tanh96→tanh48→GRU64→Gaussian18,
critic97→ELU96→ELU48→scalar, eight native crow lanes,37ticks/update, chunk8,
sequence length16, minibatch128, one epoch/update and fixed learning rate0.0003.
Environment seed2651109601 and learner seed2651109603 are engineering units.
Two independent initial publications were byte exact. Three uninterrupted
updates versus2+1 fresh-process resumed updates matched policies, rollouts,
canonical learner bytes and all53 model/Adam tensors. All three native causal
components matched at clocks37,74 and111; native wall timings are validated
separately. All four GRU parameter families changed and native failed steps
were zero. No food was consumed in these short runs, so parameter updates do
not demonstrate useful foraging or learned use of memory.

The full build and29 focused Python checks passed. Six malformed/incompatible
initialization and resume controls rejected, leaving the valid checkpoint
unchanged. The prior feedforward initial and resumed learner hashes remained
exact. The first negative-control harness expected a different error substring
for the correctly rejected sequence change; its logs and revised assertion are
both retained. An initial unused import error occurred before any native trial
and is also preserved.

Inputs, programs, source snapshots, failures and bounded raw evidence are
archived under `artifacts/crow-rl-recurrent-capacity-20261107`. Recovery uses the
committed Git blobs; duplicate captures and archive checkout payloads are
removed after push and verification. This is a configurable training-path
admission. Architecture optimality, memory-dependent behavior, manipulation,
social learning and production throughput remain unqualified.


## Persistent randomized food sites and delayed access

`FiniteResourceConfig` now accepts an optional `MRFiniteResourceAllocationGPU`
descriptor and runtime `allocationSeed`. The default all-zero descriptor keeps
the original all-sites, immediately available, observable-proximity behavior,
legacy program hashes,80-byte state layout and exported three-argument C++
fingerprint symbol. Version1 selects exactly `activePatchCount` sites from up
to eight sites on the first accepted native tick. A counter-based integer
stream with rejection sampling chooses without replacement on Metal; neither
submission chunking nor episode resets enter that stream.

`availableAfterTicks` forbids intake on the first specified number of accepted
lifetime transitions. A close during that interval consumes the physical jaw
edge; time passing while the jaw remains closed cannot manufacture a later
bite. `concealProximity=1` keeps both nutrition proximity/stock-fraction channels
exactly zero, leaving legitimate crop/intake feedback. Runtime seeds are
excluded from the policy task-program fingerprint and included in the exact
resource identity. Checkpoints validate each lane's expected selection mask,
inactive-site zero stock and exact selected-stock conservation. Changed seeds,
forged masks or stock shifted into an inactive site reject before mutation.
Ordinary episode resets preserve selection, finite stock and the lifetime clock.

The native `foraging-admission` owner now includes this mechanism. Its final
instrument passed all nine448-tick physical controls on eight supported crow
lanes plus separately labelled synthetic accepted-state calibration. One of two
sites is stocked at seed2651109701; the central site is a positive intake control
and the site0.25m away is a miss control. With a192-tick delay, the first two
closing gestures consume nothing; the next two debit40,000ug in the five lanes
with central food and zero in the other three. Food totals and clock survive
reset; chunk8/64 resource states are byte exact. No native failures or physical
terminations occurred in admitted trials.

At the paired seed2651109702, five hidden layouts change. All192 pre-intake
ticks of the97-feature actor stream remain byte exact across seeds:595,968bytes
per stream, covering eight lanes. The selected-site masks for every active count
0..8 over eight sites and eight lanes (72cases) match an independently authored
Python integer reference, as do the eight primary two-site masks. Each case
checks initial stock and resource-checkpoint restoration. Tests also cover zero
stock, delay expiration without a new edge, rejected physics, five malformed
programs, three forged/rebound checkpoints and64-bit lifetime clock carry.

Failures remain evidence. The first crop-fraction assertion was too strict for
FP32 division (0.1999999881 versus0.2); the existing1e-6 fraction bound now
applies, while concealed channels must still equal zero exactly. Review found
and removed a shadowed availability variable that suppressed legacy presence.
A separate renamed TaskPack lost the compiler's name-derived avian-ground flag
and failed with100 required contact constraints against capacity96. The revised
instrument inherits the canonical feeding task support semantics, asserts its
flags, and binds the resource variant through its state-program fingerprint.
No contact capacity or physics tolerance was increased. Task-name-dependent
physical flags remain a compiler limitation for future task authoring.

The full build and29 focused Python tests pass. Existing native recurrent PPO,
optimizer restore/next-update parity and feedforward initial/resume checkpoint
hash regressions also pass. Source/runtime variants, failures, inputs and raw
streams are archived under `artifacts/crow-rl-spatial-resources-20261107`, then
verified duplicate captures are pruned.

This admits a native resource mechanism, including resource-only restore. The
second site has not been admitted as a reachable alternative, and no visible
cue, learned search/recall, solid food handling or coupled spatial-life restore
is claimed. The next environment change must bind reachable sites and an
occluded visual cue through the existing native renderer before training or
memory ablations can qualify. American-crow walnut caching and social threat
learning motivate the broader research scope; the current tracer parameters
are engineering estimates rather than measurements from those studies:
[walnut caching](https://www.sciencedirect.com/science/article/pii/S0003347201917488),
[social learning](https://faculty.washington.edu/wirsinga/Cornell2011.pdf).
