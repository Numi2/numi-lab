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
`metalrobo_crow_retinal_contract_check` with
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
persistent memory, complete world/RNG/sensory/memory restore, social agents
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
and final q/v streams are bit-identical for chunks1/8. Explicit world reset
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
`artifacts/crow-rl-retinal-learner-20261107`. Whole-world/RNG/sensory/memory
restore, persistent trainable memory, physical object manipulation, physiology,
social agents and ecological held-out qualification remain open. The full
lifelong environment is not qualified.
