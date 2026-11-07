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
`metalrobo_crow_embodiment_check` with CMake/Ninja Release.
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
