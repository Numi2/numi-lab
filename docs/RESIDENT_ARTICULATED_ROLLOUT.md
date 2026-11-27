# Resident articulated rollout on Apple silicon

Numi Lab for Mac selects the compact Metal solver for compatible single-world
scenes. Run summaries identify the selected backend as
`metal-compact-articulation` or `metal-general`. Compatible task graphs run on the
GPU across a submitted chunk. Physics records accepted states, then parallel
passes evaluate the authored observations and rewards and assemble temporal
histories. A final pass commits continuation state and evidence. Run summaries
show the selected graph in `task_graph` as `resident` or `per-control`.
Tasks that require policy feedback, sensors outside the supported operators
or state-dependent termination use the per-control graph. Scenes with
unsupported mechanics, unavailable Metal resources or custom velocity iteration
settings use the general solver. Set `NUMI_MAC_GENERAL_ARTICULATION=1` to select
it explicitly.

The compact solver handles one fixed-base articulation with at most nine scalar
position coordinates, up to two free bodies and bounded analytic collision.
It supports position and gripper actions. CCD, authored constraint rows, task
randomization and disturbance programs require the general solver. A control
that exceeds the compact contact capacity or steps outside a joint's position
limits returns a typed failure at the last accepted state.

`metalrobo_resident_rollout` runs a recorded position-control clip in one Metal
dispatch. Each 5 ms physics substep assembles one effective articulated inertia,
factors it with LDLᵀ and applies the same inverse to free motion and the contact
normal and tangent rows. Independent contact islands solve concurrently. The
position-action response, joint state, free-body state and contact warm starts
stay on the GPU. A control sample is published after all four substeps succeed.
Each recorded contact includes the normal impulse and the world impulse on the
second body at the final substep. Divide by 0.005 seconds for the sampled force.

## Build and run

Build the macOS Release target on Apple silicon. Supply the same cooked packs
and little-endian float32 action clip used by the task rollout.

~~~sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target metalrobo_resident_rollout
build/bin/metalrobo_resident_rollout \
  build/shaders/resident_articulated_world.metallib \
  "$scene/tabletop.worldpack" \
  "$scene/tabletop.taskpack" \
  "$scene/tabletop.actuators" \
  "$scene/tabletop.sensors" \
  "$scene/tabletop.reality" \
  "$scene/tabletop.actions.f32" \
  /tmp/tabletop-resident
~~~

The optional trailing arguments set the contact sweeps, number of warmed trials
and operational contact capacity. The defaults are 1, 7 and 31. The output stem
gets accepted-state, applied-target and contact CSV traces and a binary replay
trace. Timings include command submission, completion and GPU trace writes.
Pack and pipeline creation, CSV export and file comparisons happen outside the
timer.

The compact library uses Metal's relaxed floating-point mode with NaN and
infinity handling. The clip runner times the physics and trace recording in
one dispatch. The packaged app includes the task graph and can also present
live frames. To time the full headless app with
the same clip on every repeat, run the packaged executable.

~~~sh
"$app/Contents/MacOS/metalrobo_task_rollout" \
  --scene-file "$scene/panda-tabletop.numi-window.json" \
  --headless --steps 370 --physics-substeps 4 --chunk 370 \
  --action-clip "$scene/tabletop.actions.f32" --repeats 3 --replay-each-repeat
~~~

`repeat_elapsed_seconds` records each repeat in order. A seeded reset
separates repeats. For per-control state and force export, use `--chunk 1`
and the trace options in the [Panda example](../examples/panda_tabletop/README.md).

## Panda tabletop qualification on the Apple M6

The packaged Release app on `numi` ran 370 controls at 20 ms and 1,480
physics substeps at 5 ms per case. Each engine received identical applied
position targets at every control. Each warmed median uses seven interleaved
trials with two warm-up repeats per trial. Numi's timer includes action
handling, physics, actor and critic observations, rewards, transitions,
evidence, and Swift collection. MuJoCo 4.0.1 times native integration and
state recording. Both timers exclude process launch, rendering and file
export.

| Case | Numi median ms | Numi range ms | MuJoCo median ms | First Numi repeat median ms |
| --- | ---: | ---: | ---: | ---: |
| 100 g | 19.86 | 19.80–20.26 | 20.82 | 28.75 |
| 200 g | 19.78 | 19.75–19.83 | 21.10 | 28.79 |
| 400 g | 19.79 | 19.74–19.85 | 21.29 | 29.04 |
| Open gripper | 9.95 | 9.91–9.97 | 9.97 | 18.88 |

All 28 warmed app trials were below 21 ms. The slowest was 20.26 ms.
A fresh process's first complete rollout had a median near 29 ms for the
loaded scenes. The former packaged per-control graph took 86–89 ms on the
loaded clips and about 77 ms for open gripper in single-pass checks.
The earlier standalone physics-only qualification measured about 19.1 ms
loaded and 9.47 ms with open gripper.

### Tracking and grip force

Approach arm tracking RMS over the first 150 controls was 0.925° in Numi
and 1.838° in MuJoCo in all four cases. During the final 51 controls, Numi
measured 0.160–0.174° and MuJoCo measured 0.186–0.199°. The largest hold
cube-position RMS difference between engines was 3.56 mm. Minimum loaded hold
height differed by at most 3.94 mm.

| Case | Numi left / right jaw N | MuJoCo left / right jaw N | Numi upward support N |
| --- | ---: | ---: | ---: |
| 100 g | 22.04 / 22.04 | 22.15 / 22.15 | 0.981 |
| 200 g | 22.07 / 22.06 | 22.32 / 22.32 | 1.962 |
| 400 g | 22.08 / 22.06 | 22.43 / 22.43 | 3.920 |
| Open gripper | 0 / 0 | 0 / 0 | 0 |

Both engines held all three cubes with both jaws for all 51 hold controls.
The largest mean per-jaw normal-force difference was 1.64%. Vertical
support from Numi's jaws agreed with cube weight within 0.10% for all
three masses. The engines use different finger geometry and contact laws.

### Replay and fallback

The packaged app reproduced all 370 accepted state and solved contact-force
samples in each scene. The native task API also matched bit for bit across
submission sizes of 1, 2, 19 and 370 for each control's actor and critic
observations, reward breakdowns, outcomes and status. Accepted state and
full force witnesses agreed at shared submission boundaries. An explicit
reset at control 211 and a separate 760-control timeout test agreed across
those submission sizes. MuJoCo reproduced its own FP64 state and force traces.
Across the four clips and the two reset tests, integer status and transition
metadata matched the previous per-control graph. The largest absolute
observation difference against that graph was 0.00182, and the largest reward
or outcome difference was 2.1e-7.

The eight-contact capacity test reported overflow at control 168, preserved
the first 167 accepted states byte for byte, and matched a fresh authored
reset on the next accepted control. The installed-style verification app
passed the startup and workbench checks with precompiled pipelines, exercised
both explicit and solver-setting fallbacks, and presented all 370 controls
in a live 960×540 Mac window with zero failed controls.

The original operator qualification in revision a6ffc3f compared all 370
Panda poses against the FP64 effective inertia. It measured a maximum
inverse-action residual of 9.46e-7 and a maximum COM position difference
of 0.61 µm.
