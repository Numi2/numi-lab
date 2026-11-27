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
normal and tangent rows. Collision starts as soon as the body poses are
available and overlaps the articulation dynamics. The three angular velocity
lanes of each free body share one gyroscopic impulse. A warm start searches
the ordered witnesses for the active collider pair. Independent contact
islands solve concurrently. The position-action response, joint state, free-body state and contact warm starts
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
position targets. Each warmed median uses seven interleaved trials with
two warm-up repeats per implementation. The earlier packaged Numi
revision is `72210f9`. Numi's timer covers the full headless task graph
through Swift collection, including physics, actor and critic observations,
rewards, transitions and evidence. The native MuJoCo 4.0.1 timer measures
integration. MuJoCo records and checks its reference traces in separate
passes. Both timings exclude process launch, rendering and file export.

| Case | Earlier Numi ms | Numi median ms | Numi range ms | MuJoCo median ms |
| --- | ---: | ---: | ---: | ---: |
| 100 g | 20.17 | 18.12 | 17.99–18.19 | 20.89 |
| 200 g | 20.16 | 18.01 | 17.94–18.13 | 20.96 |
| 400 g | 20.07 | 17.98 | 17.95–19.16 | 21.15 |
| Open gripper | 10.10 | 9.71 | 9.68–11.07 | 12.26 |
| 100 g, 4 mm jaw target | 20.17 | 18.11 | 17.98–18.13 | 20.73 |
| 100 g, 9 mm jaw target | 20.18 | 18.08 | 17.97–18.51 | 20.84 |

The warmed loaded medians fell by about 2 ms versus `72210f9`. All 42
warmed app trials finished below 21 ms. The first complete loaded rollout
in a fresh process took 27.04–27.13 ms by case median and the first
open-gripper rollout took 19.71 ms median. The separate nine-trial open
check measured warmed medians of 9.72 ms for the new app, 10.12 ms for
the earlier app and 13.10 ms for MuJoCo. An earlier MuJoCo measurement
for this clip was 9.97 ms.

The former per-control graph
took 86–89 ms on the loaded clips and about 77 ms on open gripper in
single-pass checks. The earlier standalone physics-only qualification
measured about 19.1 ms loaded and 9.47 ms with open gripper.

### Tracking and grip force

Approach arm tracking RMS over the first 150 controls was 0.925° in Numi
and 1.838° in MuJoCo in all six cases. During the final 51 controls, Numi
measured 0.160–0.174° and MuJoCo measured 0.186–0.199°. The largest hold
cube-position RMS difference between engines was 3.56 mm. Minimum loaded hold
height differed by at most 3.94 mm.

| Case | Numi left / right jaw N | MuJoCo left / right jaw N | Numi upward support N |
| --- | ---: | ---: | ---: |
| 100 g | 22.04 / 22.04 | 22.15 / 22.15 | 0.981 |
| 200 g | 22.07 / 22.06 | 22.32 / 22.32 | 1.962 |
| 400 g | 22.08 / 22.06 | 22.43 / 22.43 | 3.920 |
| Open gripper | 0 / 0 | 0 / 0 | 0 |
| 100 g, 4 mm jaw target | 24.36 / 24.35 | 24.52 / 24.52 | 0.981 |
| 100 g, 9 mm jaw target | 18.57 / 18.56 | 18.61 / 18.61 | 0.981 |

Both engines held the cube with both jaws for all 51 hold controls in
each loaded case. The two additional clips changed final jaw targets
while keeping the arm trajectory fixed. Their mean jaw force differed
from MuJoCo by at most 0.69%. The largest mean per-jaw normal-force
difference across all loads was 1.64%. Vertical
support from Numi's jaws agreed with cube weight within 0.10% for all
three masses. The engines use different finger geometry and contact laws.

### Placement and release

A 678-control clip continued the 100 g lift by lowering along the same
path, setting the cube down, opening the gripper and retreating. Five
interleaved warmed trials measured 33.74 ms in the full Numi app,
37.06 ms on revision `72210f9` and 36.35 ms in MuJoCo. Numi and
MuJoCo finished 0.30 mm and 3.39 mm, respectively, from the initial
horizontal cube position. Both measured zero jaw force and 0.981 N
of upward support from the table over the final 25 controls. Full-cycle
arm tracking RMS was 0.659° for Numi and 1.253° for MuJoCo. Use the
[return-to-table example](../examples/panda_tabletop/README.md) to
generate and replay the clip.

### Replay and fallback

The packaged app reproduced every accepted state and solved contact-force
sample in all six 370-control clips and in the 678-control placement clip.
The four original clips reproduced the earlier packaged app's state and
force and task output traces byte for byte. The native task API also matched
bit for bit across submission sizes of 1, 2, 19 and 370 for each control's actor and critic
observations, reward breakdowns, outcomes and status. Accepted state and
full force witnesses agreed at shared submission boundaries. An explicit
reset at control 211 and a separate 760-control timeout test agreed across
those submission sizes. The placement clip also matched across submission
sizes of 1, 2, 19 and 678. MuJoCo reproduced its own FP64 state and force
traces. In the original four clips and the two reset tests, integer status
and transition metadata matched the previous per-control graph. The largest
absolute observation difference against that graph was 0.00182, and the largest reward
or outcome difference was 2.1e-7.

The eight-contact capacity test reported overflow at control 168, preserved
the first 167 accepted states byte for byte, and matched a fresh authored
reset on the next accepted control. The installed-style verification app
passed the startup and workbench checks with precompiled pipelines, exercised
both explicit and solver-setting fallbacks, and presented the 678-control
placement clip in a live 960×540 Mac window with zero failed controls.

The original operator qualification in revision a6ffc3f compared all 370
Panda poses against the FP64 effective inertia. It measured a maximum
inverse-action residual of 9.46e-7 and a maximum COM position difference
of 0.61 µm.
