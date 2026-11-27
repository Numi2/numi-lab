# Resident articulated rollout on Apple silicon

Numi Lab for Mac selects the compact Metal solver for compatible single-world
scenes. Run summaries identify the selected backend as
`metal-compact-articulation` or `metal-general`. The standard task graph
still owns control targets, resets, observations, rewards and viewer frames.
Scenes with unsupported mechanics, unavailable Metal resources or custom
velocity iteration settings use the general backend. To select the general
backend explicitly, launch the app with `NUMI_MAC_GENERAL_ARTICULATION=1`.

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
one dispatch. The Mac app also runs the standard task passes and can present
live frames.

## Panda tabletop qualification on the Apple M6

The native Release benchmark on `numi` used 370 controls at 20 ms and
1,480 physics substeps at 5 ms per scene. Wall times are medians of seven
interleaved warmed trials. The gain-matched MuJoCo 4.0.1 comparison receives
the same applied position targets as Numi. The stock column preserves the
original authored-target comparison. Stock MuJoCo dropped the heavier cubes.

| Case | Numi clip ms | Numi range ms | MuJoCo matched ms | MuJoCo stock ms |
| --- | ---: | ---: | ---: | ---: |
| 100 g | 19.178 | 19.146–20.081 | 20.849 | 24.996 |
| 200 g | 19.113 | 19.030–20.438 | 21.090 | 15.338 |
| 400 g | 19.071 | 19.004–19.137 | 21.184 | 14.836 |
| Open gripper | 9.471 | 9.431–9.577 | 9.952 | 9.954 |

Open-gripper physics took 17.051 ms before the free-motion solve and rest
contact changes. In the packaged app, the default backend ran the headless
task graph in 87.4 ms for the 100 g scene and 77.8 ms for the open-gripper
scene. These timings include the normal task passes with rendering disabled.
Selecting the general backend explicitly took 573.8 ms for the 100 g scene
in a verification run.

### Tracking and grip force

Numi's first-150-control and final-51-control arm tracking RMS against the
applied targets was 0.925° and 0.164° for 100 g. MuJoCo measured 1.838° and
0.189°. Both engines held all three masses with both jaws for all 51 final
controls. Minimum lift was 129.75, 129.60 and 129.39 mm in Numi and 128.03,
127.04 and 125.45 mm in MuJoCo.

| Case | Numi left / right jaw N | MuJoCo left / right jaw N | Numi upward support N |
| --- | ---: | ---: | ---: |
| 100 g | 21.94 / 21.93 | 22.15 / 22.15 | 0.981 |
| 200 g | 21.97 / 21.95 | 22.32 / 22.32 | 1.962 |
| 400 g | 21.97 / 21.96 | 22.43 / 22.43 | 3.920 |
| Open gripper | 0 / 0 | 0 / 0 | 0 |

The previous nominal normal loads were 19.74 and 19.73 N. With penetration
recovery over two substeps at articulated contacts, the maximum nominal
per-jaw difference fell to 1.01%. Nominal peak geometric overlap fell from
2.89 mm to 0.93 mm. The two additional 100 g trials with 4 mm and 9 mm final
jaw targets were within 1.12% per jaw. Vertical support from the two jaws in
the three loaded scenes stayed within 0.1% of cube weight. The engines use
different finger geometry and contact laws.

### Replay and fallback

Numi reproduced its accepted state and solved-impulse traces exactly, and
MuJoCo reproduced its own FP64 state and contact force traces exactly.
The packaged app reproduced all 370 state and contact-force samples for the
four scenes. Its native task API agreed at shared boundaries with submission
sizes of 1, 2, 19 and 370, including a seeded restart and a reset at control
211 in the nominal scene. An eight-contact capacity stopped before control
168 and preserved the first 167 controls byte for byte. A clip with a NaN
was rejected.

The signed local app rendered the Panda replay on the M6. The app's startup
and workbench checks also exercised the general solver with bundled and
imported scenes.

The original operator qualification in revision a6ffc3f compared all 370
Panda poses against the FP64 effective inertia. It measured a maximum
inverse-action residual of 9.46e-7 and a maximum COM position difference
of 0.61 µm.
