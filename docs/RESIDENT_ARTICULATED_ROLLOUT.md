# Resident articulated rollout on Apple silicon

metalrobo_resident_rollout runs a recorded position-control clip in one Metal
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

The executable supports one fixed-base articulation with at most nine scalar
position coordinates, two free scene bodies and bounded analytic collision.
It handles joint position and gripper position actions with the compiled task
response. Active position limits, unsupported shapes, CCD, authored constraint
rows and general task scheduling require the general solver. A failure returns
a typed status and exposes only the accepted prefix. The opt-in executable
uses Metal's relaxed floating-point mode with NaN and infinity handling.
Interactive rendering, task observations, policy feedback and rewards are
outside this clip runner's scope.

## Panda tabletop qualification on ssh numi

Apple M6, native arm64 Release build. Each row measures 370 controls and
1,480 physics substeps. Numbers are medians of seven interleaved warmed
trials. The final MuJoCo column uses the gain-matched MuJoCo 4.0.1 model
and receives the same applied position targets. The stock column preserves
the earlier benchmark's authored-target comparison.

| Case | Numi wall ms | Numi range ms | MuJoCo stock ms | MuJoCo matched controls ms |
| --- | ---: | ---: | ---: | ---: |
| 100 g | 20.693 | 20.664–21.790 | 24.985 | 20.774 |
| 200 g | 20.752 | 20.722–20.789 | 15.356 | 21.018 |
| 400 g | 20.630 | 20.577–21.630 | 14.907 | 21.213 |
| Open gripper | 17.051 | 17.016–18.175 | 9.928 | 9.935 |

The general Numi task rollout measured 557.547 ms for the nominal clip in
this Release build. The resident rollout was 26.9 times faster. Stock MuJoCo
dropped the heavier cubes. The gain-matched runs held all
three masses. On the nominal applied targets, Numi's first-150-tick and
final-51-tick arm tracking RMS was 0.925° and 0.164°. MuJoCo measured
1.838° and 0.189°. Numi held the three cubes at least 129.37 mm above
their starting height. The corresponding MuJoCo minima were 128.03,
127.04 and 125.45 mm.

The Numi nominal jaw normals averaged 19.74 N and 19.73 N. MuJoCo measured
22.15 N per jaw. Across the three masses, the Numi upward forces from both
jaws were 0.981, 1.963 and 3.921 N, within 0.1% of weight. Both engines
had zero finger force in the open-gripper control. Peak sampled geometric overlap
on a nominal Numi cube contact was 2.89 mm. Different finger geometry and
contact laws limit physical comparison.

Every Numi state and solved-impulse trace repeated bit-for-bit across the
paired runs. Every MuJoCo native replay reproduced its FP64 state oracle
exactly and both MuJoCo force replays matched. Injecting a contact capacity
of eight returned overflow at control 168 with a byte-identical 167-control
accepted prefix. A malformed clip with a NaN was rejected. On the 370
recorded Panda poses, the inverse action's maximum absolute residual versus
the FP64 effective inertia was 9.46e-7. The maximum COM position difference
was 0.61 µm.
