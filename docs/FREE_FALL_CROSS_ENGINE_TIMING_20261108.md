# Matched free-fall timing: a narrow, noisy-host measurement

The predeclared [plan](FREE_FALL_CROSS_ENGINE_TIMING_PLAN_20261108.md) compares collider-free free fall, not contact or a robotics workload. Both implementations passed all 20 timed trajectory checks and their warmups at the 0.001 m and 0.001 m/s limits. The Numi FP32 maximum errors were 3.3811e-6 m and 6.3363e-5 m/s; MuJoCo FP64 errors were 3.02e-14 m and 9.06e-14 m/s.

| Environments | Engine | Median seconds (range) | Accepted simulated seconds / wall second |
|---:|:---|---:|---:|
| 1 | Numi GPU | 0.080436 (0.078393–0.088045) | 12.43 |
| 1 | MuJoCo serial CPU | 0.000301 (0.000226–0.000353) | 3322.71 |
| 32 | Numi GPU | 0.090121 (0.089854–0.094202) | 355.08 |
| 32 | MuJoCo serial CPU | 0.007189 (0.007088–0.008720) | 4451.40 |

Each environment simulated one physical second; throughput is the number of environments divided by median elapsed time. The individual paired values are in [timing-v2.csv](../experiments/free_fall_cross_engine/evidence/timing-v2.csv). On this tiny no-contact model and this setup the serial CPU is faster. This is not a general engine ranking: model complexity, batch size, hardware, precision, contact and CPU parallelism can change the result.

The run was **not on a quiet host**. A detached Git maintenance repack used roughly one CPU core throughout; mediaanalysisd appeared at 119% CPU in the post-run process snapshot, and brief Git fetch/index processes were also present. Thermal state was not controlled. Thus the paired observations are useful evidence for this execution, not a clean attribution or final performance ranking. The GPU was reserved for the benchmark; other Numi test owners had paused significant work.

The instrument is timing.cpp at source commit 86bf3462a; it uses the predeclared alternating order, five pairs per batch, GPU batched versus one serial MuJoCo host thread, and includes per-run reset, steps and final-state retrieval. Numi fixed zero-effort input, model compilation and context creation are outside the timer; mjData allocations are also outside it. MuJoCo's error calculation is inside its timed loop, while Numi checks returned state immediately after its measured call. MuJoCo uses double precision and Numi uses float. Neither timing includes a renderer, sensor, controller or contact.

Identity (SHA-256):

- Executable: 842cb3ae085bdf9c4f9534c8c9604d8ff6200a1fc776bb41ad46ddc5faab4086
- MuJoCo 4.0.0 dynamic library: 33d144886e581ed381785a0e88db8bf4c293ecad3330a91126bd87609617308c
- Numi metallib: 8f9137278d52ff4a083dbc0c84ad541c25db91e8c50d5e7b3f70962cb23b69fa
- Timing source: 2bb7905d830c466394cf99e2681ea1861869ec048b7c8066f886c58283b0b97d
- XML: cd35a3ddcfb0c309a5e83fb139fd3a5437871510322a134d452a2038fa18e98f

The first launcher exited 134 before a trial because dyld could not find MuJoCo's framework-shaped install name; [stderr](../experiments/free_fall_cross_engine/evidence/timing-first-failure.stderr) is retained. An attempted in-place rpath edit also failed due to insufficient header padding ([record](../experiments/free_fall_cross_engine/evidence/timing-rpath-first-failure.txt)). The successful binary was relinked with a task-local framework-shaped symlink to the same hashed MuJoCo library. No performance result is attributed to the failed launch.
