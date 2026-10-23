# Short native shared-FEM joint response pairs

These three matched fresh-process pairs exercised selected MyoSim muscle activation through the existing full shared-FEM owner. Each run used 8 accepted steps at 125 µs (1 ms simulated time), the same binary/runtime/metallib and input hashes within its pair, and no ongoing pose locks or scripted joint trajectories. The outcomes are finite-horizon engineering measurements, not ROM, clinical, grasp, or production qualification.

| Cohort | Shared FEM | Selected-route increment delivery | Joint response |
|---|---|---|---|
| Bilateral supraspinatus | 8/8 accepted in both runs | 2/2 routes delivered +0.1 | q36/q74 changed about +2.91e−7/+3.05e−7 rad over 1 ms. This is a small observed response, not range qualification. |
| Bilateral knee flexors, projected c109 envelope | 8/8 accepted in both runs | 7/8 delivered; right `bfsh_r` ID 343 was saturated | q106/q120 changed +2.451e−5/+2.556e−5 rad; v105/v119 changed +0.04243/+0.04432 rad/s. Shared-FEM endpoint passed, but the selected-route delivery gate failed for ID 343. |
| Nonsaturated MTP flexors | 8/8 accepted in both runs | 3/3 routes delivered +0.1 | Aggregate q111/q125 changed +3.255e−5/+2.315e−5 rad; v110/v124 changed +0.05743/+0.04084 rad/s. These are aggregate toe-body coordinates; there are no independent toe/IP DOFs. Left FDL ID 386 was omitted because its baseline activation was already 1.0. |

The knee candidate changes only ten equality-dependent hard-limit ranges and leaves the equality polynomial and primary knee ranges unchanged. The ten changed coordinates comprise six revolute angular coordinates in radians and four prismatic translations in metres. In particular, q114/q128 are `knee_angle_beta_rotation1_r/l`, revolute coordinates in radians. This repairs the source-equality/domain conflict for the short flexor test; it does not establish physiological knee range.

Route ID 343 maps to `bfsh_r` and to the compiled NHMYO record at index 343, with route bodies `femur_r` and `tibia_r`. The source-index/name/route binding is correct. Static standing activation was already 1.0, so the runtime's `min(1, baseline + 0.1)` clamp delivered zero increment to that selected route. The new assessor reports this separately from accepted FEM endpoints and returns failure for the delivery gate.

Use `run_shared_fem_joint_cohorts.py self-test` for CPU-only checks covering successful increments, saturation, wrong requested increments, source-ID mismatch, and initial-pose mismatch. Raw stdout/stderr remain beside each run; deterministic gzip copies are included for compact review. Exact binary, runtime, metallib, invocation, and payload hashes are recorded in each assessment.
