# Integrated tissue contact native smoke test, 2026-10-20

Executed only on SSH Numi, Apple M6 (12 GPU cores), macOS 27.0, Metal 4.
Apple standalone Metal compiler 32023.883 is installed in the user toolchain.
Activate it with: . /Users/numi/.config/numi/metal-toolchain.env
The sparse checkout must include /matter/materials/.

Build: cmake --build build-integrated-validation-20261019/profile-build-039 --target numi-human-native -j 4
Existing rope source assembly and CMake's NUMI_HUMAN_RESTING_ROPE_SOURCE option
were used; its exact source-input hashes are retained. No old Matter or
articulated runtime source overrides were used. Each invocation JSON retains
the exact command, environment and asset paths. The run-result pins the binary.

Default-budget run rejected step four after three accepted steps; retain its
circulatory line-search residual and first-failure receipt. Increasing the
existing shared Newton budget to 24, without changing tolerances, passed four
125-microsecond steps (0.0005 simulated seconds) in 84.298254 seconds, including
initialization. This is an integration smoke test, not a performance benchmark.

The shared runtime admitted 22 FEM objects, 14,302 nodes and 14.416008 kg of
debited FEM mass. It retained 495 body-contact samples across all 26 fascia
regions and two inferred ACL FEM objects, with no duplicate ligament force rows.
At the terminal capture: 190 closed samples, normal force 0.026400581 N,
force residual 0 N, moment residual 7.73853e-11 Nm, min J 0.707865448,
max J 1.121440436. Adapter passes and accepted steps both equal four.

Limitations: these 26 regions are fascia, not 26 ligaments. Only bilateral
inferred ACL mechanics are integrated; all anatomical ligaments are not
implemented. The whole-body skin/fat reduction remains pinned to its retained
construction frame; an alternate supine root is rejected by admission.
This smoke test uses existing rope support, not supported-rest qualification.
No visual verification was performed. Bleeding and blood MPM were disabled.
No sustained stability, physiological, anatomical-completeness or clinical
qualification is claimed. Failure logs for earlier launch/configuration checks
remain in native-integration-profile-088 through -094 on Numi.

Current-FEM rollback test: two accepted 125-microsecond steps plus an injected
post-dynamics rejection at step 1. Passed in 65.782378 seconds including load.
Accepted q/v/root/muscle, Matter/FEM/circulation, adapter history, Brain, respiration
and presentation remained unchanged on rejection. Body/DoF properties were restored
and the same immutable input succeeded on retry. Terminal adapter passes=3
(including rejected attempt), accepted contact audits=2. The previous fixture
required retired passive force rows; its precondition now also admits established
FEM/contact and anchor-reaction history. All equality/hash/rollback checks remain.
