# Unsegmented depth contract: bounded check

The bundled G1 ball-only depth contract selects exact simulator instance IDs.
It is an oracle and remains the default. The opt-in SensorPack
useUnsegmentedDepth contract instead passes all valid metric depth samples
through the existing authored normalization, noise, and history machinery.
Its observation fingerprint differs while its tensor dimensions are preserved.
The derived features describe the entire image, not a ball. Existing
oracle-trained policy weights must not be interpreted as qualified for it.
The C rollout visual configuration does not expose this authoring switch.

On ssh numi, 7 November 2026, a small native Metal test supplied synthetic
depth, identity, and validity planes directly to the production observation
kernel. It passed identity invariance for raw depth, identity sensitivity for
the oracle, valid versus invalid samples, reset filling of all ring slots,
full-frame dropout, and agreement with the synthetic direct geometric-winner
path. Run tests/sensing/run_depth_kernel_check.sh BUILD_DIRECTORY on an
available Metal device to repeat the isolated kernel check.

Tested SHA-256 identities:

- src/metal/HybridRenderer.metal:
  56b76c9619c2f2e919c989bd0cd79b5390f65edb1498ca94cf50688fffa471e7
- tests/sensing/depth_kernel_check.mm:
  ecb1beb4a8eb230c29680b1f42e23e849958fff7a32e352037f63d0d9d57eba2
- Compiled metallib:
  740ed13140eb88b94b16a447e6827cfbd18d3a8bd5f2c77e5570a11001662a54
- Test executable:
  865aade45d6c3f610b94a1c238ce2eddd761962c64783a9f5a4d4bf9fa178c

A focused CPU test compiled the G1 contract in both modes, checked the
unchanged 1,210-element actor tensor and different fingerprints, then wrote
and reread the SensorPack with both unsegmented depth and derived features
enabled. Its source tests/sensing/depth_contract_check.cpp has SHA-256
3fc61e8f985200175c4395991ea59f3655836e264d45c07491bf8182ef7bc591;
the executable hash was
fbfe99f12ccaf388e89a40327014cb06f4c453cffbe00b07dcb13cc8b02eea21.
It linked the changed TaskProgram and LearningPacks sources against a retained
previous owner archive with hash
ae265d71c8b20b04dc4bd02078bbe6a657a911d2cfd9aa1d4635614a09c5fb1c.
This is a focused contract check, not a fresh full-owner build. Full CMake
configuration of the sparse checkout failed due to missing numi/VERSION.in
and matter/tools/human_skin_fat_static_query.cmake.

These results do not establish a trained deployable policy, detection
accuracy, real-camera calibration, or full visual rollout correctness. They
also do not establish observation restoration after a rejected physical step.
In particular the tracker advances its host-side observation sequence during
encoding; its relationship to rejected steps and sensor cadence needs a
full transaction test before this path can be described as replay safe.

A separate CPU cadence check covers a rejected observation encoding: the
candidate sequence is now committed only when the observation encoder
successfully returns, and reset restarts the cadence. This does not cover
later command-buffer failure or a rejected physics step. Physical status is
per environment and appears only after the observation kernels have already
modified depth and object-track histories. The current task-completion path
also treats a physics failure as a terminal episode. A full repair needs
per-environment staged image, object-track and depth histories, held-frame
cadence, and a defined task/reset transaction; it cannot be inferred from the
host-only cadence check.

The CPU contract probe was subsequently extended. It passed all four
combinations of the derived-feature and unsegmented-depth SensorPack bits,
and rejected unsegmented depth with no authored visual program. The extended
source hash is
9db3f67b013daece14079c5de506a584591c2a3ec689d0eddbff9647da3e0c12
and the separate executable hash is
b124dfcf13bdbbe2f7ff9aa69e9564e0ec99f4ac668e3a92c2d266f6f4ad961f.
The original source and result remain identifiable at commit 3686ca8c.
