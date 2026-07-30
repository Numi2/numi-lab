# Franka tactile grasp-stabilization example

This example is executable through
`metalrobo_franka_tactile_example`. It uses the normal authored
`EpisodeTwin -> WorldTemplate -> WorldFamily -> MRWorldPack` pipeline and two
32×32 Franka fingertip tactile sensors.

The first MetalWorld step establishes bilateral contact with a dynamic grasp
coupon. The resulting metric maps and solved contact impulses produce depth,
wrench, centroid, area, and center-of-pressure observations. A small
deterministic policy closes each finger in proportion to its own mean-depth
deficit and is evaluated against an identical open-loop continuation.

The checked-in `policy_checkpoint` directory demonstrates the exact metadata
that accompanies a policy. Loading must reject a tactile fingerprint mismatch.
It does not contain trained weights and makes no real-hardware transfer claim.

Run:

```sh
./build/bin/metalrobo_franka_tactile_example
./build/bin/metalrobo_franka_tactile_example \
  --debug-dir /tmp/metalrobo-tactile-debug
```

The debug directory contains one 16-bit depth PGM, validity PGM, metric CSV,
3D OBJ query scene, and summary JSON per fingertip. These files are optional
inspection artifacts; the RL path consumes native Metal buffers.
