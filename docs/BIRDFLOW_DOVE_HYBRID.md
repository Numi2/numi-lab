# BirdFlow dove hybrid

`birdflow_deetjen_dove_hybrid` is a Numi-native, trainable aerial robot
informed by BirdFlowMetal's Deetjen dove surface benchmark. It is deliberately
separate from the provenance-locked BirdFlow replay exposed by `numi dove
replay`.

## Runtime contract

The compiled RobotPack has a 0.32 kg free body, a static wing collision
silhouette, and four wing load stations. The existing device-resident aerial
actuator program produces forces and moments from its action, body velocity,
attitude, angular velocity, and configured wind within the same Metal
rollback transaction as contact, sensing, reward, termination, and reset.
This makes each control transition state- and load-responsive, and exposes a
four-channel policy action contract for MLX training.

```sh
numi dove train --envs 1024 --updates 1000
numi dove evaluate --checkpoint PATH
numi dove replay
```

`numi dove train` initializes the hybrid policy contract;
`numi dove evaluate` uses the same compiled robot/task route. The task is
station keeping at 1.5 m, with randomized initial height in [1.45, 1.55] m.

## Scientific boundary

This is not a claim of measured Deetjen free flight, articulated wing
kinematics, or a coupled BirdFlow D3Q19 solve. The public Deetjen surface
sequence provides a prescribed-motion CFD benchmark, but not the complete
same-specimen inertia and bilateral wing-mass record needed to identify a
calibrated free-flight model. The hybrid's mass, inertia, wing geometry, and
load coefficients are explicit model choices. Its runtime loads are not replay
forces, while its BirdFlow replay output remains evidence-only.

The next fidelity step is a device-resident, multi-environment D3Q19 coupling
to articulated wing state, qualified against held-out BirdFlow measurements
and deterministic replay before it replaces this hybrid path.
