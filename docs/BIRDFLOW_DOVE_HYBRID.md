# BirdFlow dove hybrid

`birdflow_deetjen_dove_hybrid` is a Numi-native, trainable aerial robot
informed by BirdFlowMetal's Deetjen dove surface benchmark. It is deliberately
separate from the provenance-locked BirdFlow replay exposed by `numi dove
replay`.

## Runtime contract

The compiled RobotPack has a 0.28 kg free airframe and two 0.02 kg wing
bodies, each connected by an actuated revolute hinge. Its two policy actions
target those hinges. A device-resident blade-element pass reads the resolved
root pose/velocity, hinge angle/rate, and configured wind, then writes
per-wing world-frame wrenches before the generic articulated ABA pass. It is
therefore inside the same Metal rollback transaction as contact, sensing,
reward, termination, and reset: the next state changes with wing load rather
than receiving a fixed action-to-force mapping.

The runner also provides `--birdflow-flap-script` as a deterministic 4 Hz
open-loop qualification input. It is only a wiring/physics probe for the
normal two-action policy contract, not a controller or an evaluation claim.

```sh
numi dove train --envs 1024 --updates 1000
numi dove evaluate --checkpoint PATH
numi dove replay
```

`numi dove train` initializes the hybrid policy contract;
`numi dove evaluate` uses the same compiled robot/task route. The task is
station keeping at 1.5 m, with randomized initial height in [1.45, 1.55] m.

## Scientific boundary

This is not a claim of measured Deetjen free flight or a coupled BirdFlow
D3Q19 solve. The public Deetjen surface sequence provides a prescribed-motion
CFD benchmark, but not the complete same-specimen inertia and bilateral
wing-mass record needed to identify a calibrated free-flight model. The
hybrid's mass, inertia, geometry, passive-feathering closure, and aerodynamic
coefficients are explicit model choices. Its runtime loads are not replay
forces, while its BirdFlow replay output remains evidence-only.

The next fidelity step is a device-resident, multi-environment D3Q19 coupling
to articulated wing state, qualified against held-out BirdFlow measurements
and deterministic replay before it replaces this hybrid path.
