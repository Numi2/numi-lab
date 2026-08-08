# BirdFlow dove hybrid

`birdflow_deetjen_dove_hybrid` is a Numi-native, trainable aerial robot
informed by BirdFlowMetal's Deetjen dove surface benchmark. It is deliberately
separate from the provenance-locked BirdFlow replay exposed by `numi dove
replay`.

## Runtime contract

The compiled RobotPack has a 0.27 kg free airframe, two 0.02 kg wing bodies,
and a 0.01 kg fixed tail body. The wings are each connected by an actuated
revolute hinge. Its two policy actions
modulate bilateral stroke amplitude; the robot-owned 4 Hz cyclic phase drives
the hinge targets. The actor observes both wing position/rate and the sin/cos
phase, so an MLX policy does not have to infer a hidden wingbeat from action
lag. A device-resident blade-element pass reads the resolved root
pose/velocity, hinge angle/rate, and configured wind, then writes per-wing
world-frame wrenches before the generic articulated ABA pass. It is therefore
inside the same Metal rollback transaction as contact, sensing, reward,
termination, and reset: the next state changes with wing load rather than
receiving a fixed action-to-force mapping.

The zero policy action is an authored 0.88 normalized stroke trim, with a
bounded ±0.04 bilateral residual. This puts initial MLX exploration in the
measured viable-stroke band rather than at a no-lift half-stroke. The task also
terminates at 0.15 m or 2.50 m root height, so exploratory excess lift becomes
a normal reset event instead of an unbounded trajectory.

The fixed tail is also a resolved aerodynamic surface: its force uses the
root-relative airflow at its authored aerodynamic center, while its bounded
pitch-damping term uses the active wing-stroke dynamic pressure. Its external
wrench is written to the tail body and transferred through the fixed
articulation by ABA. This gives the learner a load-responsive pitch channel;
it is not a separate controller and never supplies a replayed force trace.

The airframe itself receives a resolved quadratic drag wrench. Its authored
forward/span/up reference areas are evaluated in the live body frame, and its
angular-rate damping is applied as a root torque. Thus translation and
rotation can dissipate aerodynamic energy even during a symmetric wingbeat;
these coefficients are hybrid model parameters, not measured Deetjen values.

The runner also provides `--birdflow-flap-script` and
`--birdflow-stroke-amplitude` as deterministic open-loop qualification inputs.
They exercise the same trim-residual contract as the policy and are not
controllers or evaluation claims.

```sh
numi dove train --envs 1024 --updates 1000
numi dove evaluate --checkpoint PATH
numi dove replay
numi dove measured-audit --input assets/birdflow/numi-private-dove-d3q19-v1.json
numi dove measured-flight --input assets/birdflow/numi-private-dove-d3q19-v1.json --steps 64
```

`numi dove train` initializes the hybrid policy contract;
`numi dove evaluate` uses the same compiled robot/task route. The task is
station keeping at 1.5 m, with randomized initial height in [1.45, 1.55] m.

## Coupled D3Q19 calibration path

`numi dove measured-*` invokes BirdFlowMetal's schema-2 measured-bird route,
not the Numi hybrid. It accepts only a fully declared complete-bird input:
registered geometry, left/right periodic kinematics and rates, whole-bird
mass/COM-frame inertia, bilateral wing mass/hinge-COM/inertia, and the fluid
condition. The stages are deliberately separate: `measured-audit` rejects an
invalid contract before Metal allocation; `measured-trim` finds a reproducible
prescribed forward-flight balance; `measured-flight` records a coupled D3Q19
six-DOF trajectory with momentum and per-part-load ledgers; and
`measured-confirm` runs its independent bounded-flight, 1/2/4 body-step, and
closure gates.

BirdFlow also supports the controlled `--pre-roll-cycles N` release protocol:
it advances one or more prescribed wingbeats in the same resident D3Q19 volume
while the body remains fixed, then releases the six-DOF body. It never
discards fluid, wing-momentum, or force state, and its step count is explicit
in the archive. It remains opt-in until the full pre-roll numerical-stability
gate passes for the particular specimen and resolution.

`assets/birdflow/numi-private-dove-d3q19-v1.json` is a complete private
calibration fixture constructed from Numi's explicitly authored bilateral wing
mechanics plus an explicit lightweight virtual-airframe and rigid-wing program.
Its analytic-proxy wing and tail
thicknesses are deliberately resolved at BirdFlow's D3Q19 refinement ladder;
they are numerical boundary envelopes, not physical feather-thickness
measurements. Its low-amplitude rigid stroke is explicitly a D3Q19 conditioning
choice after the original full Numi hinge range made the moving-wall fluid
state non-finite. Its 1 m/s, Re=100 first flight condition is likewise a
viscosity-resolved numerical calibration point, not an atmospheric or Deetjen
condition. It is not presented as a Deetjen kinematic trace. Its values are sufficient for
exercising the actual D3Q19 free-flight pipeline, but are not biological
measurements and must not be relabelled as Deetjen data or as a real-bird
prediction. A real same-specimen archive may use the same command only after
replacing every authored property with its measured provenance.

## Scientific boundary

This is not a claim of measured Deetjen free flight or a coupled BirdFlow
D3Q19 solve. The public Deetjen surface sequence provides a prescribed-motion
CFD benchmark, but not the complete same-specimen inertia and bilateral
wing-mass record needed to identify a calibrated free-flight model. The
hybrid's mass, inertia, tail/wing geometry, passive-feathering closure, and
aerodynamic coefficients are explicit model choices. Its runtime loads are
not replay forces. The tail's relative-airflow and wing-wash damping closure
is likewise authored rather than identified from Deetjen force data. The
open-loop full-stroke input is a load-path qualification probe, not evidence
of a trimmed, trained, or free-flight controller; BirdFlow replay output
remains evidence-only.

The next fidelity step is a device-resident, multi-environment D3Q19 coupling
to articulated wing state, qualified against held-out BirdFlow measurements
and deterministic replay before it replaces this hybrid path.
