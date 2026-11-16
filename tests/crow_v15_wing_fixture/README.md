# Experimental v15 articulated wing mechanics

This opt-in fixture extends the existing v15 research body with an elbow, two wrist coordinates and three additional bodies per wing. The original rigid-wing flight actuator is removed because it cannot apply aerodynamics to these articulated links. No task or deployed controller has been migrated to this model. This fixture is not a flight, landing, balance, anatomical, or visual qualification.

The candidate preserves total wing mass but uses unmeasured segment mass fractions, box inertias, joint limits and actuation gains. Forearm and hand collisions use exploratory 3 mm radius capsule proxies with deliberate clearance near the wrist. The visible feather envelope, feather interactions, elastic coupling, actuator physiology and morphing-wing aerodynamics are absent. Three of 25 sampled joint poses penetrated another body beyond 0.5 mm. They must not be treated as valid just because the native solver accepts them.

The capacity expansion in this branch changes the shared MetalWorld and Matter ABIs. This is an isolated research branch; it must not be merged into production until all consumers and affected host/device layouts have been independently qualified. Do not reuse v15 policies or checkpoints as if they were compatible with this model. Native contract validation has rejected the old actor.

The compact CTest entries cover topology, mass conservation, a brief paired free-motion response, zero-input contact stability and one GPU versus FP64 reference comparison. The longer diagnostic programs in this directory are research probes, not qualification tests. See the separate evidence repository and its preregistrations for exact inputs, failed candidates, traces, numerical results and scope limitations.

To build and run the compact checks from an appropriately configured Apple Silicon build directory:

```sh
cmake --build build/wing-capacity-48 --target crow_v15_wing_structure_check crow_v15_wing_dynamics_check crow_v15_wing_reference_check
ctest --test-dir build/wing-capacity-48 --output-on-failure -R metalrobo.crow.v15_wing_
```

The `live-wing-kinematics.mm`, `articulated-wing-aero.metal`, `aero-wing-probe.mm`, `aero-passivity.mm`, `aero-timestep.mm` and `aero-controlled-fold.mm` files are experimental research probes with fixed fixture-specific body indices and three lanes. They are not a general-purpose aerodynamic API or a compatible flight controller. The airflow model is an uncalibrated quasi-steady engineering approximation without wakes, body drag, feather flexibility, or biological validation. Its free-wing four-microstep passivity screen failed; the narrower 32-microstep screen passed for the three tested starts, without proving trajectory convergence. Preserve and consult the preregistered experiments and their failures in the separately versioned evidence repository before drawing conclusions.
