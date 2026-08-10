# Numifly

Numifly is a separate robot identity, not a mode of `unitree_g1`. It combines
the 29-DoF Unitree G1 articulation at 0.09 linear scale with a bilateral
measured-surface wing actuator mounted on `torso_link`:

- 29 articulated position-drive actions;
- 20 measured-wing residual actions;
- 49 actions in the compiled flight task;
- density-preserving G1 mass and inertia scaling;
- a distinct RobotPack, task, C/Swift source, fingerprints, and Numi Window
  catalog scene.

## Data selection and provenance

The wing input is Maeda et al., *Quantifying the dynamic wing morphing of
hovering hummingbird* (Royal Society Open Science, 2017),
[article DOI](https://doi.org/10.1098/rsos.170307)
`10.1098/rsos.170307`,
[data DOI](https://doi.org/10.6084/m9.figshare.5406124.v1)
`10.6084/m9.figshare.5406124.v1`, CC BY 4.0.
The source reconstructs the measured right wing of an *Amazilia amazilia*
hummingbird through one hovering stroke. The checked-in source artifact has
17 phases, a 21-by-41 chord/span grid, 861 vertices per phase, and a measured
28.8 Hz wingbeat.

`tools/build_numifly_maeda_pack.py` verifies the source archive hash and emits
the immutable runtime pack in `assets/numifly/maeda-wing-pack-v1`. Its explicit
design transforms are:

- preserve the measured right-wing topology and phase sequence;
- derive the left wing by sagittal `y` reflection and reverse its winding;
- scale both wing surfaces by 2.5;
- translate them to the scaled G1 torso-back mount;
- map the 17 measured phases uniformly onto one periodic 28.8 Hz cycle,
  explicitly time-reparameterizing the larger source wrap interval;
- export frame-major positions, fixed bilateral topology, a phase-zero visual
  mesh, source/design metadata, and SHA-256 identities.

The right wing is measured. The left wing, robot scale, wing scale, torso
mount, aerodynamic coefficients, and residual controls are derived design
choices. This is not a measured complete bird, and mirrored bilateral symmetry
is not biological left-wing measurement.

The repository also contains a higher-temporal-resolution Deetjen dove surface
pack. That source is useful for complete-bird flight and transfer studies, but
it is not substituted into Numifly because this robot intentionally binds the
requested Maeda wing geometry and provenance.

## Live physics path

The measured-surface Metal pass runs before ABA on every physics substep. It
evaluates the current and candidate wing surfaces, relative air velocity, drag
loading, and the resulting force and torque in the floating-root frame. The
wrench targets `torso_link` in the ordinary articulated-body wrench buffer and
is therefore solved with the scaled G1 articulation and contact world. The
current model does not add waist-relative torso velocity or orientation to the
aerodynamic frame; this approximation preserves the qualified root-owned bird
kernel while still applying the load physically through the torso body.
Invalid transforms, actions, surface states, loads, or candidate physics reject
the environment transactionally.

The 20 wing controls are the live prefix of the measured-surface action ABI:
rhythm, flap/glide, and eight bounded residual modes for each wing. Zero
residual action replays the measured wingbeat. No tail surface or tail actions
are invented.

## Scale and executable evidence

An 18%-scale design was rejected for flight because its 0.194 kg mass produced
1.91 N weight, above the measured-wing sweep's 1.13 N sampled maximum. At 9%
scale, the production morphology is 0.0243 kg and weighs 0.238 N.

The deterministic native-Metal authority probe uses a gravity-free 10,000x
inertial force-balance fixture so a complete wingbeat can be integrated without
vehicle attitude contaminating the load measurement. It does not replace the
production free-flight rollout. With 256 candidates, 18 control steps, and
four physics substeps it measured:

- neutral mean force `(-0.0460, 0.0000, 0.2620) N`;
- best sampled upward force `1.1344 N`, or 4.76 times robot weight;
- near-weight candidate upward force `0.2335 N`;
- bit-identical signed-load replay;
- zero failed environment steps.

The ordinary gravity/contact rollout completes with zero failed physics steps,
but neutral replay is not stable flight: it accumulates tilt and task resets.
A one-update native Metal/MLX PPO smoke run completed 128 samples, advanced
policy revision 1 to 2, and reported zero failed environment steps. This proves
the simulation and learning path executes; it does not prove a trained hover
or hardware flight policy.

## Commands

Regenerate and verify the pack:

```sh
python3 tools/build_numifly_maeda_pack.py \
  /Users/home/BirdFlowMetal/ValidationInputs/maeda-hovering-right-wing-surface-v1.json \
  assets/numifly/maeda-wing-pack-v1

./build/bin/metalrobo_measured_surface_robot_probe \
  --numifly assets/numifly/maeda-wing-pack-v1/manifest.json 256 18
```

Run the separate robot:

```sh
./build/bin/metalrobo_task_rollout \
  --robot-source numifly --task numifly-flight --scene ground \
  --numifly-wing-manifest \
    assets/numifly/maeda-wing-pack-v1/manifest.json \
  --envs 1 --steps 64 --chunk 1 --zero-actions \
  --metallib build/shaders/MetalRobo.metallib
```

Train from a fresh policy. Numifly defaults to a `-4.0` initial log standard
deviation unless explicitly overridden because the exact wingbeat is already
near weight support and full-size G1 exploration is too aggressive for the
small articulation.

```sh
./build/bin/metalrobo_task_train \
  --task numifly-flight --scene ground \
  --numifly-wing-manifest \
    assets/numifly/maeda-wing-pack-v1/manifest.json \
  --initialize-policy numifly_flight_v1 \
  --policy-pack /tmp/numifly-initial.policypack \
  --updated-policy-pack /tmp/numifly-updated.policypack \
  --rollout-pack /tmp/numifly.rolloutpack \
  --mlx-python python/.venv/bin/python --python-root python \
  --native-library build/lib/libmetalrobo.dylib \
  --metallib build/shaders/MetalRobo.metallib
```

## Presentation boundary

The Numi Window catalog contains `numifly-ground-flight` and a separate
9%-scale visual pack. The phase-zero bilateral wing mesh is visibly attached
to the torso. The presentation wing is currently static; live wing deformation
and aerodynamic forces are in the Metal physics path, not yet streamed into
the visual instance. The aerodynamic sheet also has no authored wing collision
thickness. Neither limitation is relabelled as visual or contact completion.
