# Numi Lab for Mac

Numi Lab runs on Apple silicon with macOS 26 or later. Expand the preview
ZIP and open **Numi Lab.app**. The app starts a PX4 X500 flight demo
with the bundled native engine, Metal shaders, and scene assets. The
preparation indicator stays visible while Metal initializes.

| Control | Action |
| --- | --- |
| Pause / Resume or Space | Hold the current frame and pause physics, or continue the run. |
| Restart or Command-R | Return the current simulation to its initial state and resume it. |
| Period while paused | Advance one control tick. |
| View → Simulation State or Command-I | Show accepted state, select a coordinate or contact force to plot, and collect CSV samples. |
| Drag, Shift-drag, scroll | Orbit, pan, and zoom the camera while running or paused. |
| Reset View, R, or Command-0 | Return to the scene's authored camera. |
| Command-W or Command-Q | Close the simulation and exit the app. |

**File → Show Scene Library** opens `~/Library/Application Support/Numi Lab/Scenes`.
Copy a compiled Numi Window catalog there with its asset folders intact, then
reopen the app. The app adds the catalog's robot, environment, and task options
to the toolbar. Compatible policy packs can live in the sibling `Policies`
folder. Use the [source viewer](NUMI_CLI.md) for building catalogs and training.

**File → Open Robot or Project** imports a fixed-base URDF with its optional
same-name SRDF or a supported MuJoCo MJCF XML model. Cooked scenes are stored
in the scene library. **File → Edit Experiment** saves a new copy with its own gravity,
friction, and effort amplitude. **File → Export Robot Project** creates a
portable `.numiproject` package with the source, assets, settings, and native
packs. Open the package to compile it into the local library. Each project
must include a movable, actuated joint with a positive authored effort limit
and at least one visual. Relative OBJ, STL, and GLB meshes must live under
the URDF directory. Visual primitives support box, sphere, and cylinder.
Collision meshes use closed convex OBJ or STL. Start with **Bench Pendulum**
from the robot selector.

### MJCF support

Use a standalone `.xml` file with named fixed-base bodies, explicit mass
and inertia, up to one hinge or limited slide per body, and at least one
direct joint motor. Motors use positive scalar gear and symmetric control
limits. The importer preserves joint pivots, quaternion orientations, limits,
damping, and the authored downward gravity and timestep. The timestep must
divide the 20 ms control period into 1–100 equal substeps. Body geoms can be
boxes, spheres, and cylinders with standard collision masks or both masks
set to zero for visual-only geometry. An optional horizontal world plane at
Z = 0 uses Numi's ground contact. The viewer still shows its stage grid for
models without a colliding ground.

Normalized action rows follow the MJCF motor declaration order. The project's
`project.json` lists each channel's joint, effort limit, and authored motor
name when present. When recooking an older project, use the regenerated
channel map to align existing action clips.

Models with defaults, includes, meshes, floating joints, custom contact
parameters, or other unsupported MJCF features report an error with the XML
line that needs attention. Numi Lab runs imported projects with its own Metal
solver and contact model. Exported MJCF projects retain the input XML and the
compiled model. The [comparison fixture](../tests/fixtures/workbench_pendulum.xml)
is a small example you can open directly in the app.

**View → Export State Samples** writes up to the latest 300 samples, taken
about five times a second while the inspector is open. Headless
`--state-trace` and `--contact-trace` require `--chunk 1` and sample every
control tick. Contact readback reports the active contact count and strongest
normal force from the last accepted physics substep. Add `--contact-body NAME`
to restrict both values to a body. The
[Python experiment API](#script-a-run) can compile and run the same project.

**File → Open Panda Model** imports the pinned stock Panda from MuJoCo
Menagerie into the native tabletop grasp workflow. The [Panda example](../examples/panda_tabletop/README.md)
lists the supported model revision and the headless comparison commands.

The bundled X500 Flight Demo repeats a climb, turn, and descent from a
fixed camera. All four propellers follow the simulated motor speeds and
turn in their modeled directions. The viewer draws them at 1% speed so
their rotation is visible at the display frame rate. Pause or single-step
the demo with the same controls.

The Mac viewer renders at 1920 × 1080 on launch. Set
`NUMI_MAC_RENDER_720P=1` when launching the app executable from Terminal
to use 1280 × 720. Run with `NUMI_MAC_PROFILE_PLAYBACK=1` to log completed
presentations and simulation steps during continuous playback.

## Build and check

From the source checkout on Apple silicon with Xcode, CMake 3.28 or later,
and Ninja:

```sh
./tools/package_macos.sh /path/to/output
```

The command checks the packaged app on Metal before writing a signed
local-preview ZIP and a SHA-256 checksum.
The default package includes precompiled Metal pipelines for the base M1
through M5 GPUs and a shader library for runtime compilation. To build a
smaller local preview that compiles pipelines on first use, configure CMake
with `-DNUMI_MAC_AOT_PIPELINES=OFF`.

The local preview uses an ad-hoc signature. For distribution with Gatekeeper,
set `NUMI_MAC_SIGN_IDENTITY` to a Developer ID Application identity and
`NUMI_MAC_NOTARY_PROFILE` to a configured `notarytool` Keychain profile.
The packaging tool submits, staples, and verifies that build.

After extracting the archive, run the native smoke check from a path outside
the checkout. It requires a logged-in macOS desktop and uses real Metal
simulation and presentation.

```sh
"/path/to/Numi Lab.app/Contents/MacOS/metalrobo_task_rollout" --check
```

The check waits for the first frame, restarts the simulation, verifies recovery
from an invalid scene selection, and renders each available bundled scene.

### Precompile pipelines for the base M6 GPU

For a build toolchain that supports `applegpu_g18g`, append that architecture to
`NUMI_MAC_GPU_ARCHITECTURES` at CMake configure time. With an older toolchain,
build the bake executable and inputs on the build Mac:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0
cmake --build build --target numi_lab_macos_pipeline_inputs
```

Copy `build/metalrobo_numi_mac_bake_pipelines`, `build/shaders/MetalRobo.metallib`
and `build/shaders/NumiLab.mtlp-json` into one directory on a base M6 Mac. Run
the following there:

```sh
./metalrobo_numi_mac_bake_pipelines MetalRobo.metallib NumiLab.mtlp-json \
    MetalRobo-G18G.binary.metallib
shasum -a 256 MetalRobo.metallib NumiLab.mtlp-json > inputs.sha256
```

Copy the resulting archive and `inputs.sha256` back into one directory on the
build Mac. Package from the original build directory with
`NUMI_MAC_BUILD_DIR=build NUMI_MAC_G18G_ARCHIVE_DIR=/absolute/path/to/capture`
in the packaging command's environment. CMake checks both input hashes and
the GPU architecture before adding the capture to the offline archive.

Verify a fresh launch on the M6 with `NUMI_MAC_PROFILE_PIPELINES=1`. With the
app's cache absent and its user catalogs isolated, the viewer should report an
archive hit for every pipeline and promptly present its first frame.


## Script a run

```python
from pathlib import Path
from metalrobo.experiment import Experiment

experiment = Experiment.compile(
    Path("assets/window/bench-pendulum.urdf"),
    Path("My Pendulum.numiproject"),
    app=Path("/Applications/Numi Lab.app"),
    gravity=9.81,
    effort=0.12,
)
result = experiment.run(steps=200, state_trace=Path("state.tsv"))
print(result.stats["environment_control_steps_per_second"])
```

Supply `actions=[(0.1,), (0.2,)]` to drive a one-actuator project directly.
The native runner broadcasts each row across environments and holds the
last row when the sequence ends. A result from explicit action rows retains
the packed bytes in `result.action_clip` and records their SHA-256 under
`result.stats["action_clip_sha256"]`. Save the bytes with
`Path("actions.f32").write_bytes(result.action_clip)` and pass that path as
`action_clip` to replay them. Pass an MJCF XML path to the same
`Experiment.compile` call to run an imported MJCF model. The authored
gravity and timestep apply unless you pass `gravity` to `compile` or
`physics_substeps` to `run`. See the [matched MuJoCo fixture](../examples/bench_pendulum/README.md)
and [iOS target](IOS.md).
