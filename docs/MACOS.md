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

**File → Open Robot or Project** imports a fixed-base URDF and an optional
same-name SRDF from the same folder. Cooked scenes are stored in the scene
library. **File → Edit Experiment** saves a new copy with its own gravity,
friction, and effort amplitude. **File → Export Robot Project** creates a
portable `.numiproject` package with the source, assets, settings, and native
packs. Open the package to compile it into the local library. Each project
must include a movable, actuated joint with a positive authored effort limit
and at least one visual. Relative OBJ, STL, and GLB meshes must live under
the URDF directory. Visual primitives support box, sphere, and cylinder.
Collision meshes use closed convex OBJ or STL. Start with **Bench Pendulum**
from the robot selector.

**View → Export State Samples** writes up to the latest 300 samples, taken
about five times a second while the inspector is open. Headless
`--state-trace` and `--contact-trace` capture every control tick. The
[Python experiment API](#script-a-run) can compile and run the same project.

**File → Open Panda Model** imports the pinned stock Panda from MuJoCo
Menagerie into the native tabletop grasp workflow. The [Panda example](../examples/panda_tabletop/README.md)
lists the supported model revision and the headless comparison commands.

The bundled X500 Flight Demo repeats a climb, turn, and descent from a
fixed camera. All four propellers follow the simulated motor speeds and
turn in their modeled directions. The viewer draws them at 1% speed so
their rotation is visible at the display frame rate. Pause or single-step
the demo with the same controls.

## Build and check

From the source checkout on Apple silicon with Xcode, CMake 3.28 or later,
and Ninja:

```sh
./tools/package_macos.sh /path/to/output
```

The command writes a signed local-preview ZIP and a SHA-256 checksum.
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
last row when the sequence ends. See the [matched MuJoCo fixture](../examples/bench_pendulum/README.md)
and [iOS target](IOS.md).
