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
| View → Simulation State or Command-I | Show joint coordinates, scene position, and live contact diagnostics. |
| Drag, Shift-drag, scroll | Orbit, pan, and zoom the camera while the simulation is running. |
| Reset View, R, or Command-0 | Return to the scene's authored camera. |
| Command-W or Command-Q | Close the simulation and exit the app. |

**File → Show Scene Library** opens `~/Library/Application Support/Numi Lab/Scenes`.
Copy a compiled Numi Window catalog there with its asset folders intact, then
reopen the app. The app adds the catalog's robot, environment, and task options
to the toolbar. Compatible policy packs can live in the sibling `Policies`
folder. Use the [source viewer](NUMI_CLI.md) for building catalogs and training.

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

The command writes a signed local-preview ZIP and a SHA-256 checksum. The
app includes precompiled Metal pipelines for Apple silicon GPU families
and the shader library for devices that need runtime compilation.

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

The check waits for the first frame, restarts the simulation, and verifies
that the app recovers from an invalid scene selection.
