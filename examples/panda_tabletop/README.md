# Panda Tabletop

Open **File → Open Panda Model** in Numi Lab and select `panda.xml` or
`scene.xml` from the pinned [MuJoCo Menagerie Panda model](https://github.com/google-deepmind/mujoco_menagerie/tree/eea4c7431b63da8278b0ff31638e0b0e941cec0e/franka_emika_panda).
Keep the stock `assets` directory beside the XML. The app saves the
compiled scene at `~/Library/Application Support/Numi Lab/Scenes/Panda Tabletop`.
The task starts with open fingers, approaches the cube, closes, lifts, and
holds. The app pauses at the final frame. **Command-R** replays the task
from the same initial pose.

- **Space** pauses or resumes. **Period** advances one 20 ms control tick
  while paused.
- **Command-I** shows joint coordinates and the cube position. The contact
  count covers the whole scene. The peak normal load is the largest solved
  normal impulse on the cube divided by the last physics substep duration.
  The body pair identifies the contact that produced it.

The importer accepts the unmodified stock Panda files from the linked revision.
It reads the MJCF's visual mesh bindings and materials. The compiled WorldPack
uses Numi's native Panda dynamics, fingertip colliders and independent finger
drives. The fixed mount uses the link origin as in the stock model. The cube
interacts with the fingers and workcell surfaces. MuJoCo's finger equality,
tendon drive and collision geometry remain in the stock reference. The
cross-engine check evaluates completion under shared joint targets.

## Headless replay and comparison

Use the scene the app compiled and the example Python runner in
`examples/panda_tabletop/compare.py`. It runs three native rollouts with
the same 370-tick arm trajectory, including an open-gripper control. Pass
`--mujoco-panda` with the original `panda.xml` to add the reference
(`python3 -m pip install mujoco==4.0.1` in a virtual environment). The
native-only run uses the Python 3.11+ standard library.

```sh
app="/path/to/Numi Lab.app"
scene="$HOME/Library/Application Support/Numi Lab/Scenes/Panda Tabletop/panda-tabletop.numi-window.json"

python3 compare.py "$scene" \
  --rollout "$app/Contents/MacOS/metalrobo_task_rollout" \
  --metallib "$app/Contents/Resources/MetalRobo.metallib" \
  --mujoco-panda "/path/to/Panda Model/panda.xml" \
  --output /path/to/new-panda-results
```

The output contains state and contact traces, runtime and artifact identities,
and a JSON summary. The checks require zero failed native steps, identical
native state and contact replays, at least 60 mm of sustained lift during the
last second, a fingertip as the strongest loaded cube contact on every
hold sample, and at most 5 mm of lift in the open-gripper control. If
requested, the MuJoCo run uses the same 60 mm hold threshold. Each control
tick has four 5 ms physics substeps. The recorded runtime includes tracing.

You can also compile without opening the app. Use an empty output directory.

```sh
"$app/Contents/Helpers/metalrobo_numi_window_catalog_compile" \
  --panda-model "/path/to/Panda Model/scene.xml" \
  --output /path/to/new-panda-scene \
  --metallib "$app/Contents/Resources/MetalRobo.metallib"
```

To repeat the same trajectory with a different object, pass
`--panda-cube-mass` in kilograms and `--panda-cube-friction` as a coefficient
while compiling. Use a fresh output directory per variant. The comparison
reads those values from the compiled scene and applies them to the MuJoCo cube.
