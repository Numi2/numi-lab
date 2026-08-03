# Franka directional exploration

This capability uses the joint targets and minimum-jerk timing from Franka
Robotics' official `pylibfranka` joint-impedance example at libfranka revision
`85912fe02258d8cb811d3eff1f11e52ce89e3217`. The sequence is home, forward,
right, left, and home, with a three-second transition and half-second dwell.

The Apple-native path executes the trajectory through NumiSolver's implicit
position drives and emits exact replay plus physical end-effector evidence:

```sh
numi franka-explore
```

`libfranka` is the real-robot transport, state, model, limit, and control
boundary. It is deliberately built separately on a supported real-time Ubuntu
host, because the main MetalRobo project targets Apple Silicon macOS:

```sh
cmake -S examples/franka_explore/libfranka \
  -B build/franka-libfranka -DCMAKE_BUILD_TYPE=Release
cmake --build build/franka-libfranka -j
```

Hardware motion remains disarmed unless every explicit owner acknowledgement
is present:

```sh
build/franka-libfranka/numi_franka_libfranka_explore \
  --robot-ip ROBOT_IP \
  --arm-hardware \
  --confirm-free-space \
  --confirm-user-stop
```

The bridge explicitly enables libfranka's command rate limiter and 100 Hz
low-pass filter. It does not perform automatic error recovery, silently weaken
collision thresholds, or infer that a simulator run is hardware evidence.

## Cinematic render

The retained GIF uses the official FER visual meshes from
`franka_description` 2.8.1, exact link-frame poses exported from the accepted
Numi Lab trajectory, and a Blender studio scene. The pose exporter recovers
URDF link origins from Numi's COM-centred body state; the render path preserves
the mesh hierarchy and explicitly cancels the Collada-to-glTF Y-up transport
rotation before applying Numi's Z-up poses.

```sh
git clone --depth 1 --branch 2.8.1 \
  https://github.com/frankarobotics/franka_description.git \
  build/franka_description

numi franka-explore \
  --output build/franka-exploration.csv \
  --pose-output build/franka-exploration-poses.csv

python3 -m venv build/franka-render-venv
build/franka-render-venv/bin/pip install trimesh pycollada pygltflib
build/franka-render-venv/bin/python \
  examples/franka_explore/convert_meshes.py \
  --franka-description build/franka_description \
  --output build/franka-render-meshes

build/franka-render-venv/bin/python \
  examples/franka_explore/validate_render_geometry.py \
  --poses build/franka-exploration-poses.csv \
  --converted-meshes build/franka-render-meshes \
  --step 174

blender --background --python examples/franka_explore/render_gif.py -- \
  --poses build/franka-exploration-poses.csv \
  --franka-description build/franka_description \
  --converted-meshes build/franka-render-meshes \
  --frames build/franka-render-frames \
  --frame-count 96 --start-step 174 --width 960 --height 540

ffmpeg -y -framerate 24 \
  -i build/franka-render-frames/franka-%04d.png \
  -vf 'palettegen=max_colors=256:stats_mode=diff' \
  build/franka-render-palette.png
ffmpeg -y -framerate 24 \
  -i build/franka-render-frames/franka-%04d.png \
  -i build/franka-render-palette.png \
  -lavfi 'paletteuse=dither=sierra2_4a:diff_mode=rectangle' \
  -loop 0 docs/media/numi-lab-franka-exploration.gif
```

Step 174 is the measured end of the official home waypoint. Starting there
omits the model-default-to-home pre-roll, so the visible sequence is the closed
home, forward, right, left, home loop. Geometry validation requires every
parent/child mesh pair to meet within 12 mm; the retained home frame's maximum
measured adjacent gap is 4.648 mm at the hand/finger boundary (the seven arm
boundaries are at or below 1.001 mm).

The result is simulator visualization, not evidence of hardware execution.
