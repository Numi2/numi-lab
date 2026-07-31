# README media provenance

Every image referenced by the project README is a native MetalRobo
`sensor_reference` output captured on Apple M4. There is no generated concept
art and no screenshot from another simulator. RGB previews use the same
scene-linear radiance returned to the sensor contract; ACES tone mapping and
high-quality downsampling are presentation-only.

The raw frame and cooked robot/environment packs are transient build
artifacts. Only the compact documentation images are retained in this
repository.

## G1 standing-policy transfer animation

`g1-standing-native-20s.gif` and `g1-standing-native-20s.mp4` show an actual
one-environment, zero-command MetalRobo rollout driven by the official Unitree
MuJoCo-Lab actor. The source checkout is pinned at revision
`1425b15f73bd4095f0df53709d7c389c3eb9e790`; its ONNX policy SHA-256 is
`2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`.
The actor was converted once into the generic native PolicyPack format and ran
through MetalRobo's Metal inference engine.

The rollout used the production `temporalCone` contact path, flat ground, a
20 ms control interval, four physics microsteps, seed `20260731`, and a zero
velocity command. All 1,000 control steps completed: mean tilt was `0.009 rad`,
mean pelvis height was `0.785 m`, tracking was `0.9998`, no physics step failed,
and the only termination was the expected 20-second episode timeout.

Every displayed frame is a native MetalRobo `sensor_reference` capture on
Apple M4. The official G1 mesh geometry was posed from MetalRobo's final-q
state, cooked through `metalrobo_visual_cook`, and rendered against the same
Studio Small 03 environment used by the static README captures. The 100 poses
sample every tenth control step and are displayed at 5 Hz for exactly 20
seconds. The H.264 stream repeats each displayed pose to produce a broadly
compatible 25 fps stream; it does not interpolate simulator state. MuJoCo is
not used for these pixels. ACES preview conversion and encoding are the only
presentation-only stages. The retained hashes are:

```text
g1-standing-native-20s.gif=sha256:5b06ae06b9f39f429ee0794ac75dc0f69de0c76244546e4f2b4f7a17ccce3a28
g1-standing-native-20s.mp4=sha256:05b47c635024a25864b63e806fa23105ae8ae5ad014e72275903aa54cfb1b514
```

This validates the native 20-second simulation gate. It does not establish
real-robot transfer.

## G1 dynamic-ball disturbance animation

`g1-standing-ball-disturbance-20s.gif` and
`g1-standing-ball-disturbance-20s.mp4` use the same policy, zero command,
20 ms control interval, four microsteps, and native rendering path as the
standing capture above. Four authored dynamic spheres enter the shared scene
body/contact graph with masses `0.08`, `0.06`, `0.08`, and `0.05 kg`. Their
radii are `0.12`, `0.11`, `0.13`, and `0.10 m`; launch speed is `3 m/s` in the
horizontal plane with ballistic vertical velocity.

The traced contacts occur at `0.36`, `0.48`, `0.56`, and `0.68 s`. The
1,000-step rollout reports zero failed physics steps, maximum tilt
`0.0167597 rad`, 19 maximum active contact blocks, nine maximum manifolds, and
only the expected episode timeout. Tilt remains below `0.012 rad` for a
sustained half-second beginning at `0.92 s`. All robot and sphere poses come
from the accepted MetalRobo state trace. The visual spheres are authored
colored meshes cooked together with the official G1 visual scene; collision
geometry is not used as a presentation fallback.

The 100 displayed states sample every tenth control step. MuJoCo supplies no
pixels, contact, motion, or state. Retained hashes are:

```text
g1-standing-ball-disturbance-20s.gif=sha256:88ba9eeb66280de173c4fdbcb2e643affaa2cdec4aff54134deaad973029d7f1
g1-standing-ball-disturbance-20s.mp4=sha256:e1fcce61903081f8a7a2640dd140ef8354a7155e160a73769f6f5f2de1dc53e5
```

This is simulator disturbance-recovery evidence, not a hardware claim.

## Native G1 policy-rollout animations

`numi-lab-g1-native-rollout.gif` and
`numi-lab-g1-sensor-rollout.gif` are native numi-lab captures, not generated
images. The source was a 96-control-step, one-environment G1 terrain rollout
executed on Apple M4 through `metalrobo_task_rollout` with the compiled native
policy path, TGS, eight-step submissions, and seed `20260731`.

The animation uses all 48 consecutive rollout steps from 7 through 54. Each
displayed joint configuration is reconstructed from the rollout's clean
critic joint-position state plus the pinned G1 reset pose. This removes
policy-observation noise from the presentation without filtering, smoothing,
or synthesizing motion. The robot root is held fixed for the presentation
camera. Each 20 ms simulator state is displayed for 40 ms, giving each
720x405 GIF a 48-frame, 1.92-second half-speed loop.

Each pose was assembled from the official G1 visual geometry, cooked through
`metalrobo_visual_cook`, and rendered through numi-lab
`sensor_reference` against the same Studio Small 03 environment used by the
static README captures. The sensor animation reads scene-linear RGB, metric
depth, camera-space normals, and packed authored identities from each native
frame. ACES preview conversion, panel layout, labels, half-speed playback, and
GIF encoding are presentation-only. Both GIFs use a single sequence-wide
palette so material highlights and sensor colors do not flicker between
frames.

The retained provenance is:

```text
rollout pack fingerprint=8243761031265860489
policy fingerprint=7205423632253474391
task fingerprint=15986245760138582396
maximum active contacts=13
failed environment steps=0
numi-lab-g1-native-rollout.gif=sha256:264bb2005ed6cf47af4f0c40d6dca8ab497a9d590dc3037a3db818396b40b7f8
numi-lab-g1-sensor-rollout.gif=sha256:08dc33bb42658a8478687651354d813cf4dff55d40b5270f3b8c7857d79a7ff2
```

This capture validates presentation and rollout integration only. It does not
establish policy quality, locomotion success, or sim-to-real transfer.

## Shared HDR environment

Both robot scenes use Greg Zaal's
[`Studio Small 03`](https://polyhaven.com/a/studio_small_03) HDRI from Poly
Haven under CC0. The 2K Radiance HDR source has SHA-256
`156c2946a6cd0c1d8b662cd97e8a492436331ec2418a33901a324f4420618076`.

`metalrobo_environment_cook` converted that source as linear Rec.709 into a
512-pixel diffuse/specular cubemap hierarchy and split-sum BRDF resources. The
transient `VisualEnvironmentPackV2` reported:

```text
pack=sha256:3f202a1dd68bd5084ed495066b40d63f412c527ab10da6917011f9943e2100d8
```

## Official Unitree G1 render and sensor gallery

`metalrobo-unitree-g1.webp` and `metalrobo-sensor-gallery.webp` come from one
1600×900 reference frame, downsampled to 1280×720. The gallery presents
scene-linear RGB through an ACES preview, metric depth, camera-space surface
normals, and the stable authored-primitive component of the packed integer
identity buffer. All four panels therefore share the exact camera, exposure,
geometry, and timestamp.

The camera is at `(1.05, -1.30, 1.05)` metres and targets
`(0.0, 0.0, 0.66)`, with focal length equal to image height. Environment
intensity is `0.12` and rotation is `0.0` radians. The source frame rendered
in 183.987 ms and retained 257,807,088 private GPU bytes.

Robot geometry comes from Unitree's official
[`unitree_ros`](https://github.com/unitreerobotics/unitree_ros) source at
commit `aa0f5c68b5aba347bad409e71b6430407da758d7`, specifically
`robots/g1_description/g1_29dof_rev_1_0.urdf` and its referenced STL visual
meshes. The pose uses the named reset preset in `src/core/G1.cpp`. The
official meshes were composed at that pose, transported through GLB without
replacement geometry, cooked into `VisualAssetPackV2`, and rendered through
MetalRobo's native PBR/IBL path. The presentation platform is
MetalRobo-authored.

The transient cooked pack reported:

```text
vertices=1119458
indices=1197030
materials=5
pack=sha256:64db11cc07c176f8e0e706f87a130a26d23ee5e7a4adaa3dea321a312b9c7718
```

## Official Franka FR3v2 render

`metalrobo-franka-fr3v2.webp` is a 1600×900 MetalRobo reference frame,
downsampled to 1280×720. Its camera is at `(0.75, -0.95, 0.72)` metres and
targets `(0.16, 0.0, 0.40)`. Environment intensity is `0.12`, environment
rotation is `0.55` radians, and the preview exposure multiplier is `0.62`.
The source frame rendered in 180.833 ms and retained 172,321,752 private GPU
bytes.

The arm uses the official FR3v2 visual geometry and white Franka hand from
[`frankarobotics/franka_description`](https://github.com/frankarobotics/franka_description)
tag `2.8.1`, commit `02afaae282d4a8e10d7d2f781b23b3515c303ce5`.
The official DAE assets were converted offline with trimesh 4.12.2 and
pycollada 0.9.3, posed using the pinned Franka kinematics, transported through
GLB, and cooked by `metalrobo_visual_cook`. The workcell, task objects, and PBR
material translation are MetalRobo-authored; no replacement robot geometry
was introduced.

The transient cooked pack reported:

```text
vertices=196026
indices=774198
materials=65
pack=sha256:faa6868328514497f67d97a7f01360f84fc6cd2fe08adddb80747dac3b1f7a55
```

The upstream robot meshes and transient cooked packs are not redistributed.
Their licenses and pinned source revisions are recorded in
`THIRD_PARTY_NOTICES.md`.
