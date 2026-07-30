# README media provenance

The images in this directory are documentation artifacts rendered by
MetalRobo. They are not external simulator screenshots.

## Official Unitree G1 render

`metalrobo-unitree-g1.webp` and `metalrobo-sensor-gallery.webp` were produced
from matching MetalRobo `sensor_reference` frames on Apple M4. The hero was
rendered at 1920×1080 and downsampled to 1280×720 for cleaner presentation.
The gallery uses a 1280×720 frame and contains scene-linear RGB presented
through an ACES preview transform, metric depth, camera-space surface normals,
and packed instance identities. Both frames use the renderer's stratified
eight-sample space-time integration and show the authored HDR environment
through the calibrated camera.

The hero camera is at `(1.22, -1.45, 1.15)` metres and targets
`(0.0, 0.0, 0.62)`, with focal length equal to image height. Environment
intensity is `0.55` and rotation is `0.35` radians. The 1920×1080 source frame
rendered in 254 ms and retained 307,224,368 private GPU bytes on Apple M4.

Robot geometry comes from Unitree's official
[`unitree_ros`](https://github.com/unitreerobotics/unitree_ros) source at
commit `aa0f5c68b5aba347bad409e71b6430407da758d7`, specifically
`robots/g1_description/g1_29dof_rev_1_0.urdf` and its referenced STL visual
meshes. The pose uses the named reset preset retained in `src/core/G1.cpp`.
The robot meshes were composed at that pose, transported as GLB without
replacement geometry, cooked into `VisualAssetPackV2`, and rendered through
MetalRobo's native PBR/IBL path. The presentation platform and lighting are
MetalRobo-authored.

The transient cooked pack reported:

```text
vertices=1119458
indices=1197030
materials=5
pack=sha256:64db11cc07c176f8e0e706f87a130a26d23ee5e7a4adaa3dea321a312b9c7718
```

The upstream mesh files and transient pack are not redistributed in this
repository. Unitree's BSD-3-Clause notice is reproduced in
`THIRD_PARTY_NOTICES.md`.

## Dual PSM research scene

`metalrobo-dual-psm.webp` is a MetalRobo reference render of the authored dual
PSM needle-and-thread presentation scene. The presentation geometry is
MetalRobo-authored; it does not redistribute ORBIT-Surgical or JHU mesh
assets. Physical model provenance and the boundary between pinned source
facts and MetalRobo research defaults are documented in
`THIRD_PARTY_NOTICES.md`.
