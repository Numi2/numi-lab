# Franka surgical workcell scene

This authored presentation scene places an official-mesh Franka at an accepted
Numi Lab pose beside a dimensioned operating table. The visible tissue, curved
needle, and coiled needle/thread presentation come from the local DrAnmar
closure asset set; the remaining room, lighting, instrument tray, and fixtures
are authored procedurally in Blender.

The retained render used DrAnmar revision
`cc731786a75bea57b8e369c1c13d49407675c6a0` and its `SurgicalClosure/Needle`,
`SurgicalClosure/ClosureRobot`, and `SurgicalClosure/NeedleThread` GLB assets.

```sh
blender --background \
  --python examples/franka_surgery_room/render_scene.py -- \
  --poses build/franka-exploration-poses.csv \
  --franka-description build/franka_description \
  --converted-meshes build/franka-render-meshes \
  --surgery-assets-root /path/to/orbit.surgical.assets/data/Props \
  --pose-step 349 \
  --output docs/media/numi-lab-franka-surgery-room.png

blender --background \
  --python examples/franka_surgery_room/render_scene.py -- \
  --poses build/franka-exploration-poses.csv \
  --franka-description build/franka_description \
  --converted-meshes build/franka-render-meshes \
  --surgery-assets-root /path/to/orbit.surgical.assets/data/Props \
  --pose-step 349 \
  --frames build/franka-surgery-room-frames \
  --frame-count 72 --width 960 --height 540

ffmpeg -y -framerate 24 \
  -i build/franka-surgery-room-frames/surgery-%04d.png \
  -vf 'palettegen=max_colors=256:stats_mode=diff' \
  build/franka-surgery-room-palette.png
ffmpeg -y -framerate 24 \
  -i build/franka-surgery-room-frames/surgery-%04d.png \
  -i build/franka-surgery-room-palette.png \
  -lavfi 'paletteuse=dither=sierra2_4a:diff_mode=rectangle' \
  -loop 0 docs/media/numi-lab-franka-surgery-room.gif
```

The scene is visual evidence of authored layout only. The surgical props are
not yet bound to Numi Lab bodies, collision, deformable tissue, rod/suture
physics, or a surgical task contract. It therefore does not claim needle
contact, tissue deformation, suturing, task completion, or hardware execution.
