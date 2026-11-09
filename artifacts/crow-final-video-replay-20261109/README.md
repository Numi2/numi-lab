# Crow navigation: two unused layouts

This side-by-side video shows lane 1 of the reserved layout I and J assessments
from commit `7d32ee3722a1f26c58ad111b46e9edb9ce3b401c`. Both episodes are
continuous, autonomous, teacher-off 60-second simulations. The video samples the
3,000 accepted poses at approximately 2x speed and holds the final frame for
two seconds. Waypoint counts come from the recorded simulation outcomes;
supported landing is displayed only at the verified final state. The surface
is an estimated articulated crow visualization, not biological video.

The native runs were deterministically replayed with the frozen controller,
original seeds, and all 16 lanes. The regenerated lane-1 accepted position and
scene traces match the original archived text traces for all 3,000 steps, and
first-episode outcome fields match. The full native replay packs, verification
reports, compact visualization arrays, and source script are archived in this
directory; `independent-restore.json` verifies both compressed packs restored
from GitHub to their original SHA-256 hashes.

To reproduce, unpack `visualization-inputs.tar.gz` next to `render_replay.py`,
then use Blender 5.1.2, ffmpeg, NumPy, and Pillow to execute:

```
blender -b --python render_replay.py -- --layout i --out layout-i.mp4
blender -b --python render_replay.py -- --layout j --out layout-j.mp4
python3 compose_video.py
```

The renderer attaches the archived estimated BirdFlow crow surface and course
meshes to the recorded native body transforms. Only the camera is smoothed.
The controller uses simulator-provided navigation targets; this does not
establish visual-only navigation or biological flight fidelity.
