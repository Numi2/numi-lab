# Failed continuous-wingbeat development replay

`side.mp4` and `front.mp4` render the same accepted simulated states from lane 5 of `crow-v15-continuous-wingbeat-screen-20261113`, development seed 20261126 on the training course. The recorded 400 frames are shown at 50 fps. The 5.6 Hz test had 18 toe-speed threshold substeps and recontact beginning at tick 365 after a three-interval unload at tick 359; this is a failed launch, not flight, navigation or supported landing. It is not either held-out layout. The render follows accepted articulated states through an estimated crow mesh, with camera-relative root framing; it is not photographic or a biological validation.

Both renderer audits report 400 source/rendered frames. `render.sh`, `SHA256SUMS` and renderer commit identify the rendering process and inputs. The full physics trace, replay, source, bounded controller and analysis are separately retained in the associated Git artifact.
