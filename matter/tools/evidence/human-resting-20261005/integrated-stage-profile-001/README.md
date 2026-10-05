# Integrated native GPU stage profile

The existing native scene ran 128 accepted 2 ms steps on SSH Mac mini (Apple
M4 Pro). Profile 001 uses the existing articulated and support timestamp
hooks. Profile 002 adds opt-in Matter, respiratory and Brain timestamps on
the same borrowed command buffer; it leaves all physical shaders unchanged.
The terminal q/v record, coupled CSV and geometry CSV were exactly identical
between these runs. Both exited zero with every runtime input unchanged.

At root 100, Matter's vascular solve/continuation took 7.021 ms, respiratory
mechanics 0.058 ms, gas exchange 0.047 ms, and Brain advance/resolve 0.014 ms.
The sampled body stages include a 3.638 ms median final constraint solve,
1.623 ms kinematics and 2.671 ms skin support geometry. Body samples cover
different roots; their median sum is approximate attribution, not an exact
decomposition of one physical step. The larger six-second integrated profile
is retained separately in `accepted-geometry-binding-001`.

This is performance diagnosis, not a 0.256-second speed benchmark or anatomy
qualification. Startup dominates such a short run. Known thoracic/cardiac
geometry issues remain in these old-map inputs. Original logs and recordings
remain on the Mini in `native-integrated-stage-profile-001` and `002` under
`/Users/n/numi-human-resting-evidence-20261005`.

The root candidate source's Anatomy header and cardiac helper had been
modified for independent geometry work after its prior compile. Before the
profile-002 rebuild, both were restored to published `8b68d03` versions,
matching the earlier compiled map. The unrelated local standing source edits
were not included. The immutable runtime hashes are in each invocation.

`NUMI_MATTER_GPU_TIMING=1` samples root 100. Optional
`NUMI_MATTER_GPU_TIMING_STAGE` limits sampling to one named stage. The launcher
JSON records these flags (the older launcher used for this profile did not
yet include them in its own nested environment receipt). Unavailable counters
produce no timing row, never a zero-duration claim.
