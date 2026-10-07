# Measure the complete native presentation path

The existing NUMI_HUMAN_TRAINING_PROFILE switch now records the native renderer command's GPU time, CPU encoding/wait time, accepted-surface audit/export time, and complete presentation wall time. This reads timestamps on the existing completed Metal command and wraps existing native presentation; it adds no GPU command, state history, simulation readback, or physical update.

The 128-step integrated Mac mini regression passes. All 47 coupled columns, 44 COM columns, and 21 support columns match the unchanged baseline at the corresponding prefix, and the actual initial native pack is byte-identical to the previous build. Existing rollback/retry probes also pass.

For the three frames without geometry export, renderer GPU time averages 225.964 ms and the additional presentation wall time averages 4.008 ms. Physics GPU time averages 936.727 ms per 32 steps after the first segment. The 2 ms timestep represents 64 ms per segment. Physics remains the dominant measured cost; these initial-settling samples are not a sustained performance qualification.

Each explicitly requested full geometry export costs about 1.9 seconds in this run. Those exports are retained scientific evidence, not normal per-frame readback. The presentation remainder includes window, HUD and recording work and is not exclusively codec time. CPU-only geometry analysis was active elsewhere on the same Mac, while this run had sole native simulation GPU ownership.

The failed 223-step respiratory geometry witness remains unresolved by this instrumentation. This 128-step test does not imply complete anatomical acceptance or five-minute operation on the current source assets.
