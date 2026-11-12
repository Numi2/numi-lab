# Opposite hip-offset screen: interrupted

The eight-lane batch rolled back at native step 25 (status 11, environment 6, reported failing substep 17). Only 25 intervals were accepted. The affected lane used the -0.24 rad setting and positive initial pitch; the other lanes were also stopped by the batch rollback. This is an incomplete physical test and cannot establish long-term outcomes for any setting. The logged contact diagnostics and all partial traces are retained; the exact cause is not established. The runner incorrectly reported its overall process success after its child failed; the child exit code and error have been preserved.

The intervention arms matched their same-pitch baseline for all ten pre-intervention sampled intervals. Repeat smaller offsets without -0.24 in a separate experiment, with failure propagated.
