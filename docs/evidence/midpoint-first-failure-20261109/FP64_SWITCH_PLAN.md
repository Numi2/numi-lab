# Execution switch for the n=64 fine reference

The four-worker evaluator passed its energy self-test and produced byte-for-byte
identical n=2 and n=8 0.25 us outputs and n=32 0.5 us output. The n=32 run
took 127.44 s versus 255.35 s in the retained single-worker run. These are
separately timed executions, not a controlled application benchmark. The
original single-worker n=64 0.5 us run remains in progress and will finish.

Its already-running supervisor automatically attempts the single-worker 0.25 us
run as soon as the first completes. To prevent duplicate CPU work, create an
explicit `not_run` marker at that second run's reserved output path. The
supervisor will preserve the completed first result and stop on its existing
refuse-overwrite check; its nonzero termination and marker are intentional,
not a failed or successful fine reference. Run the validated four-worker
0.25 us reference afterward under a distinct filename with a two-hour bound.
Retain the exact source and executable hashes, calibration outputs, runtime
metadata, supervisor termination and all failures. Never interpret the marker
as data. Apply the original temporal and spatial gates only to the actual two
completed traces. If the fine run fails, retain that failure and do not claim
a spatial result.
