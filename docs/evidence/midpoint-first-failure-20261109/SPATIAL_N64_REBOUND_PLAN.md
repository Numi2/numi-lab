# Hash-bound n=64 reference repeat

The historical n=64 coarse and fine execution records lack their own command
and executable hash. Repeat both 10 ms reference trajectories with the exact
hash-pinned n=128 ordered FP64 evaluator, at n=64, h=0.5 µs and h=0.25 µs.
Keep its physical parameters, geometry, activation width, area coefficient,
initial conditions, and clearance checks unchanged. The source-law audit and
n=2/n=32 ordered parity are retained separately; neither substitutes for
binding the actual executable to each new run.

Do not start while the existing n=128 reference, backward Euler ladder, or
conditional cost study is running. Execute the two repeats sequentially,
record the command, source and binary hashes, output hashes, sampled maximum
resident memory and stopping reason, and fail closed on a 2-hour-per-run wall
limit, 2 GiB resident-memory limit, or less than 3.1 GiB free storage. Stop
if another recognized Numi worker appears. That process-name check is not a
complete measurement of all host activity. These CPU observations are not a
performance benchmark.

Apply the original temporal force/impulse, area-distribution, clearance and
physical-state checks. Compare the new traces with the historical n=64 traces;
exact identity is useful corroboration, while any difference must remain
visible. Recompute n=64-to-128 spatial errors using the new fine trace only if
both meshes pass their separate temporal gates. A failure stays a failure; no
physical gate is relaxed. This repeat does not validate physical materials,
native n=64 Matter, or a continuum limit by itself.
