# Backward Euler next smaller timestep

For the same two-division physical law, initial state, material parameters,
10 ms duration and temporal reference as the prior 1.953125 microsecond
backward-Euler trial, test one further halving: 0.9765625 microseconds (10,240
steps). Keep the previously registered force-bin, sampled-peak, total and
cumulative impulse, clearance, local momentum and replay gates unchanged.
Use the same hash-frozen executable and shader as the 1.953125 microsecond
trial. Retain failure or timeout as well as any successful trace. Stop this
single diagnostic at 30 minutes. This instrumented run may overlap CPU
reference evaluation; its elapsed time is not benchmark evidence. A passing
accuracy trial would require a separate uncontended, snapshot-free comparison
and the largest qualifying *tested* step, not an inference of speedup from
this attempt. A failed trial leaves smaller steps unknown.
